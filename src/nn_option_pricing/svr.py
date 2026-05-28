"""Support Vector Regression benchmark utilities.

The neural network is the main surrogate model in the project. This module
provides a reduced-scale Support Vector Regression baseline, useful as a
classical machine-learning comparison without changing the Black-Scholes
function-approximation research question.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from nn_option_pricing.config import DatasetConfig
from nn_option_pricing.dataset import (
    TARGET_COLUMN,
    generate_synthetic_dataset,
    get_feature_columns,
)
from nn_option_pricing.evaluation import regression_metrics


@dataclass(frozen=True)
class SVRBenchmarkConfig:
    """Configuration for the reduced-scale SVR benchmark."""

    n_samples: int = 5_000
    seeds: tuple[int, ...] = (11, 42, 73)
    feature_set: str = "with_moneyness"
    test_size: float = 0.2
    c: float = 100.0
    epsilon: float = 0.01
    gamma: float | Literal["scale", "auto"] = "scale"


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return mean and sample standard deviation for repeated runs."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def summarize_svr_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed SVR benchmark metrics into mean/std summaries."""
    metric_names = [
        "mae",
        "rmse",
        "r2",
        "mape_percent_price_gt_1",
        "fit_time_seconds",
        "prediction_time_seconds",
        "total_time_seconds",
    ]
    return {
        metric: _mean_std([float(run[metric]) for run in runs])
        for metric in metric_names
    }


def run_single_svr_benchmark(
    seed: int,
    config: SVRBenchmarkConfig,
) -> dict[str, float | int]:
    """Train and evaluate one SVR benchmark run for a single random seed."""
    dataset_config = DatasetConfig(n_samples=config.n_samples, seed=seed)
    df = generate_synthetic_dataset(dataset_config)
    feature_columns = get_feature_columns(config.feature_set)

    x = df[feature_columns].to_numpy(dtype=np.float64)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = SVR(C=config.c, epsilon=config.epsilon, gamma=config.gamma, kernel="rbf")

    fit_start = time.perf_counter()
    model.fit(x_train_scaled, y_train_scaled)
    fit_time = time.perf_counter() - fit_start

    prediction_start = time.perf_counter()
    y_pred_scaled = model.predict(x_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.maximum(y_pred, 0.0)
    prediction_time = time.perf_counter() - prediction_start

    metrics = regression_metrics(y_test, y_pred)
    return {
        "seed": seed,
        "n_samples": config.n_samples,
        "test_samples": int(len(y_test)),
        **metrics,
        "fit_time_seconds": float(fit_time),
        "prediction_time_seconds": float(prediction_time),
        "total_time_seconds": float(fit_time + prediction_time),
    }


def run_svr_benchmark(
    config: SVRBenchmarkConfig,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the SVR benchmark over all configured seeds.

    Parameters
    ----------
    config
        Reduced-scale SVR benchmark configuration.
    output_dir
        Optional directory where per-seed and aggregate metrics are written.

    Returns
    -------
    dict[str, Any]
        Configuration, per-seed runs, and mean/std metric summary.
    """
    runs = [run_single_svr_benchmark(seed=seed, config=config) for seed in config.seeds]
    result = {
        "config": asdict(config),
        "runs": runs,
        "summary": summarize_svr_runs(runs),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "svr_benchmark.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    return result
