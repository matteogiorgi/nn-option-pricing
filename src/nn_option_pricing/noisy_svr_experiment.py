"""Support Vector Regression benchmark with noisy Black-Scholes targets."""

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
from nn_option_pricing.noise import (
    CLEAN_TARGET_COLUMN,
    NOISY_TARGET_COLUMN,
    PRICE_NOISE_COLUMN,
    NoiseConfig,
    add_relative_gaussian_price_noise,
)


@dataclass(frozen=True)
class NoisySVRBenchmarkConfig:
    """Configuration for reduced-scale noisy-target SVR experiments."""

    n_samples: int = 5_000
    seeds: tuple[int, ...] = (11, 42, 73)
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05)
    feature_set: str = "with_moneyness"
    test_size: float = 0.2
    c: float = 100.0
    epsilon: float = 0.01
    gamma: float | Literal["scale", "auto"] = "scale"
    price_floor: float = 1.0
    noise_seed_offset: int = 10_000


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return mean and sample standard deviation for repeated runs."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def _noise_summary(noisy_df) -> dict[str, float]:
    """Compute descriptive statistics for the applied target noise."""
    clean_prices = noisy_df[CLEAN_TARGET_COLUMN].to_numpy(dtype=np.float64)
    noisy_prices = noisy_df[NOISY_TARGET_COLUMN].to_numpy(dtype=np.float64)
    price_noise = noisy_df[PRICE_NOISE_COLUMN].to_numpy(dtype=np.float64)
    clipped_mask = (noisy_prices == 0.0) & (clean_prices > 0.0)
    return {
        "mean_noise": float(np.mean(price_noise)),
        "mean_abs_noise": float(np.mean(np.abs(price_noise))),
        "std_noise": float(np.std(price_noise)),
        "clipped_fraction": float(np.mean(clipped_mask)),
    }


def run_single_noisy_svr_benchmark(
    seed: int,
    noise_level: float,
    config: NoisySVRBenchmarkConfig,
) -> dict[str, Any]:
    """Train SVR on one noisy target dataset and evaluate clean/noisy metrics."""
    clean_df = generate_synthetic_dataset(
        DatasetConfig(n_samples=config.n_samples, seed=seed)
    )
    noisy_df = add_relative_gaussian_price_noise(
        clean_df,
        NoiseConfig(
            level=noise_level,
            seed=config.noise_seed_offset + seed,
            price_floor=config.price_floor,
        ),
    )
    feature_columns = get_feature_columns(config.feature_set)

    x = noisy_df[feature_columns].to_numpy(dtype=np.float64)
    y_noisy = noisy_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    y_clean = noisy_df[CLEAN_TARGET_COLUMN].to_numpy(dtype=np.float64)
    x_train, x_test, y_noisy_train, y_noisy_test, _y_clean_train, y_clean_test = (
        train_test_split(
            x,
            y_noisy,
            y_clean,
            test_size=config.test_size,
            random_state=seed,
        )
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_noisy_train_scaled = y_scaler.fit_transform(
        y_noisy_train.reshape(-1, 1)
    ).ravel()

    model = SVR(C=config.c, epsilon=config.epsilon, gamma=config.gamma, kernel="rbf")

    fit_start = time.perf_counter()
    model.fit(x_train_scaled, y_noisy_train_scaled)
    fit_time = time.perf_counter() - fit_start

    prediction_start = time.perf_counter()
    y_pred_scaled = model.predict(x_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.maximum(y_pred, 0.0)
    prediction_time = time.perf_counter() - prediction_start

    metrics_vs_clean = regression_metrics(y_clean_test, y_pred)
    metrics_vs_noisy = regression_metrics(y_noisy_test, y_pred)

    return {
        "seed": seed,
        "noise_level": float(noise_level),
        "n_samples": config.n_samples,
        "test_samples": int(len(y_clean_test)),
        "metrics_vs_clean": metrics_vs_clean,
        "metrics_vs_noisy": metrics_vs_noisy,
        "noise_summary": _noise_summary(noisy_df),
        "fit_time_seconds": float(fit_time),
        "prediction_time_seconds": float(prediction_time),
        "total_time_seconds": float(fit_time + prediction_time),
    }


def summarize_noisy_svr_runs(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate noisy SVR metrics by noise level."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = ["mae", "rmse", "r2", "mape_percent_price_gt_1"]
    time_names = [
        "fit_time_seconds",
        "prediction_time_seconds",
        "total_time_seconds",
    ]
    noise_levels = sorted({float(run["noise_level"]) for run in runs})

    for level in noise_levels:
        level_runs = [run for run in runs if float(run["noise_level"]) == level]
        level_key = f"{level:.2f}"
        summary[level_key] = {}
        for metric in metric_names:
            summary[level_key][f"clean_{metric}"] = _mean_std(
                [float(run["metrics_vs_clean"][metric]) for run in level_runs]
            )
            summary[level_key][f"noisy_{metric}"] = _mean_std(
                [float(run["metrics_vs_noisy"][metric]) for run in level_runs]
            )
        for metric in time_names:
            summary[level_key][metric] = _mean_std(
                [float(run[metric]) for run in level_runs]
            )

    return summary


def run_noisy_svr_benchmark(
    config: NoisySVRBenchmarkConfig,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the noisy-target SVR benchmark over all levels and seeds."""
    runs = [
        run_single_noisy_svr_benchmark(seed=seed, noise_level=level, config=config)
        for level in config.noise_levels
        for seed in config.seeds
    ]
    result = {
        "config": asdict(config),
        "runs": runs,
        "summary": summarize_noisy_svr_runs(runs),
    }

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "noisy_svr_benchmark.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

    return result
