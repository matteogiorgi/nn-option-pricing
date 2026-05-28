"""Noisy Black-Scholes target robustness experiment."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nn_option_pricing.config import DatasetConfig, TrainingConfig, config_to_dict
from nn_option_pricing.dataset import generate_synthetic_dataset, save_dataset
from nn_option_pricing.evaluation import regression_metrics, save_metrics
from nn_option_pricing.noise import (
    CLEAN_TARGET_COLUMN,
    NOISY_TARGET_COLUMN,
    PRICE_NOISE_COLUMN,
    NoiseConfig,
    add_relative_gaussian_price_noise,
    noise_level_slug,
)
from nn_option_pricing.training import predict, train_model


@dataclass(frozen=True)
class NoisyTargetsExperimentConfig:
    """Configuration for the controlled noisy-target robustness experiment."""

    dataset: DatasetConfig = DatasetConfig(n_samples=50_000, seed=42)
    training: TrainingConfig = TrainingConfig(
        seed=42,
        feature_set="with_moneyness",
        max_epochs=100,
        batch_size=1024,
        activation="silu",
    )
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.05)
    noise_seed: int = 123
    price_floor: float = 1.0
    data_dir: Path = Path("data/experiments/noisy_targets")
    output_dir: Path = Path("outputs/experiments/noisy_targets")
    results_dir: Path = Path("results/experiments/noisy_targets")


def _noise_summary(noisy_df: pd.DataFrame) -> dict[str, float]:
    """Compute descriptive statistics for the applied price noise."""
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


def run_single_noisy_target_experiment(
    clean_df: pd.DataFrame,
    noise_level: float,
    config: NoisyTargetsExperimentConfig,
) -> dict[str, Any]:
    """Train on one noisy target level and evaluate against clean prices."""
    slug = noise_level_slug(noise_level)
    noise_config = NoiseConfig(
        level=noise_level,
        seed=config.noise_seed,
        price_floor=config.price_floor,
    )
    noisy_df = add_relative_gaussian_price_noise(clean_df, noise_config)

    level_data_dir = config.data_dir / slug
    level_output_dir = config.output_dir / slug
    level_metrics_dir = level_output_dir / "metrics"
    save_dataset(noisy_df, level_data_dir / "synthetic_options.csv")

    start_time = time.perf_counter()
    artifacts = train_model(noisy_df, config.training)
    training_time = time.perf_counter() - start_time

    y_pred = predict(
        artifacts.model,
        artifacts.x_test,
        target_scaler=artifacts.target_scaler,
        batch_size=config.training.batch_size,
    )
    y_clean_test = noisy_df.iloc[artifacts.test_indices][
        CLEAN_TARGET_COLUMN
    ].to_numpy(dtype=np.float64)

    metrics_vs_clean = regression_metrics(y_clean_test, y_pred)
    metrics_vs_noisy = regression_metrics(artifacts.y_test, y_pred)

    level_metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics_vs_clean, level_metrics_dir / "metrics_vs_clean.json")
    save_metrics(metrics_vs_noisy, level_metrics_dir / "metrics_vs_noisy.json")

    return {
        "noise_level": float(noise_level),
        "slug": slug,
        "dataset_path": str(level_data_dir / "synthetic_options.csv"),
        "metrics_vs_clean": metrics_vs_clean,
        "metrics_vs_noisy": metrics_vs_noisy,
        "noise_summary": _noise_summary(noisy_df),
        "training_time_seconds": float(training_time),
        "epochs_ran": len(artifacts.history["train_loss"]),
    }


def run_noisy_targets_experiment(
    config: NoisyTargetsExperimentConfig,
) -> dict[str, Any]:
    """Run the noisy-target robustness experiment for all noise levels."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    clean_df = generate_synthetic_dataset(config.dataset)
    runs = [
        run_single_noisy_target_experiment(clean_df, level, config)
        for level in config.noise_levels
    ]

    result = {
        "config": config_to_dict(config),
        "runs": runs,
    }
    (config.results_dir / "noisy_targets_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result
