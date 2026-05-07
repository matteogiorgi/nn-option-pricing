"""End-to-end experiment orchestration.

The pipeline connects all project components: synthetic data generation,
supervised training, neural network evaluation, Monte Carlo benchmarking,
artifact persistence, and diagnostic plots.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from option_pricing_nn.config import ExperimentConfig
from option_pricing_nn.dataset import (
    FEATURE_COLUMNS,
    generate_synthetic_dataset,
    save_dataset,
)
from option_pricing_nn.evaluation import regression_metrics, save_metrics
from option_pricing_nn.monte_carlo import call_price_mc
from option_pricing_nn.plots import (
    plot_error_against_feature,
    plot_error_distribution,
    plot_loss,
    plot_true_vs_predicted,
)
from option_pricing_nn.training import predict, train_model


def ensure_directories(config: ExperimentConfig) -> None:
    """Create all directories required by the experiment outputs."""
    for path in [
        config.paths.data_dir,
        config.paths.output_dir,
        config.paths.figure_dir,
        config.paths.metrics_dir,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    """Run the full Black-Scholes neural pricing experiment.

    Parameters
    ----------
    config
        Complete experiment configuration.

    Returns
    -------
    dict[str, float]
        Neural network regression metrics computed against analytical
        Black-Scholes prices on the held-out test set.
    """
    ensure_directories(config)

    # The analytical Black-Scholes formula is the label generator. This makes
    # the supervised task a controlled function approximation problem.
    df = generate_synthetic_dataset(config.dataset)
    save_dataset(df, config.paths.dataset_path)

    artifacts = train_model(df, config.training)
    y_pred = predict(
        artifacts.model,
        artifacts.x_test,
        target_scaler=artifacts.target_scaler,
        batch_size=config.training.batch_size,
    )
    metrics = regression_metrics(artifacts.y_test, y_pred)

    torch.save(
        {
            "model_state_dict": artifacts.model.state_dict(),
            "hidden_layers": config.training.hidden_layers,
            "feature_columns": FEATURE_COLUMNS,
        },
        config.paths.model_path,
    )
    joblib.dump(artifacts.scaler, config.paths.scaler_path)
    joblib.dump(
        artifacts.target_scaler, config.paths.output_dir / "target_scaler.joblib"
    )
    save_metrics(metrics, config.paths.metrics_dir / "nn_metrics.json")

    # Reconstruct original feature values for diagnostic plots and for the
    # Monte Carlo benchmark, which should operate on financial units rather
    # than standardized inputs.
    test_df = pd.DataFrame(
        artifacts.scaler.inverse_transform(artifacts.x_test), columns=FEATURE_COLUMNS
    )
    errors = y_pred - artifacts.y_test

    # Monte Carlo is more expensive than a neural-network forward pass. We use
    # a deterministic subset of the test set to keep the benchmark informative
    # but computationally bounded.
    mc_n = min(config.monte_carlo.evaluation_samples, len(test_df))
    mc_indices = np.linspace(0, len(test_df) - 1, num=mc_n, dtype=int)
    mc_df = test_df.iloc[mc_indices]
    mc_prices = call_price_mc(
        mc_df["s0"].to_numpy(),
        mc_df["k"].to_numpy(),
        mc_df["t"].to_numpy(),
        mc_df["r"].to_numpy(),
        mc_df["sigma"].to_numpy(),
        n_paths=config.monte_carlo.n_paths,
        option_batch_size=config.monte_carlo.option_batch_size,
        path_batch_size=config.monte_carlo.path_batch_size,
        seed=config.monte_carlo.seed,
    )
    mc_metrics = regression_metrics(artifacts.y_test[mc_indices], mc_prices)
    save_metrics(
        mc_metrics,
        config.paths.metrics_dir / "monte_carlo_vs_black_scholes_metrics.json",
    )

    plot_loss(artifacts.history, config.paths.figure_dir / "loss.png")
    plot_true_vs_predicted(
        artifacts.y_test, y_pred, config.paths.figure_dir / "true_vs_predicted.png"
    )
    plot_error_distribution(errors, config.paths.figure_dir / "error_distribution.png")
    plot_error_against_feature(
        test_df["s0"].to_numpy() / test_df["k"].to_numpy(),
        errors,
        "Moneyness S0/K",
        config.paths.figure_dir / "error_vs_moneyness.png",
    )
    plot_error_against_feature(
        test_df["t"].to_numpy(),
        errors,
        "Maturity T",
        config.paths.figure_dir / "error_vs_maturity.png",
    )
    plot_error_against_feature(
        test_df["sigma"].to_numpy(),
        errors,
        "Volatility sigma",
        config.paths.figure_dir / "error_vs_volatility.png",
    )

    return metrics
