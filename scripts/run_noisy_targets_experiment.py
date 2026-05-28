"""Command-line entry point for noisy Black-Scholes target experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_option_pricing.config import DatasetConfig, TrainingConfig
from nn_option_pricing.noisy_experiment import (
    NoisyTargetsExperimentConfig,
    run_noisy_targets_experiment,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the noisy-target experiment."""
    parser = argparse.ArgumentParser(
        description="Run neural-network robustness experiments with noisy targets."
    )
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=123)
    parser.add_argument("--price-floor", type=float, default=1.0)
    parser.add_argument(
        "--feature-set",
        choices=["base", "with_moneyness"],
        default="with_moneyness",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "leaky_relu", "silu", "gelu"],
        default="silu",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/experiments/noisy_targets"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/noisy_targets"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/experiments/noisy_targets"),
    )
    return parser.parse_args()


def main() -> None:
    """Build the noisy-target configuration, run it, and print results."""
    args = parse_args()
    config = NoisyTargetsExperimentConfig(
        dataset=DatasetConfig(n_samples=args.n_samples, seed=args.seed),
        training=TrainingConfig(
            seed=args.seed,
            feature_set=args.feature_set,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            activation=args.activation,
        ),
        noise_levels=tuple(args.noise_levels),
        noise_seed=args.noise_seed,
        price_floor=args.price_floor,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )
    result = run_noisy_targets_experiment(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
