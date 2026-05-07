"""Command-line entry point for running the pricing experiment."""

from __future__ import annotations

import argparse
import json

from nn_option_pricing.config import (
    DatasetConfig,
    ExperimentConfig,
    MonteCarloConfig,
    TrainingConfig,
)
from nn_option_pricing.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line options into experiment overrides."""
    parser = argparse.ArgumentParser(
        description="Run the Black-Scholes neural pricing experiment."
    )
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mc-n-paths", type=int, default=50_000)
    parser.add_argument("--mc-evaluation-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build the experiment configuration, run the pipeline, and print metrics."""
    args = parse_args()
    config = ExperimentConfig(
        dataset=DatasetConfig(n_samples=args.n_samples, seed=args.seed),
        training=TrainingConfig(
            seed=args.seed,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
        monte_carlo=MonteCarloConfig(
            n_paths=args.mc_n_paths,
            evaluation_samples=args.mc_evaluation_samples,
        ),
    )
    metrics = run_experiment(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
