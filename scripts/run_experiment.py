"""Command-line entry point for running the pricing experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nn_option_pricing.config import (
    DatasetConfig,
    ExperimentConfig,
    MonteCarloConfig,
    TrainingConfig,
    make_path_config,
)
from nn_option_pricing.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    """Parse command-line options into experiment overrides."""

    # first thing is to create a parser object from the argparse module
    # and then we can add arguments to the parser object that we want to accept from
    parser = argparse.ArgumentParser(
        description="Run the Black-Scholes neural pricing experiment."
    )
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument(
        "--feature-set",
        choices=["base", "with_moneyness"],
        default="base",
        help="Input feature set used by the neural network.",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "leaky_relu", "silu", "gelu"],
        default="relu",
        help="Hidden-layer activation function.",
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mc-n-paths", type=int, default=50_000)
    parser.add_argument("--mc-evaluation-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))

    # now with parse_args method from the parser object we can parse
    # the command line arguments and return them as a Namespace object
    return parser.parse_args()


def main() -> None:
    """Build the experiment configuration, run the pipeline, and print metrics."""

    # first we parse the command line arguments
    args = parse_args()

    # then we create an ExperimentConfig object that contains
    # all the settings for the experiment
    config = ExperimentConfig(
        dataset=DatasetConfig(n_samples=args.n_samples, seed=args.seed),
        training=TrainingConfig(
            seed=args.seed,
            feature_set=args.feature_set,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            activation=args.activation,
        ),
        monte_carlo=MonteCarloConfig(
            n_paths=args.mc_n_paths,
            evaluation_samples=args.mc_evaluation_samples,
        ),
        paths=make_path_config(data_dir=args.data_dir, output_dir=args.output_dir),
    )

    # now we can run the experiment pipeline with the given configuration
    # and print the resulting metrics in a nicely formatted JSON structure
    metrics = run_experiment(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
