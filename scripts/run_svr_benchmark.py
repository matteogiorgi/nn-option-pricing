"""Command-line entry point for the reduced-scale SVR benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from nn_option_pricing.svr import SVRBenchmarkConfig, run_svr_benchmark


def parse_gamma(value: str) -> float | Literal["scale", "auto"]:
    """Parse an SVR gamma value from the command line."""
    if value == "scale" or value == "auto":
        return value
    return float(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the SVR benchmark."""
    parser = argparse.ArgumentParser(
        description="Run a reduced-scale Support Vector Regression benchmark."
    )
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument(
        "--feature-set",
        choices=["base", "with_moneyness"],
        default="with_moneyness",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=parse_gamma, default="scale")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/svr_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    """Build the SVR benchmark configuration, run it, and print results."""
    args = parse_args()
    config = SVRBenchmarkConfig(
        n_samples=args.n_samples,
        seeds=tuple(args.seeds),
        feature_set=args.feature_set,
        test_size=args.test_size,
        c=args.c,
        epsilon=args.epsilon,
        gamma=args.gamma,
    )
    result = run_svr_benchmark(config=config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
