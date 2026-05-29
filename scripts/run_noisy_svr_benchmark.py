"""Command-line entry point for the noisy-target SVR benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from nn_option_pricing.noisy_svr_experiment import (
    NoisySVRBenchmarkConfig,
    run_noisy_svr_benchmark,
)


def parse_gamma(value: str) -> float | Literal["scale", "auto"]:
    """Parse an SVR gamma value from the command line."""
    if value == "scale" or value == "auto":
        return value
    return float(value)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the noisy-target SVR benchmark."""
    parser = argparse.ArgumentParser(
        description="Run a reduced-scale SVR benchmark with noisy targets."
    )
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 73])
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05],
    )
    parser.add_argument(
        "--feature-set",
        choices=["base", "with_moneyness"],
        default="with_moneyness",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=100.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=parse_gamma, default="scale")
    parser.add_argument("--price-floor", type=float, default=1.0)
    parser.add_argument("--noise-seed-offset", type=int, default=10_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/noisy_svr_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    """Build the noisy-target SVR configuration, run it, and print results."""
    args = parse_args()
    config = NoisySVRBenchmarkConfig(
        n_samples=args.n_samples,
        seeds=tuple(args.seeds),
        noise_levels=tuple(args.noise_levels),
        feature_set=args.feature_set,
        test_size=args.test_size,
        c=args.c,
        epsilon=args.epsilon,
        gamma=args.gamma,
        price_floor=args.price_floor,
        noise_seed_offset=args.noise_seed_offset,
    )
    result = run_noisy_svr_benchmark(config=config, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
