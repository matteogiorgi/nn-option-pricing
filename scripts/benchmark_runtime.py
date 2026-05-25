"""Benchmark pricing runtimes across analytical, neural, and Monte Carlo methods."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from nn_option_pricing.black_scholes import call_price
from nn_option_pricing.config import DatasetConfig, MonteCarloConfig, TrainingConfig
from nn_option_pricing.dataset import generate_synthetic_dataset
from nn_option_pricing.monte_carlo import call_price_mc
from nn_option_pricing.training import predict, train_model

T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark Black-Scholes, neural inference, and Monte Carlo runtimes."
    )
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--n-options", type=int, default=5_000)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--feature-set", choices=["base", "with_moneyness"], default="base")
    parser.add_argument(
        "--activation",
        choices=["relu", "tanh", "leaky_relu", "silu", "gelu"],
        default="relu",
    )
    parser.add_argument("--mc-n-paths", type=int, default=20_000)
    parser.add_argument("--mc-option-batch-size", type=int, default=512)
    parser.add_argument("--mc-path-batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mc-repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runtime_benchmark"))
    return parser.parse_args()


def timed_call(fn: Callable[[], T]) -> tuple[T, float]:
    """Run a callable once and return its result with elapsed wall-clock seconds."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def repeated_timings(fn: Callable[[], T], repeats: int) -> tuple[T, list[float]]:
    """Measure repeated calls and return the final result plus all elapsed times."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    result, elapsed = timed_call(fn)
    times = [elapsed]
    for _ in range(1, repeats):
        result, elapsed = timed_call(fn)
        times.append(elapsed)
    return result, times


def summarize_times(times: list[float], n_options: int) -> dict[str, float | list[float]]:
    """Summarize elapsed times and throughput for a pricing method."""
    arr = np.asarray(times, dtype=np.float64)
    mean_seconds = float(np.mean(arr))
    median_seconds = float(np.median(arr))
    return {
        "times_seconds": [float(value) for value in arr],
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
        "options_per_second_mean": float(n_options / mean_seconds),
        "options_per_second_median": float(n_options / median_seconds),
    }


def main() -> None:
    """Run the runtime benchmark and save a JSON summary."""
    args = parse_args()

    dataset_config = DatasetConfig(n_samples=args.n_samples, seed=args.seed)
    training_config = TrainingConfig(
        seed=args.seed,
        feature_set=args.feature_set,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        activation=args.activation,
    )
    monte_carlo_config = MonteCarloConfig(
        n_paths=args.mc_n_paths,
        option_batch_size=args.mc_option_batch_size,
        path_batch_size=args.mc_path_batch_size,
        seed=args.mc_seed,
    )

    df = generate_synthetic_dataset(dataset_config)
    artifacts, training_seconds = timed_call(lambda: train_model(df, training_config))

    n_options = min(args.n_options, len(artifacts.x_test))
    x_scaled = artifacts.x_test[:n_options]
    original_features = pd.DataFrame(
        artifacts.scaler.inverse_transform(x_scaled),
        columns=artifacts.feature_columns,
    )

    s0 = original_features["s0"].to_numpy()
    k = original_features["k"].to_numpy()
    t = original_features["t"].to_numpy()
    r = original_features["r"].to_numpy()
    sigma = original_features["sigma"].to_numpy()

    # A warmup forward pass avoids counting one-off PyTorch setup overhead in
    # the repeated inference timings.
    predict(
        artifacts.model,
        x_scaled[: min(len(x_scaled), args.batch_size)],
        target_scaler=artifacts.target_scaler,
        batch_size=args.batch_size,
    )

    _, bs_times = repeated_timings(
        lambda: call_price(s0, k, t, r, sigma),
        repeats=args.repeats,
    )
    _, nn_times = repeated_timings(
        lambda: predict(
            artifacts.model,
            x_scaled,
            target_scaler=artifacts.target_scaler,
            batch_size=args.batch_size,
        ),
        repeats=args.repeats,
    )
    _, mc_times = repeated_timings(
        lambda: call_price_mc(
            s0,
            k,
            t,
            r,
            sigma,
            n_paths=monte_carlo_config.n_paths,
            option_batch_size=monte_carlo_config.option_batch_size,
            path_batch_size=monte_carlo_config.path_batch_size,
            seed=monte_carlo_config.seed,
        ),
        repeats=args.mc_repeats,
    )

    results = {
        "config": {
            "n_samples": args.n_samples,
            "n_options": n_options,
            "feature_set": args.feature_set,
            "activation": args.activation,
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "repeats": args.repeats,
            "mc_repeats": args.mc_repeats,
            "mc_n_paths": monte_carlo_config.n_paths,
            "mc_option_batch_size": monte_carlo_config.option_batch_size,
            "mc_path_batch_size": monte_carlo_config.path_batch_size,
            "mc_seed": monte_carlo_config.seed,
        },
        "training": {
            "seconds": training_seconds,
            "epochs_ran": len(artifacts.history["train_loss"]),
        },
        "pricing": {
            "black_scholes": summarize_times(bs_times, n_options),
            "neural_network_inference": summarize_times(nn_times, n_options),
            "monte_carlo": summarize_times(mc_times, n_options),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "runtime_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
