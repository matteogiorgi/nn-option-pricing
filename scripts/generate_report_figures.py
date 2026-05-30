"""Generate report-specific comparison figures from saved experiment results."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_FIGURE_DIR = ROOT / "report" / "figures"

PURPLE = "#6A1B9A"
ORANGE = "#D97706"
GRAY = "#4B5563"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact produced by an experiment script."""
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def save_figure(path: Path) -> None:
    """Save the active Matplotlib figure using report-friendly defaults."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def style_axes(ax: plt.Axes) -> None:
    """Apply a clean visual style suitable for the LaTeX report."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_nn_vs_monte_carlo() -> None:
    """Plot final neural-network errors against the Monte Carlo benchmark."""
    nn = load_json(ROOT / "results" / "final" / "metrics" / "nn_metrics.json")
    mc = load_json(
        ROOT / "results" / "final" / "metrics" / "monte_carlo_vs_black_scholes_metrics.json"
    )

    metrics = ["mae", "rmse", "mape_percent_price_gt_1"]
    labels = ["MAE", "RMSE", "MAPE (%)"]
    nn_values = [nn[metric] for metric in metrics]
    mc_values = [mc[metric] for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.36

    _, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, nn_values, width, label="Neural network", color=PURPLE)
    ax.bar(x + width / 2, mc_values, width, label="Monte Carlo", color=ORANGE)

    ax.set_xticks(x, labels)
    ax.set_ylabel("Error")
    ax.set_title("Final Error Metrics")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(REPORT_FIGURE_DIR / "nn_vs_monte_carlo_metrics.png")


def plot_runtime_comparison() -> None:
    """Plot pricing runtime comparison for the final surrogate discussion."""
    runtime = load_json(
        ROOT / "results" / "experiments" / "runtime_benchmark" / "runtime_benchmark.json"
    )
    pricing = runtime["pricing"]

    labels = ["Black-Scholes", "Neural network", "Monte Carlo"]
    values = [
        pricing["black_scholes"]["mean_seconds"],
        pricing["neural_network_inference"]["mean_seconds"],
        pricing["monte_carlo"]["mean_seconds"],
    ]
    colors = [GRAY, PURPLE, ORANGE]

    _, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Mean seconds for 1,000 options (log scale)")
    ax.set_title("Pricing Runtime Comparison")
    style_axes(ax)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.15,
            f"{value:.4f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(REPORT_FIGURE_DIR / "runtime_comparison.png")


def plot_noisy_targets() -> None:
    """Plot neural-network and SVR robustness under controlled label noise."""
    noisy_nn = load_json(
        ROOT / "results" / "experiments" / "noisy_targets" / "noisy_targets_metrics.json"
    )
    noisy_svr = load_json(
        ROOT
        / "results"
        / "experiments"
        / "noisy_svr_benchmark"
        / "noisy_svr_benchmark.json"
    )

    noise_levels = [run["noise_level"] * 100 for run in noisy_nn["runs"]]
    nn_mae = [run["metrics_vs_clean"]["mae"] for run in noisy_nn["runs"]]
    svr_summary = noisy_svr["summary"]
    svr_mae = [svr_summary[f"{level / 100:.2f}"]["clean_mae"]["mean"] for level in noise_levels]

    _, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(noise_levels, nn_mae, marker="o", linewidth=2, color=PURPLE, label="Neural network")
    ax.plot(noise_levels, svr_mae, marker="s", linewidth=2, color=ORANGE, label="SVR")

    ax.set_xticks(noise_levels, [f"{level:.0f}%" for level in noise_levels])
    ax.set_xlabel("Label noise level")
    ax.set_ylabel("MAE against clean Black-Scholes")
    ax.set_title("Noisy-Target Robustness")
    ax.legend(frameon=False)
    style_axes(ax)
    save_figure(REPORT_FIGURE_DIR / "noisy_target_robustness.png")


def main() -> None:
    """Generate all report-level figures."""
    plot_nn_vs_monte_carlo()
    plot_runtime_comparison()
    plot_noisy_targets()


if __name__ == "__main__":
    main()
