from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_loss(history: dict[str, list[float]], path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training and Validation Loss")
    plt.legend()
    _save_current(path)


def plot_true_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.35)
    low = min(float(np.min(y_true)), float(np.min(y_pred)))
    high = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel("Black-Scholes price")
    plt.ylabel("Neural network price")
    plt.title("True Price vs Predicted Price")
    _save_current(path)


def plot_error_distribution(errors: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(errors, bins=80, alpha=0.85)
    plt.xlabel("Prediction error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    _save_current(path)


def plot_error_against_feature(feature: np.ndarray, errors: np.ndarray, feature_name: str, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter(feature, np.abs(errors), s=6, alpha=0.3)
    plt.xlabel(feature_name)
    plt.ylabel("Absolute error")
    plt.title(f"Absolute Error vs {feature_name}")
    _save_current(path)
