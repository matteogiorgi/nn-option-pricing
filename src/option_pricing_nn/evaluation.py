"""Evaluation metrics and persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics for option price predictions.

    Parameters
    ----------
    y_true
        Reference prices, typically analytical Black-Scholes prices.
    y_pred
        Model or Monte Carlo estimates to evaluate.

    Returns
    -------
    dict[str, float]
        MAE, RMSE, R-squared, and MAPE on prices greater than 1.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Percentage errors explode for near-zero option prices. Restricting MAPE
    # to prices greater than 1 keeps the metric interpretable while MAE/RMSE
    # still evaluate the full test set.
    mask = np.abs(y_true) > 1.0
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        if np.any(mask)
        else float("nan")
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape_percent_price_gt_1": mape,
    }


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Save metrics as a pretty-printed JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
