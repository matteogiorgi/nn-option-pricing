"""Synthetic dataset generation for the supervised pricing task.

The dataset is synthetic by design: option parameters are sampled from
predefined ranges and labels are computed with the analytical Black-Scholes
formula. This keeps the learning problem controlled and makes evaluation
unambiguous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nn_option_pricing.black_scholes import call_price
from nn_option_pricing.config import DatasetConfig

BASE_FEATURE_COLUMNS = ["s0", "k", "t", "r", "sigma"]
MONEYNESS_COLUMN = "moneyness"
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
TARGET_COLUMN = "call_price"


def get_feature_columns(feature_set: str = "base") -> list[str]:
    """Return the model input columns associated with a named feature set.

    Parameters
    ----------
    feature_set
        Feature set identifier. ``"base"`` uses only the primitive
        Black-Scholes inputs, while ``"with_moneyness"`` also includes
        ``s0 / k`` as an engineered feature.

    Returns
    -------
    list[str]
        Ordered feature columns to pass to the neural network.
    """
    if feature_set == "base":
        return list(BASE_FEATURE_COLUMNS)
    if feature_set == "with_moneyness":
        return [*BASE_FEATURE_COLUMNS, MONEYNESS_COLUMN]
    raise ValueError(
        f"Unknown feature_set={feature_set!r}. " "Expected 'base' or 'with_moneyness'."
    )


def generate_synthetic_dataset(config: DatasetConfig) -> pd.DataFrame:
    """Generate option parameters and analytical Black-Scholes labels.

    Parameters
    ----------
    config
        Dataset generation settings, including sample size, random seed, and
        parameter ranges.

    Returns
    -------
    pd.DataFrame
        DataFrame containing input features, the Black-Scholes target price,
        and moneyness as an additional diagnostic column.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_samples

    data = {
        "s0": rng.uniform(config.s0_min, config.s0_max, size=n),
        "k": rng.uniform(config.k_min, config.k_max, size=n),
        "t": rng.uniform(config.t_min, config.t_max, size=n),
        "r": rng.uniform(config.r_min, config.r_max, size=n),
        "sigma": rng.uniform(config.sigma_min, config.sigma_max, size=n),
    }
    data[TARGET_COLUMN] = call_price(
        data["s0"],
        data["k"],
        data["t"],
        data["r"],
        data["sigma"],
    )
    data[MONEYNESS_COLUMN] = data["s0"] / data["k"]
    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save a dataset to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset previously saved with :func:`save_dataset`."""
    return pd.read_csv(path)
