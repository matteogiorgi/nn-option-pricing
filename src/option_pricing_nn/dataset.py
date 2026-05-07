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

from option_pricing_nn.black_scholes import call_price
from option_pricing_nn.config import DatasetConfig

FEATURE_COLUMNS = ["s0", "k", "t", "r", "sigma"]
TARGET_COLUMN = "call_price"


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
    data["moneyness"] = data["s0"] / data["k"]
    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Save a dataset to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a dataset previously saved with :func:`save_dataset`."""
    return pd.read_csv(path)
