"""Controlled target-noise utilities for robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nn_option_pricing.dataset import TARGET_COLUMN

CLEAN_TARGET_COLUMN = "clean_call_price"
NOISY_TARGET_COLUMN = "noisy_call_price"
PRICE_NOISE_COLUMN = "price_noise"
NOISE_STD_COLUMN = "price_noise_std"


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for controlled Black-Scholes target perturbations."""

    level: float = 0.0
    seed: int = 123
    price_floor: float = 1.0


def add_relative_gaussian_price_noise(
    df: pd.DataFrame,
    config: NoiseConfig,
) -> pd.DataFrame:
    """Return a copy of a pricing dataset with noisy target columns.

    The clean analytical Black-Scholes price is preserved in
    ``clean_call_price``. A noisy target is generated as

    ``clean_call_price + Normal(0, level * max(clean_call_price, price_floor))``

    and then clipped at zero to respect the non-negativity of call prices.
    The standard target column ``call_price`` is set to the noisy target so the
    existing training utilities can be reused without changing the main clean
    pipeline.
    """
    if config.level < 0:
        raise ValueError("Noise level must be non-negative.")
    if config.price_floor < 0:
        raise ValueError("Price floor must be non-negative.")

    noisy_df = df.copy()
    clean_prices = noisy_df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(config.seed)

    noise_std = config.level * np.maximum(clean_prices, config.price_floor)
    sampled_noise = rng.normal(loc=0.0, scale=noise_std)
    noisy_prices = np.maximum(clean_prices + sampled_noise, 0.0)
    actual_noise = noisy_prices - clean_prices

    noisy_df[CLEAN_TARGET_COLUMN] = clean_prices
    noisy_df[NOISE_STD_COLUMN] = noise_std
    noisy_df[PRICE_NOISE_COLUMN] = actual_noise
    noisy_df[NOISY_TARGET_COLUMN] = noisy_prices
    noisy_df[TARGET_COLUMN] = noisy_prices
    return noisy_df


def noise_level_slug(level: float) -> str:
    """Return a filesystem-friendly label for a noise level."""
    return f"noise_{level:.2f}"
