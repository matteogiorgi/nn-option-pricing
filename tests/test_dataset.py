"""Tests for synthetic dataset generation and persistence."""

import numpy as np
import pandas as pd

from nn_option_pricing.black_scholes import call_price
from nn_option_pricing.config import DatasetConfig
from nn_option_pricing.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_synthetic_dataset,
    load_dataset,
    save_dataset,
)


def test_generate_synthetic_dataset_schema_and_ranges():
    config = DatasetConfig(
        n_samples=128,
        seed=7,
        s0_min=80.0,
        s0_max=120.0,
        k_min=70.0,
        k_max=130.0,
        t_min=0.25,
        t_max=1.5,
        r_min=0.01,
        r_max=0.04,
        sigma_min=0.15,
        sigma_max=0.45,
    )

    df = generate_synthetic_dataset(config)

    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN, "moneyness"]
    assert len(df) == config.n_samples
    assert df["s0"].between(config.s0_min, config.s0_max).all()
    assert df["k"].between(config.k_min, config.k_max).all()
    assert df["t"].between(config.t_min, config.t_max).all()
    assert df["r"].between(config.r_min, config.r_max).all()
    assert df["sigma"].between(config.sigma_min, config.sigma_max).all()
    assert np.all(df[TARGET_COLUMN].to_numpy() >= 0.0)
    assert np.allclose(df["moneyness"], df["s0"] / df["k"])


def test_generate_synthetic_dataset_is_reproducible_with_seed():
    config = DatasetConfig(n_samples=32, seed=123)

    first = generate_synthetic_dataset(config)
    second = generate_synthetic_dataset(config)

    pd.testing.assert_frame_equal(first, second)


def test_generate_synthetic_dataset_prices_match_black_scholes():
    df = generate_synthetic_dataset(DatasetConfig(n_samples=64, seed=99))

    expected = call_price(
        df["s0"].to_numpy(),
        df["k"].to_numpy(),
        df["t"].to_numpy(),
        df["r"].to_numpy(),
        df["sigma"].to_numpy(),
    )

    assert np.allclose(df[TARGET_COLUMN].to_numpy(), expected)


def test_save_and_load_dataset_round_trip(tmp_path):
    df = generate_synthetic_dataset(DatasetConfig(n_samples=16, seed=5))
    path = tmp_path / "nested" / "synthetic_options.csv"

    save_dataset(df, path)
    loaded = load_dataset(path)

    pd.testing.assert_frame_equal(loaded, df)
