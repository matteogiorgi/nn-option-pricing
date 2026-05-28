"""Tests for controlled target-noise generation."""

import numpy as np

from nn_option_pricing.config import DatasetConfig, TrainingConfig
from nn_option_pricing.dataset import TARGET_COLUMN, generate_synthetic_dataset
from nn_option_pricing.noise import (
    CLEAN_TARGET_COLUMN,
    NOISY_TARGET_COLUMN,
    PRICE_NOISE_COLUMN,
    NoiseConfig,
    add_relative_gaussian_price_noise,
)
from nn_option_pricing.noisy_experiment import (
    NoisyTargetsExperimentConfig,
    run_noisy_targets_experiment,
)


def test_zero_noise_preserves_clean_prices():
    df = generate_synthetic_dataset(DatasetConfig(n_samples=64, seed=7))

    noisy_df = add_relative_gaussian_price_noise(df, NoiseConfig(level=0.0, seed=11))

    assert np.allclose(noisy_df[CLEAN_TARGET_COLUMN], df[TARGET_COLUMN])
    assert np.allclose(noisy_df[NOISY_TARGET_COLUMN], df[TARGET_COLUMN])
    assert np.allclose(noisy_df[TARGET_COLUMN], df[TARGET_COLUMN])
    assert np.allclose(noisy_df[PRICE_NOISE_COLUMN], 0.0)


def test_positive_noise_is_reproducible_and_nonnegative():
    df = generate_synthetic_dataset(DatasetConfig(n_samples=64, seed=7))
    config = NoiseConfig(level=0.05, seed=11)

    first = add_relative_gaussian_price_noise(df, config)
    second = add_relative_gaussian_price_noise(df, config)

    assert np.allclose(first[NOISY_TARGET_COLUMN], second[NOISY_TARGET_COLUMN])
    assert np.all(first[NOISY_TARGET_COLUMN] >= 0.0)
    assert np.all(first[TARGET_COLUMN] >= 0.0)
    assert not np.allclose(first[NOISY_TARGET_COLUMN], first[CLEAN_TARGET_COLUMN])


def test_negative_noise_level_is_rejected():
    df = generate_synthetic_dataset(DatasetConfig(n_samples=8, seed=7))

    try:
        add_relative_gaussian_price_noise(df, NoiseConfig(level=-0.01))
    except ValueError as exc:
        assert "Noise level" in str(exc)
    else:
        raise AssertionError("Negative noise level should raise ValueError.")


def test_run_noisy_targets_experiment_writes_results(tmp_path):
    config = NoisyTargetsExperimentConfig(
        dataset=DatasetConfig(n_samples=80, seed=7),
        training=TrainingConfig(
            seed=7,
            feature_set="with_moneyness",
            max_epochs=1,
            batch_size=64,
            hidden_layers=(4,),
            activation="tanh",
        ),
        noise_levels=(0.02,),
        noise_seed=11,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        results_dir=tmp_path / "results",
    )

    result = run_noisy_targets_experiment(config)

    assert len(result["runs"]) == 1
    assert (tmp_path / "results" / "noisy_targets_metrics.json").exists()
    assert (tmp_path / "data" / "noise_0.02" / "synthetic_options.csv").exists()
    assert "metrics_vs_clean" in result["runs"][0]
    assert "metrics_vs_noisy" in result["runs"][0]
    assert result["runs"][0]["noise_level"] == 0.02
