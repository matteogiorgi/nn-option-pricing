"""Configuration objects for reproducible experiments.

The project uses frozen dataclasses instead of global constants so that
experiments can be configured explicitly from scripts while keeping defaults in
one place.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    """Ranges and random seed used to generate synthetic option data."""

    n_samples: int = 100_000
    seed: int = 42
    s0_min: float = 50.0
    s0_max: float = 150.0
    k_min: float = 50.0
    k_max: float = 150.0
    t_min: float = 0.1
    t_max: float = 2.0
    r_min: float = 0.0
    r_max: float = 0.05
    sigma_min: float = 0.1
    sigma_max: float = 0.6


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for neural network training and data splitting."""

    seed: int = 42
    test_size: float = 0.15
    validation_size: float = 0.15
    batch_size: int = 1024
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    hidden_layers: tuple[int, ...] = (64, 64, 32)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Parameters controlling the Monte Carlo benchmark."""

    n_paths: int = 50_000
    option_batch_size: int = 512
    path_batch_size: int = 10_000
    evaluation_samples: int = 512
    seed: int = 123


@dataclass(frozen=True)
class PathConfig:
    """Filesystem locations for generated data, outputs, and artifacts."""

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    metrics_dir: Path = Path("outputs/metrics")
    dataset_path: Path = Path("data/synthetic_options.csv")
    model_path: Path = Path("outputs/model.pt")
    scaler_path: Path = Path("outputs/scaler.joblib")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration grouping all experiment settings."""

    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    paths: PathConfig = PathConfig()


def make_path_config(
    data_dir: Path | str = Path("data"),
    output_dir: Path | str = Path("outputs"),
) -> PathConfig:
    """Create a consistent set of experiment paths from base directories.

    Parameters
    ----------
    data_dir
        Directory where generated datasets are written.
    output_dir
        Directory where models, scalers, metrics, figures, and configuration
        snapshots are written.

    Returns
    -------
    PathConfig
        Fully resolved path configuration derived from the two base
        directories.
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    return PathConfig(
        data_dir=data_path,
        output_dir=output_path,
        figure_dir=output_path / "figures",
        metrics_dir=output_path / "metrics",
        dataset_path=data_path / "synthetic_options.csv",
        model_path=output_path / "model.pt",
        scaler_path=output_path / "scaler.joblib",
    )


def config_to_dict(config: Any) -> Any:
    """Convert experiment configuration objects into JSON-serializable values."""
    if is_dataclass(config) and not isinstance(config, type):
        return {key: config_to_dict(value) for key, value in asdict(config).items()}
    if isinstance(config, Path):
        return str(config)
    if isinstance(config, tuple):
        return [config_to_dict(value) for value in config]
    if isinstance(config, list):
        return [config_to_dict(value) for value in config]
    if isinstance(config, dict):
        return {key: config_to_dict(value) for key, value in config.items()}
    return config
