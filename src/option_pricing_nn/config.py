from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
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
    n_paths: int = 50_000
    option_batch_size: int = 512
    path_batch_size: int = 10_000
    evaluation_samples: int = 512
    seed: int = 123


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    figure_dir: Path = Path("outputs/figures")
    metrics_dir: Path = Path("outputs/metrics")
    dataset_path: Path = Path("data/synthetic_options.csv")
    model_path: Path = Path("outputs/model.pt")
    scaler_path: Path = Path("outputs/scaler.joblib")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()
    monte_carlo: MonteCarloConfig = MonteCarloConfig()
    paths: PathConfig = PathConfig()
