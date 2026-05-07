import json

from nn_option_pricing.config import (
    DatasetConfig,
    ExperimentConfig,
    MonteCarloConfig,
    PathConfig,
    TrainingConfig,
)
from nn_option_pricing.pipeline import run_experiment


def test_run_experiment_smoke_test_writes_expected_artifacts(tmp_path):
    paths = PathConfig(
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        figure_dir=tmp_path / "outputs" / "figures",
        metrics_dir=tmp_path / "outputs" / "metrics",
        dataset_path=tmp_path / "data" / "synthetic_options.csv",
        model_path=tmp_path / "outputs" / "model.pt",
        scaler_path=tmp_path / "outputs" / "scaler.joblib",
    )
    config = ExperimentConfig(
        dataset=DatasetConfig(n_samples=160, seed=17),
        training=TrainingConfig(
            seed=17,
            batch_size=32,
            max_epochs=2,
            patience=2,
            hidden_layers=(8,),
        ),
        monte_carlo=MonteCarloConfig(
            n_paths=200,
            option_batch_size=16,
            path_batch_size=100,
            evaluation_samples=8,
            seed=17,
        ),
        paths=paths,
    )

    metrics = run_experiment(config)

    expected_metric_keys = {"mae", "rmse", "r2", "mape_percent_price_gt_1"}
    assert set(metrics) == expected_metric_keys
    assert paths.dataset_path.exists()
    assert paths.model_path.exists()
    assert paths.scaler_path.exists()
    assert (paths.output_dir / "experiment_config.json").exists()
    assert (paths.output_dir / "target_scaler.joblib").exists()
    assert (paths.metrics_dir / "nn_metrics.json").exists()
    assert (paths.metrics_dir / "monte_carlo_vs_black_scholes_metrics.json").exists()

    expected_figures = [
        "loss.png",
        "true_vs_predicted.png",
        "error_distribution.png",
        "error_vs_moneyness.png",
        "error_vs_maturity.png",
        "error_vs_volatility.png",
    ]
    for figure in expected_figures:
        assert (paths.figure_dir / figure).exists()

    saved_metrics = json.loads((paths.metrics_dir / "nn_metrics.json").read_text())
    assert set(saved_metrics) == expected_metric_keys

    saved_config = json.loads((paths.output_dir / "experiment_config.json").read_text())
    assert saved_config["dataset"]["n_samples"] == config.dataset.n_samples
    assert saved_config["training"]["hidden_layers"] == list(
        config.training.hidden_layers
    )
    assert saved_config["paths"]["output_dir"] == str(paths.output_dir)
