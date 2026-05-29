"""Tests for the noisy-target SVR benchmark."""

import json

from nn_option_pricing.noisy_svr_experiment import (
    NoisySVRBenchmarkConfig,
    run_noisy_svr_benchmark,
)


def test_run_noisy_svr_benchmark_writes_grouped_summary(tmp_path):
    config = NoisySVRBenchmarkConfig(
        n_samples=120,
        seeds=(3, 7),
        noise_levels=(0.0, 0.02),
        feature_set="with_moneyness",
        c=10.0,
        epsilon=0.05,
    )

    result = run_noisy_svr_benchmark(config=config, output_dir=tmp_path)

    assert result["config"]["n_samples"] == 120
    assert len(result["runs"]) == 4
    assert set(result["summary"]) == {"0.00", "0.02"}
    assert "clean_mae" in result["summary"]["0.02"]
    assert "noisy_mae" in result["summary"]["0.02"]
    assert result["summary"]["0.02"]["clean_mae"]["mean"] >= 0.0

    output_path = tmp_path / "noisy_svr_benchmark.json"
    assert output_path.exists()
    saved_result = json.loads(output_path.read_text())
    assert len(saved_result["runs"]) == 4
