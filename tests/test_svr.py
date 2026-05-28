"""Tests for the reduced-scale SVR benchmark."""

import json

from nn_option_pricing.svr import SVRBenchmarkConfig, run_svr_benchmark


def test_run_svr_benchmark_writes_summary(tmp_path):
    config = SVRBenchmarkConfig(
        n_samples=160,
        seeds=(3, 7),
        feature_set="with_moneyness",
        c=10.0,
        epsilon=0.05,
    )

    result = run_svr_benchmark(config=config, output_dir=tmp_path)

    assert result["config"]["n_samples"] == 160
    assert result["config"]["seeds"] == (3, 7)
    assert len(result["runs"]) == 2
    assert "mae" in result["summary"]
    assert "fit_time_seconds" in result["summary"]
    assert result["summary"]["mae"]["mean"] >= 0.0

    output_path = tmp_path / "svr_benchmark.json"
    assert output_path.exists()
    saved_result = json.loads(output_path.read_text())
    assert saved_result["config"]["seeds"] == [3, 7]
    assert len(saved_result["runs"]) == 2
