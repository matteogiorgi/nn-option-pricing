# Final Experiment Results

This directory contains the selected artifacts from the final experiment run.
Generated datasets, trained model checkpoints, and scalers are kept in
`data/final/` and `outputs/final/`, which are ignored by Git because they can be
reproduced from the code and configuration.

## Command

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 100000 \
  --max-epochs 200 \
  --batch-size 1024 \
  --mc-n-paths 50000 \
  --mc-evaluation-samples 512 \
  --data-dir data/final \
  --output-dir outputs/final
```

## Neural Network Metrics

```json
{
  "mae": 0.0619787834584713,
  "rmse": 0.08218503059459481,
  "r2": 0.9999874234199524,
  "mape_percent_price_gt_1": 0.6746048331260681
}
```

## Monte Carlo Metrics

Monte Carlo estimates are evaluated against analytical Black-Scholes prices on
a deterministic subset of 512 test options.

```json
{
  "mae": 0.088821478006634,
  "rmse": 0.15790186868176595,
  "r2": 0.9999513355216346,
  "mape_percent_price_gt_1": 0.6483135955335606
}
```

## Included Artifacts

- `experiment_config.json`
- `verification.md`
- `metrics/nn_metrics.json`
- `metrics/monte_carlo_vs_black_scholes_metrics.json`
- `figures/loss.png`
- `figures/true_vs_predicted.png`
- `figures/error_distribution.png`
- `figures/error_vs_moneyness.png`
- `figures/error_vs_maturity.png`
- `figures/error_vs_volatility.png`
