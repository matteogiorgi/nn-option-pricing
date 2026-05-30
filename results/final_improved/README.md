# Improved Final Configuration Results

This directory contains selected artifacts from the improved final run. The run
uses the strongest configuration identified by the intermediate experiments:

- feature set: `with_moneyness`;
- activation function: `silu`.

The same selected artifacts have also been promoted to `results/final/`, which
is the canonical directory used by the report. This directory is kept as the
historical improved-run location because the command below writes reproducible
artifacts to `data/final_improved/` and `outputs/final_improved/`.

## Command

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 100000 \
  --max-epochs 200 \
  --batch-size 1024 \
  --mc-n-paths 50000 \
  --mc-evaluation-samples 512 \
  --feature-set with_moneyness \
  --activation silu \
  --seed 42 \
  --data-dir data/final_improved \
  --output-dir outputs/final_improved
```

## Neural Network Metrics

```json
{
  "mae": 0.04289555922150612,
  "rmse": 0.06208079538235526,
  "r2": 0.9999927878379822,
  "mape_percent_price_gt_1": 0.48025041818618774
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

## Comparison With Earlier Final Run

| Metric | Earlier final | Selected final | Relative reduction |
|---|---:|---:|---:|
| MAE | 0.0619787835 | 0.0428955592 | 30.7899% |
| RMSE | 0.0821850306 | 0.0620807954 | 24.4622% |
| MAPE, price > 1 | 0.6746048331% | 0.4802504182% | 28.8101% |

The selected final run is better than the earlier final neural network on all
reported neural-network error metrics.

## Interpretation

The result supports the intermediate experimental findings: adding moneyness as
an engineered feature and using the SiLU activation improves the neural network
surrogate in this controlled Black-Scholes setting.

This run is the selected final reported model. The improvement comes from two
controlled changes:

1. exposing `s0 / k` as an additional input feature;
2. replacing ReLU with the smoother SiLU activation.

## Included Artifacts

- `experiment_config.json`
- `metrics/nn_metrics.json`
- `metrics/monte_carlo_vs_black_scholes_metrics.json`
- `figures/loss.png`
- `figures/true_vs_predicted.png`
- `figures/error_distribution.png`
- `figures/error_vs_moneyness.png`
- `figures/error_vs_maturity.png`
- `figures/error_vs_volatility.png`
