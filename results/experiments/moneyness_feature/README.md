# Moneyness Feature Experiment

This experiment compares the baseline neural network feature set with an
engineered feature set that also includes `moneyness = s0 / k`.

The goal is to test whether making a financially meaningful relation explicit
helps the network approximate the Black-Scholes pricing function. Moneyness
does not add new information beyond `s0` and `k`, but it may improve
optimization by exposing a useful ratio directly to the model.

## Configuration

Both runs use the same random seed, dataset size, model architecture, activation
function, training hyperparameters, and Monte Carlo benchmark settings. The
only changed parameter is `--feature-set`.

Baseline command:

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 50000 \
  --max-epochs 100 \
  --batch-size 1024 \
  --mc-n-paths 20000 \
  --mc-evaluation-samples 256 \
  --feature-set base \
  --activation relu \
  --seed 42 \
  --data-dir data/experiments/base_relu \
  --output-dir outputs/experiments/base_relu
```

Moneyness command:

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 50000 \
  --max-epochs 100 \
  --batch-size 1024 \
  --mc-n-paths 20000 \
  --mc-evaluation-samples 256 \
  --feature-set with_moneyness \
  --activation relu \
  --seed 42 \
  --data-dir data/experiments/moneyness_relu \
  --output-dir outputs/experiments/moneyness_relu
```

## Neural Network Metrics

| Experiment | MAE | RMSE | R2 | MAPE, price > 1 |
|---|---:|---:|---:|---:|
| `base + relu` | 0.1726914048 | 0.2297268905 | 0.9999036789 | 1.8686523438% |
| `with_moneyness + relu` | 0.1541707218 | 0.2092986996 | 0.9999200702 | 1.6669688225% |

Relative error reductions from adding moneyness:

- MAE: 10.7247%;
- RMSE: 8.8924%;
- MAPE on prices greater than 1: 10.7930%.

## Interpretation

The intermediate experiment suggests that including moneyness as an engineered
feature improves predictive accuracy under the tested configuration. This is
plausible because option prices depend strongly on the relation between spot
price and strike, and `s0 / k` exposes this relation directly.

The result should still be interpreted carefully: this is an intermediate
50,000-sample experiment, not yet a final large-scale run. A final comparison
should repeat the experiment with the selected final training configuration
before updating the report conclusions.
