# Support Vector Regression Benchmark

This experiment evaluates a classical machine-learning baseline for the
Black-Scholes pricing surrogate task.

Support Vector Regression is run on reduced synthetic datasets because RBF
kernel methods scale less favorably than neural networks as the number of
training samples grows. The goal is therefore not to replace the final neural
network experiment, but to provide an additional baseline on a controlled and
computationally reasonable setting.

## Command

```bash
.venv/bin/python scripts/run_svr_benchmark.py \
  --n-samples 5000 \
  --seeds 11 42 73 \
  --feature-set with_moneyness \
  --output-dir results/experiments/svr_benchmark
```

## Configuration

- samples per run: `5000`;
- test split: `20%`;
- seeds: `11`, `42`, `73`;
- feature set: `with_moneyness`;
- model: RBF Support Vector Regression;
- `C`: `100.0`;
- `epsilon`: `0.01`;
- `gamma`: `scale`.

Both input features and target prices are standardized before fitting the SVR.
Predictions are mapped back to original price units before computing metrics.

## Aggregate Results

| Metric | Mean | Std |
| --- | ---: | ---: |
| MAE | 0.1508814487 | 0.0073181594 |
| RMSE | 0.2310028935 | 0.0059229286 |
| R2 | 0.9999049414 | 0.0000048517 |
| MAPE, price > 1 | 1.8252930199% | 0.2422206040% |
| Fit time | 12.3197846853 s | 0.7977419153 s |
| Prediction time | 0.0300548457 s | 0.0028954640 s |

## Interpretation

The SVR baseline approximates the Black-Scholes function well on reduced
datasets, with high R2 and low absolute error. Its error is nevertheless higher
than the final neural network configuration evaluated on the full experimental
setup.

This supports the role of the neural network as the main scalable surrogate
model, while SVR remains a useful classical benchmark for smaller controlled
experiments.

## Files

- `svr_benchmark.json`: full configuration, per-seed metrics, and aggregate
  mean/std summary.

## Verification

After adding the SVR benchmark, the following checks were run successfully:

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/sphinx-build -W -b html docs/source docs/build/html
```
