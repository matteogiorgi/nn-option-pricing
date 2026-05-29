# Noisy SVR Benchmark

This experiment repeats the controlled noisy-target robustness setting with a
classical Support Vector Regression baseline.

The SVR model is trained on noisy Black-Scholes targets and evaluated against
the clean analytical Black-Scholes prices. This mirrors the neural-network
noisy-target experiment, but uses reduced datasets because RBF kernel methods
scale less favorably than neural networks.

## Command

```bash
.venv/bin/python scripts/run_noisy_svr_benchmark.py \
  --n-samples 5000 \
  --seeds 11 42 73 \
  --noise-levels 0.0 0.01 0.05 \
  --feature-set with_moneyness \
  --output-dir results/experiments/noisy_svr_benchmark
```

## Configuration

- samples per run: `5000`;
- seeds: `11`, `42`, `73`;
- noise levels: `0%`, `1%`, `5%`;
- feature set: `with_moneyness`;
- model: RBF Support Vector Regression;
- `C`: `100.0`;
- `epsilon`: `0.01`;
- `gamma`: `scale`;
- evaluation target: clean Black-Scholes price.

Datasets are generated in memory and are fully reproducible from the saved
configuration. Only aggregate results are tracked.

## Aggregate Results Against Clean Black-Scholes Prices

| Noise level | MAE mean | MAE std | RMSE mean | RMSE std | R2 mean | MAPE mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0.1508814487 | 0.0073181594 | 0.2310028935 | 0.0059229286 | 0.9999049414 | 1.8252930199% |
| 1% | 0.1928422200 | 0.0048361064 | 0.3076495032 | 0.0169511638 | 0.9998301207 | 1.8704652733% |
| 5% | 0.5618524050 | 0.0208576478 | 1.0505699989 | 0.0517905754 | 0.9980248753 | 3.4902511686% |

## Interpretation

The SVR baseline remains accurate under mild target noise, but it degrades more
strongly at the 5% noise level. This experiment provides a classical ML
robustness reference for the neural-network noisy-target experiment, while the
neural network remains the main scalable surrogate model.

As with the neural-network noisy-target experiment, these noisy labels are a
controlled perturbation of analytical Black-Scholes prices and should not be
interpreted as real market prices.

## Files

- `noisy_svr_benchmark.json`: full configuration, per-seed metrics, and
  aggregate summaries by noise level.

## Verification

After adding this benchmark, the following checks were run:

```bash
.venv/bin/python -m pytest
.venv/bin/sphinx-build -W -b html docs/source docs/build/html
cd report && latexmk -g -xelatex main.tex
cd presentation && latexmk -g -xelatex main.tex
```
