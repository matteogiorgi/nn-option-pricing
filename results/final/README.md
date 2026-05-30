# Final Experiment Results

This directory contains the selected artifacts from the final experiment run.
Generated datasets, trained model checkpoints, and scalers are kept in
`data/final_improved/` and `outputs/final_improved/`, which are ignored by Git
because they can be reproduced from the code and configuration.

The final model uses the improved configuration selected after the intermediate
feature-engineering and activation-function experiments:

- feature set: `with_moneyness`;
- activation function: `silu`.

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

## Final Verification

The following validation checks were executed after generating the final
experiment artifacts.

Repository state:

- final improved configuration promoted to `results/final/`;
- final experiment artifacts directory: `results/final/`;
- generated but untracked reproducible artifacts:
  - `data/final_improved/`;
  - `outputs/final_improved/`.

Test suite:

```bash
.venv/bin/pytest
```

Result:

```text
24 passed
```

Documentation build:

```bash
.venv/bin/sphinx-build -W -b html docs/source docs/build/html
```

Result:

```text
build succeeded
```
