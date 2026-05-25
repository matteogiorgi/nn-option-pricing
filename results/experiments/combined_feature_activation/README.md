# Combined Feature and Activation Experiment

This experiment combines the two most promising intermediate extensions:

- adding `moneyness = s0 / k` as an engineered feature;
- using SiLU instead of ReLU as the hidden-layer activation function.

The goal is to check whether the improvements observed separately also hold
when the two changes are applied together.

## Configuration

The combined run uses the same intermediate setup as the previous experiments:

```bash
.venv/bin/python scripts/run_experiment.py \
  --n-samples 50000 \
  --max-epochs 100 \
  --batch-size 1024 \
  --mc-n-paths 20000 \
  --mc-evaluation-samples 256 \
  --feature-set with_moneyness \
  --activation silu \
  --seed 42 \
  --data-dir data/experiments/moneyness_silu \
  --output-dir outputs/experiments/moneyness_silu
```

## Neural Network Metrics

| Experiment | MAE | RMSE | R2 | MAPE, price > 1 |
|---|---:|---:|---:|---:|
| `base + relu` | 0.1726914048 | 0.2297268905 | 0.9999036789 | 1.8686523438% |
| `with_moneyness + relu` | 0.1541707218 | 0.2092986996 | 0.9999200702 | 1.6669688225% |
| `base + silu` | 0.1077493951 | 0.1581683768 | 0.9999543428 | 1.2487852573% |
| `with_moneyness + silu` | 0.1025449410 | 0.1516892483 | 0.9999579787 | 1.3213086128% |

Relative error reductions versus the baseline `base + relu`:

| Experiment | MAE reduction | RMSE reduction | MAPE reduction |
|---|---:|---:|---:|
| `with_moneyness + relu` | 10.7247% | 8.8924% | 10.7930% |
| `base + silu` | 37.6058% | 31.1494% | 33.1719% |
| `with_moneyness + silu` | 40.6195% | 33.9697% | 29.2908% |

## Interpretation

The combined configuration gives the best MAE and RMSE among the tested
intermediate experiments. This suggests that the engineered moneyness feature
and the smoother SiLU activation can be beneficial together.

However, the best MAPE on prices greater than 1 is still obtained by
`base + silu`, not by the combined configuration. This means the combined model
has lower average absolute and squared errors overall, but does not uniformly
dominate every metric.

The combined configuration is therefore a strong candidate for a final
large-scale run, but it should be presented carefully: it improves the main
absolute-error metrics in this intermediate setup, while the percentage-error
metric is slightly better for `base + silu`.
