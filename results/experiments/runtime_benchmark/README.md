# Runtime Benchmark

This benchmark compares pricing runtimes for:

- vectorized analytical Black-Scholes pricing;
- neural network inference;
- Monte Carlo pricing.

The benchmark uses the strongest intermediate neural configuration identified
so far: `with_moneyness + silu`.

## Configuration

```bash
.venv/bin/python scripts/benchmark_runtime.py \
  --n-samples 50000 \
  --n-options 1000 \
  --max-epochs 100 \
  --batch-size 1024 \
  --feature-set with_moneyness \
  --activation silu \
  --mc-n-paths 20000 \
  --mc-option-batch-size 512 \
  --mc-path-batch-size 10000 \
  --repeats 5 \
  --mc-repeats 1 \
  --output-dir results/experiments/runtime_benchmark
```

The neural network training time is reported separately from pricing time.
This distinction is important because the network has an upfront training cost,
while inference is the relevant quantity once the surrogate model has already
been trained.

## Results

Training:

| Quantity | Value |
|---|---:|
| Training time | 359.1667 s |
| Epochs run | 100 |

Pricing 1,000 options:

| Method | Mean time | Median time | Mean options/s |
|---|---:|---:|---:|
| Black-Scholes analytical | 0.000908 s | 0.000523 s | 1,101,094.82 |
| Neural network inference | 0.031897 s | 0.028669 s | 31,350.77 |
| Monte Carlo, 20,000 paths | 1.282354 s | 1.282354 s | 779.82 |

Approximate mean-time speedups:

| Comparison | Speedup |
|---|---:|
| Neural inference vs Monte Carlo | 40.20x |
| Black-Scholes vs neural inference | 35.12x |
| Black-Scholes vs Monte Carlo | 1411.99x |

## Interpretation

The analytical Black-Scholes formula is by far the fastest method in this
specific setting, as expected. Since a closed-form formula is available, the
neural network is not meant to beat Black-Scholes analytically.

The relevant comparison for the surrogate-model argument is neural inference
versus Monte Carlo. In this benchmark, once trained, the neural network prices
the same 1,000 options roughly 40 times faster than Monte Carlo with 20,000
paths.

This supports the interpretation of the neural network as a fast pricing
surrogate. The trade-off is the initial training cost, which was about 359
seconds for this intermediate configuration.

The benchmark should be interpreted as hardware- and configuration-dependent:
absolute runtimes may change across machines, but the separation between
training cost and inference cost is the main conceptual point.
