# Neural Network Option Pricing

Project for the Machine Learning for Finance course.

The project studies whether a feed-forward neural network can accurately
approximate the Black-Scholes pricing function for European call options. The
neural network is trained on synthetic Black-Scholes data and evaluated against
the analytical Black-Scholes formula, with Monte Carlo simulation, runtime
benchmarks, Support Vector Regression, and noisy-target experiments used as
additional points of comparison.

## Research Question

Can a feed-forward neural network approximate the Black-Scholes European call
pricing function with high accuracy, and can it act as a fast pricing surrogate
after training?

The project is deliberately not a trading or market-price prediction system.
It is a controlled supervised-learning experiment where the target function is
known exactly.

## Repository Structure

```text
.
├── data/                       # generated or selected datasets
├── docs/                       # Sphinx documentation sources
├── presentation/               # Beamer presentation
├── report/                     # LaTeX report
├── results/                    # tracked experiment summaries and figures
│   ├── experiments/            # additional benchmark and robustness results
│   └── final/                  # selected final experiment artifacts
├── scripts/                    # command-line experiment entry points
├── src/nn_option_pricing/      # reusable Python package
├── tests/                      # automated test suite
├── pyproject.toml
├── requirements-docs.txt
└── requirements.txt
```

Large reproducible run artifacts are written to `data/` and `outputs/`.
Generated outputs are ignored by Git unless they are selected final results or
small experiment summaries needed by the report. The report and presentation
PDFs are tracked because they are project deliverables.

## Quick Start

Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

Run the default experiment:

```bash
python scripts/run_experiment.py
```

For a fast smoke test:

```bash
python scripts/run_experiment.py \
  --n-samples 10000 \
  --max-epochs 40 \
  --batch-size 1024 \
  --mc-n-paths 5000 \
  --mc-evaluation-samples 128
```

The selected final configuration uses moneyness as an engineered feature and
SiLU hidden activations:

```bash
python scripts/run_experiment.py \
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

The run generates:

- synthetic option data;
- trained model checkpoint;
- input and target scalers;
- experiment configuration snapshot;
- metrics in JSON format;
- diagnostic figures.

## Additional Experiments

Runtime benchmark:

```bash
python scripts/benchmark_runtime.py \
  --feature-set with_moneyness \
  --activation silu \
  --output-dir results/experiments/runtime_benchmark
```

Reduced-scale Support Vector Regression baseline:

```bash
python scripts/run_svr_benchmark.py \
  --n-samples 5000 \
  --seeds 11 42 73 \
  --feature-set with_moneyness \
  --output-dir results/experiments/svr_benchmark
```

Noisy-target neural-network robustness experiment:

```bash
python scripts/run_noisy_targets_experiment.py \
  --n-samples 50000 \
  --noise-levels 0.0 0.01 0.05 \
  --max-epochs 100 \
  --feature-set with_moneyness \
  --activation silu \
  --results-dir results/experiments/noisy_targets
```

Noisy-target SVR benchmark:

```bash
python scripts/run_noisy_svr_benchmark.py \
  --n-samples 5000 \
  --seeds 11 42 73 \
  --noise-levels 0.0 0.01 0.05 \
  --feature-set with_moneyness \
  --output-dir results/experiments/noisy_svr_benchmark
```

Tracked summaries of these experiments are stored in `results/experiments/`.

## Testing

Run the automated test suite:

```bash
pytest
```

The tests cover analytical Black-Scholes pricing, synthetic dataset generation,
Monte Carlo pricing, model construction, target-noise generation, SVR
benchmarking, and small end-to-end smoke tests.

## Documentation

Build the Sphinx documentation locally:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs/source docs/build/html
```

Strict build, treating warnings as errors:

```bash
sphinx-build -W -b html docs/source docs/build/html
```

The generated local entry point is `docs/build/html/index.html`.

The documentation is also published through GitHub Pages:

```text
https://matteogiorgi.github.io/nn-option-pricing
```

## Report and Presentation

Build the technical report:

```bash
cd report
latexmk -xelatex main.tex
```

Build the Beamer presentation:

```bash
cd presentation
latexmk -xelatex main.tex
```

Both documents use XeLaTeX.

## Final Status

The codebase implements the complete experimental pipeline:

- synthetic Black-Scholes dataset generation;
- analytical Black-Scholes pricing;
- batched Monte Carlo benchmark;
- feed-forward neural-network training with scaling and early stopping;
- feature-set and activation-function comparisons;
- runtime benchmark;
- reduced-scale SVR baseline;
- noisy-target robustness experiments for both NN and SVR;
- tests, Sphinx documentation, LaTeX report, and Beamer slides.

The selected final neural-network configuration is:

```text
feature set: with_moneyness
activation: silu
```

Selected final metrics, figures, and configuration snapshots are tracked in
`results/final/`.

Possible future extensions include Greeks estimation, more detailed
regime-based error analysis, stochastic-volatility models, path-dependent
options, American options, and real exchange-traded option data. The last
extension would change the research question from Black-Scholes function
approximation to market-price modeling.
