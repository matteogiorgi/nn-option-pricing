# Neural Network Option Pricing

Project for the Machine Learning for Finance course.

The goal is to test whether a feed-forward neural network can accurately
approximate the Black-Scholes pricing function for European call options.

## Core Question

Can a feed-forward neural network approximate the Black-Scholes European call
pricing function with high accuracy, and how does it compare with analytical
Black-Scholes prices and Monte Carlo estimates?

## Project Structure

```text
.
├── docs/
│   └── source/
├── presentation/
│   ├── main.pdf
│   └── main.tex
├── report/
│   ├── main.pdf
│   ├── main.tex
│   └── sections/
├── results/
│   └── final/
├── scripts/
│   └── run_experiment.py
├── src/
│   └── nn_option_pricing/
├── tests/
├── ISTRUZIONI.md
├── README.md
├── pyproject.toml
├── requirements-docs.txt
└── requirements.txt
```

Generated experiment artifacts are written to `data/` and `outputs/`.
These directories are ignored by Git because they can be reproduced from the
code and configuration. The report and presentation PDFs are tracked because
they are final project deliverables. Selected final metrics and figures are
tracked in `results/final/`.

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

Run the complete experiment:

```bash
python scripts/run_experiment.py
```

Use `--data-dir` and `--output-dir` to keep different runs separate:

```bash
python scripts/run_experiment.py \
  --data-dir data/intermediate \
  --output-dir outputs/intermediate
```

The baseline uses the primitive Black-Scholes inputs
`(s0, k, t, r, sigma)` and ReLU activations. Experimental variants can be
selected from the command line:

```bash
python scripts/run_experiment.py \
  --feature-set with_moneyness \
  --activation silu \
  --data-dir data/experiments/moneyness_silu \
  --output-dir outputs/experiments/moneyness_silu
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

The run generates:

- synthetic option data in `data/`;
- trained model checkpoint in `outputs/`;
- experiment configuration snapshot in `outputs/experiment_config.json`;
- metrics in `outputs/metrics/`;
- figures in `outputs/figures/`.

Benchmark pricing runtimes:

```bash
python scripts/benchmark_runtime.py \
  --feature-set with_moneyness \
  --activation silu \
  --output-dir outputs/runtime_benchmark
```

Run the reduced-scale Support Vector Regression baseline:

```bash
python scripts/run_svr_benchmark.py \
  --n-samples 5000 \
  --seeds 11 42 73 \
  --feature-set with_moneyness \
  --output-dir results/experiments/svr_benchmark
```

## Testing

Run the automated test suite:

```bash
pytest
```

The tests cover analytical Black-Scholes pricing, synthetic dataset generation,
Monte Carlo pricing, and a small end-to-end pipeline smoke test.

## Documentation

Build the Sphinx documentation:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs/source docs/build/html
```

For a clean environment, install the runtime requirements and the local package
before building the API documentation:

```bash
pip install -r requirements.txt
pip install -r requirements-docs.txt
pip install -e . --no-build-isolation
```

Open `docs/build/html/index.html` in a browser to read the generated
documentation.

## Report and Presentation

The technical report is written in LaTeX:

```bash
cd report
latexmk -xelatex main.tex
```

The oral presentation is written with Beamer:

```bash
cd presentation
latexmk -xelatex main.tex
```

Both documents use XeLaTeX.



## Project Status and Future Work

The core experimental extensions have been implemented:

- moneyness can be included as an engineered feature with
  `--feature-set with_moneyness`;
- hidden-layer activations can be selected with `--activation`;
- runtime benchmarking is available through `scripts/benchmark_runtime.py`.
- a reduced-scale Support Vector Regression baseline is available through
  `scripts/run_svr_benchmark.py`.

The final selected configuration uses `with_moneyness + silu`. Final metrics,
figures, and configuration snapshots are tracked in `results/final/`.

Possible future extensions:

- study noisy Black-Scholes targets as a robustness experiment;
- explore pseudo-real or exchange-traded option data, making clear that this
  changes the research question from Black-Scholes approximation to market-price
  modeling.
