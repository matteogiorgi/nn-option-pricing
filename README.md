# Neural Network Option Pricing

Project for Machine Learning for Finance.

The goal is to test whether a feed-forward neural network can accurately
approximate the Black-Scholes pricing function for European call options.

## Core Question

Can a feed-forward neural network approximate the Black-Scholes European call
pricing function with high accuracy, and how does it compare with analytical
Black-Scholes prices and Monte Carlo estimates?

## Project Structure

```text
.
├── data/
├── outputs/
│   ├── figures/
│   └── metrics/
├── scripts/
│   └── run_experiment.py
├── src/
│   └── nn_option_pricing/
├── tests/
├── ISTRUZIONI.md
├── README.md
└── requirements.txt
```

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

For a fast smoke test:

```bash
python scripts/run_experiment.py \
  --n-samples 10000 \
  --max-epochs 40 \
  --mc-n-paths 5000 \
  --mc-evaluation-samples 128
```

The run generates:

- synthetic option data in `data/`;
- trained model checkpoint in `outputs/`;
- metrics in `outputs/metrics/`;
- figures in `outputs/figures/`.

## Documentation

Build the Sphinx documentation:

```bash
pip install -r requirements-docs.txt
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser to read the generated
documentation.
