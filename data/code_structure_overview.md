# Code Structure Overview

This note summarizes how the codebase is organized and what each component does.
It is intended as a study note for presenting the project structure to the
group.

## General Idea

The code is organized as a complete experimental pipeline:

```text
dataset generation
-> Black-Scholes pricing
-> neural network training
-> evaluation
-> Monte Carlo benchmark
-> metrics, figures, model, and configuration saving
```

The main package is:

```text
src/nn_option_pricing/
```

## Main Package Structure

```text
src/nn_option_pricing/
├── black_scholes.py
├── config.py
├── dataset.py
├── evaluation.py
├── model.py
├── monte_carlo.py
├── pipeline.py
├── plots.py
└── training.py
```

## `config.py`

This module contains the configuration objects used throughout the project.

It defines:

- dataset ranges;
- number of observations;
- training hyperparameters;
- Monte Carlo parameters;
- input and output paths.

Main classes:

```python
DatasetConfig
TrainingConfig
MonteCarloConfig
PathConfig
ExperimentConfig
```

The purpose of this module is to avoid hardcoded values scattered across the
codebase. Experiment settings are grouped in structured dataclasses, making
runs easier to reproduce and document.

## `black_scholes.py`

This module contains the core analytical pricing utilities.

Main functions:

```python
call_price(...)
call_payoff(...)
```

`call_price` computes the European call option price using the analytical
Black-Scholes formula.

`call_payoff` computes the call payoff at maturity:

```text
max(S_T - K, 0)
```

This module provides the mathematical ground truth for the project.

## `dataset.py`

This module generates the synthetic dataset.

Main function:

```python
generate_synthetic_dataset(config)
```

It performs the following steps:

1. samples `s0`, `k`, `t`, `r`, and `sigma` from uniform distributions;
2. computes the target `call_price` with the analytical Black-Scholes formula;
3. adds `moneyness = s0 / k` as a diagnostic variable;
4. returns a `pandas.DataFrame`.

It also provides:

```python
save_dataset(...)
load_dataset(...)
```

The dataset is synthetic by design because the goal is to test whether the
neural network can approximate the Black-Scholes pricing function in a
controlled and reproducible setting.

## `monte_carlo.py`

This module implements the Monte Carlo benchmark.

Main function:

```python
call_price_mc(...)
```

It:

- samples the exact terminal distribution of geometric Brownian motion;
- computes the European call payoff;
- discounts the average payoff;
- processes options and simulation paths in batches to control memory usage.

Monte Carlo is used as a classical numerical benchmark against the analytical
Black-Scholes formula and the neural network surrogate.

## `model.py`

This module defines the PyTorch neural network.

Main class:

```python
PricingMLP
```

The model is a feed-forward multilayer perceptron:

```text
Input(5)
-> hidden layers
-> Output(1)
```

The five input features are:

```text
s0, k, t, r, sigma
```

The output is:

```text
call_price
```

## `training.py`

This module contains the training and inference logic.

Main functions:

```python
train_model(...)
predict(...)
set_seed(...)
```

`train_model` handles:

- train/validation/test split;
- input standardization;
- target scaling;
- PyTorch training loop;
- mini-batch gradient descent;
- early stopping;
- recovery of the best validation model;
- return of the trained model and useful artifacts.

`predict` runs inference with the trained model and converts predictions back
to price units.

## `evaluation.py`

This module computes evaluation metrics and saves them to disk.

Main functions:

```python
regression_metrics(...)
save_metrics(...)
```

Metrics:

- MAE;
- RMSE;
- R-squared;
- MAPE on prices greater than 1.

MAPE is computed only for prices greater than 1 because percentage errors can
become unstable when the option price is close to zero.

## `plots.py`

This module generates diagnostic plots.

Main functions:

```python
plot_loss(...)
plot_true_vs_predicted(...)
plot_error_distribution(...)
plot_error_against_feature(...)
```

The plots show:

- training and validation loss;
- true Black-Scholes prices versus neural network predictions;
- distribution of prediction errors;
- error versus moneyness;
- error versus maturity;
- error versus volatility.

## `pipeline.py`

This is the main orchestration module.

Main function:

```python
run_experiment(config)
```

It coordinates the whole experiment:

1. creates the required directories;
2. saves the experiment configuration;
3. generates the synthetic dataset;
4. saves the dataset;
5. trains the neural network;
6. computes neural network predictions;
7. evaluates neural network metrics;
8. saves model and scalers;
9. runs the Monte Carlo benchmark;
10. saves Monte Carlo metrics;
11. generates diagnostic figures;
12. returns the neural network metrics.

In other words, `pipeline.py` connects all the other modules into one
reproducible experiment.

## Command-Line Script

The entry point for running experiments is:

```text
scripts/run_experiment.py
```

Example final run:

```bash
python scripts/run_experiment.py \
  --n-samples 100000 \
  --max-epochs 200 \
  --batch-size 1024 \
  --mc-n-paths 50000 \
  --mc-evaluation-samples 512 \
  --data-dir data/final \
  --output-dir outputs/final
```

The script builds an `ExperimentConfig` object and calls:

```python
run_experiment(config)
```

## Tests

Tests are stored in:

```text
tests/
```

The test files are:

```text
test_black_scholes.py
test_dataset.py
test_monte_carlo.py
test_pipeline.py
```

They verify:

- analytical Black-Scholes pricing;
- synthetic dataset generation;
- Monte Carlo pricing;
- the full end-to-end pipeline through a small smoke test.

Run tests with:

```bash
pytest
```

## Outputs

Generated experiment outputs are stored in:

```text
data/
outputs/
```

These directories are ignored by Git because their content is reproducible from
the code and configuration.

The selected final results are stored in:

```text
results/final/
```

This directory contains:

- final experiment configuration;
- final neural network metrics;
- final Monte Carlo metrics;
- final diagnostic figures;
- final verification information in the README.

## One-Sentence Summary

The code is modular: each file has one clear responsibility, while
`pipeline.py` coordinates all components to run the full experiment in a
reproducible way.
