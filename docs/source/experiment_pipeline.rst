Experiment Pipeline
===================

The command-line entry point is:

.. code-block:: bash

   python scripts/run_experiment.py

Different runs can be isolated by choosing explicit data and output
directories:

.. code-block:: bash

   python scripts/run_experiment.py \
     --data-dir data/intermediate \
     --output-dir outputs/intermediate

The baseline experiment uses the primitive Black-Scholes inputs
``(s0, k, t, r, sigma)`` and ReLU activations. The command-line interface also
supports controlled experimental variants:

.. code-block:: bash

   python scripts/run_experiment.py \
     --feature-set with_moneyness \
     --activation silu \
     --data-dir data/experiments/moneyness_silu \
     --output-dir outputs/experiments/moneyness_silu

Supported feature sets are ``base`` and ``with_moneyness``. Supported
activation functions are ``relu``, ``tanh``, ``leaky_relu``, ``silu``, and
``gelu``.

The pipeline performs the following steps:

1. Create output directories.
2. Generate a synthetic dataset.
3. Compute analytical Black-Scholes labels.
4. Split the data into train, validation, and test sets.
5. Fit input and target scalers.
6. Train a feed-forward neural network with early stopping.
7. Evaluate the network on the held-out test set.
8. Run a Monte Carlo benchmark on a deterministic test subset.
9. Save metrics, model artifacts, and diagnostic figures.

Generated Artifacts
-------------------

The experiment writes:

.. code-block:: text

   data/synthetic_options.csv
   outputs/experiment_config.json
   outputs/model.pt
   outputs/scaler.joblib
   outputs/target_scaler.joblib
   outputs/metrics/nn_metrics.json
   outputs/metrics/monte_carlo_vs_black_scholes_metrics.json
   outputs/figures/*.png

The generated data and experiment outputs are intentionally ignored by Git,
because they can be reproduced from the code and configuration.

Runtime Benchmark
-----------------

The runtime benchmark compares analytical Black-Scholes pricing, neural network
inference, and Monte Carlo simulation:

.. code-block:: bash

   python scripts/benchmark_runtime.py \
     --feature-set with_moneyness \
     --activation silu \
     --output-dir outputs/runtime_benchmark

The benchmark also reports the neural network training time separately from
pricing/inference time. This distinction is important because a neural pricing
surrogate has an upfront training cost, but can be fast once trained.

Fast Smoke Test
---------------

For a quick local check:

.. code-block:: bash

   python scripts/run_experiment.py \
     --n-samples 10000 \
     --max-epochs 40 \
     --batch-size 1024 \
     --mc-n-paths 5000 \
     --mc-evaluation-samples 128 \
     --data-dir data/smoke \
     --output-dir outputs/smoke

Intermediate Experiment
-----------------------

Before running the final experiment, a useful intermediate configuration is:

.. code-block:: bash

   python scripts/run_experiment.py \
     --n-samples 50000 \
     --max-epochs 100 \
     --batch-size 1024 \
     --mc-n-paths 20000 \
     --mc-evaluation-samples 256 \
     --data-dir data/intermediate \
     --output-dir outputs/intermediate
