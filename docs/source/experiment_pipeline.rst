Experiment Pipeline
===================

The command-line entry point is:

.. code-block:: bash

   python scripts/run_experiment.py

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
   outputs/model.pt
   outputs/scaler.joblib
   outputs/target_scaler.joblib
   outputs/metrics/nn_metrics.json
   outputs/metrics/monte_carlo_vs_black_scholes_metrics.json
   outputs/figures/*.png

The generated data and experiment outputs are intentionally ignored by Git,
because they can be reproduced from the code and configuration.

Fast Smoke Test
---------------

For a quick local check:

.. code-block:: bash

   python scripts/run_experiment.py \
     --n-samples 10000 \
     --max-epochs 40 \
     --batch-size 1024 \
     --mc-n-paths 5000 \
     --mc-evaluation-samples 128

