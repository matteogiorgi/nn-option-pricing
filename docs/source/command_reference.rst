Command Reference
=================

This page collects the main executable scripts. The commands are intended to be
run from the repository root unless stated otherwise.

Main Experiment
---------------

``scripts/run_experiment.py`` runs the complete neural-network pricing
pipeline:

.. code-block:: bash

   python scripts/run_experiment.py

Important options include:

* ``--n-samples`` for the synthetic dataset size;
* ``--max-epochs`` and ``--batch-size`` for training;
* ``--feature-set`` with values ``base`` or ``with_moneyness``;
* ``--activation`` with values ``relu``, ``tanh``, ``leaky_relu``, ``silu``,
  or ``gelu``;
* ``--mc-n-paths`` and ``--mc-evaluation-samples`` for the Monte Carlo
  benchmark;
* ``--data-dir`` and ``--output-dir`` for isolating experiment artifacts.

Runtime Benchmark
-----------------

``scripts/benchmark_runtime.py`` compares analytical Black-Scholes pricing,
neural-network inference, and Monte Carlo pricing time:

.. code-block:: bash

   python scripts/benchmark_runtime.py \
     --feature-set with_moneyness \
     --activation silu \
     --output-dir results/experiments/runtime_benchmark

The script reports neural-network training time separately from pricing time,
because surrogate pricing has an upfront training cost and a later inference
phase.

SVR Benchmark
-------------

``scripts/run_svr_benchmark.py`` evaluates a reduced-scale Support Vector
Regression baseline:

.. code-block:: bash

   python scripts/run_svr_benchmark.py \
     --n-samples 5000 \
     --seeds 11 42 73 \
     --feature-set with_moneyness \
     --output-dir results/experiments/svr_benchmark

The benchmark is intentionally reduced-scale because RBF kernel SVR does not
scale as naturally as the neural-network pipeline to the full final dataset.

Noisy-Target Neural-Network Experiment
--------------------------------------

``scripts/run_noisy_targets_experiment.py`` trains the neural network on
controlled noisy Black-Scholes labels and evaluates against clean analytical
prices:

.. code-block:: bash

   python scripts/run_noisy_targets_experiment.py \
     --n-samples 50000 \
     --noise-levels 0.0 0.01 0.05 \
     --max-epochs 100 \
     --batch-size 1024 \
     --feature-set with_moneyness \
     --activation silu \
     --seed 42 \
     --noise-seed 123 \
     --data-dir data/experiments/noisy_targets \
     --output-dir outputs/experiments/noisy_targets \
     --results-dir results/experiments/noisy_targets

The experiment perturbs only the training targets. Validation and test errors
are measured against the clean Black-Scholes prices.

Noisy-Target SVR Benchmark
--------------------------

``scripts/run_noisy_svr_benchmark.py`` repeats the controlled noisy-target
setting with SVR:

.. code-block:: bash

   python scripts/run_noisy_svr_benchmark.py \
     --n-samples 5000 \
     --seeds 11 42 73 \
     --noise-levels 0.0 0.01 0.05 \
     --feature-set with_moneyness \
     --output-dir results/experiments/noisy_svr_benchmark

This provides a classical machine-learning robustness reference while keeping
the computational cost bounded.

Report Figures
--------------

``scripts/generate_report_figures.py`` regenerates the compact comparison
figures used in the report from tracked experiment summaries:

.. code-block:: bash

   python scripts/generate_report_figures.py

The generated figures are written under ``report/figures/`` and are used by the
experimental-results chapter of the report.

Testing and Documentation
-------------------------

Run tests with:

.. code-block:: bash

   pytest

Build Sphinx documentation with:

.. code-block:: bash

   sphinx-build -W -b html docs/source docs/build/html
