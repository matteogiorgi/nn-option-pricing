Project Overview
================

The project is built around a controlled supervised learning problem.
Synthetic option contracts are sampled from predefined parameter ranges and
their labels are computed using the analytical Black-Scholes formula.
A feed-forward neural network is then trained to approximate the mapping

.. math::

   (S_0, K, T, r, \sigma) \mapsto C_{BS}.

This setup is intentionally simple and mathematically transparent. Since the
target function is known exactly, model errors can be interpreted directly as
approximation errors with respect to Black-Scholes prices.
The selected final configuration also includes moneyness :math:`S_0/K` as an
engineered input feature.

Main Components
---------------

The project contains the following components:

* analytical Black-Scholes pricing;
* Monte Carlo pricing under the same dynamics;
* synthetic dataset generation;
* feed-forward neural network training with PyTorch;
* feature-engineering and activation-function experiments;
* reduced-scale Support Vector Regression baselines;
* controlled noisy-target robustness experiments;
* runtime benchmarking against Monte Carlo simulation;
* evaluation metrics and diagnostic plots;
* automated tests and Sphinx documentation;
* a LaTeX report for the final project submission;
* a Beamer presentation for the oral discussion.

Final Deliverables
------------------

The final experiment results are reflected in the tracked project deliverables:

* ``results/final/`` contains the selected final configuration, metrics,
  figures, and verification notes;
* ``report/main.pdf`` contains the final technical report generated from the
  LaTeX sources in ``report/``;
* ``report/main_print.pdf`` contains the printable report version with
  black link text and visible PDF link borders;
* ``presentation/main.pdf`` contains the final Beamer slide deck generated
  from ``presentation/main.tex``;
* the Sphinx documentation is generated from ``docs/source/`` and published
  through GitHub Pages.

Generated datasets, trained model checkpoints, scalers, and full run outputs
are kept reproducible but untracked under ``data/final_improved/`` and
``outputs/final_improved/``.

Repository Layout
-----------------

.. code-block:: text

   src/nn_option_pricing/
       black_scholes.py
       config.py
       monte_carlo.py
       dataset.py
       model.py
       training.py
       evaluation.py
       noise.py
       noisy_experiment.py
       noisy_svr_experiment.py
       plots.py
       pipeline.py
       svr.py
   scripts/
       run_experiment.py
       benchmark_runtime.py
       run_svr_benchmark.py
       run_noisy_targets_experiment.py
       run_noisy_svr_benchmark.py
       generate_report_figures.py
   report/
       main.tex
       main.pdf
       main_print.pdf
   presentation/
       main.tex
       main.pdf
   results/
       final/
       experiments/
   docs/
       source/
