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

Main Components
---------------

The project contains the following components:

* analytical Black-Scholes pricing;
* Monte Carlo pricing under the same dynamics;
* synthetic dataset generation;
* feed-forward neural network training with PyTorch;
* evaluation metrics and diagnostic plots;
* a LaTeX report for the final project submission;
* a Beamer presentation for the oral discussion.

Final Deliverables
------------------

The final experiment results are reflected in the tracked project deliverables:

* ``results/final/`` contains the selected final configuration, metrics,
  figures, and verification notes;
* ``report/main.pdf`` contains the final technical report generated from the
  LaTeX sources in ``report/``;
* ``presentation/main.pdf`` contains the final Beamer slide deck generated
  from ``presentation/main.tex``.

Generated datasets, trained model checkpoints, scalers, and full run outputs
are kept reproducible but untracked under ``data/final/`` and
``outputs/final/``.

Repository Layout
-----------------

.. code-block:: text

   src/nn_option_pricing/
       black_scholes.py
       monte_carlo.py
       dataset.py
       model.py
       training.py
       evaluation.py
       plots.py
       pipeline.py
   scripts/
       run_experiment.py
   report/
       main.tex
       main.pdf
   presentation/
       main.tex
       main.pdf
   results/
       final/
   docs/
       source/
