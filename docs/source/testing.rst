Testing and Validation
======================

The project includes automated tests to keep the numerical and machine learning
components reliable as the code evolves.

Run the test suite with:

.. code-block:: bash

   pytest

Current Test Coverage
---------------------

The current tests cover:

* analytical Black-Scholes pricing, including a known at-the-money reference
  case and vectorized inputs;
* synthetic dataset generation, including schema, parameter ranges,
  non-negative prices, moneyness, reproducibility, and CSV round trips;
* controlled noisy-target generation, including zero-noise consistency,
  reproducibility, non-negative noisy targets, and experiment output writing;
* Monte Carlo pricing, including output shape, non-negative estimates,
  reproducibility, and approximate convergence to the analytical
  Black-Scholes price;
* model construction and activation-function selection;
* reduced-scale Support Vector Regression benchmarking;
* reduced-scale noisy-target SVR benchmarking;
* a small end-to-end pipeline smoke test, which verifies that a minimal
  experiment can run successfully and write the expected metrics, figures, and
  model artifacts.

Testing Philosophy
------------------

The tests are intentionally split between numerical correctness and integration
checks.

Numerical tests validate the building blocks of the project, while the pipeline
smoke test verifies that data generation, training, evaluation, Monte Carlo
benchmarking, and artifact persistence work together. The smoke test uses a
small temporary experiment configuration, so it remains fast and does not write
outputs into the project directories.
