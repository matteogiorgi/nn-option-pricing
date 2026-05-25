Experiments
===========

The first experiment is designed to verify that the end-to-end pipeline works
and that the network can learn the Black-Scholes pricing function.

The main neural network metrics are:

* Mean Absolute Error (MAE);
* Root Mean Squared Error (RMSE);
* coefficient of determination :math:`R^2`;
* MAPE on prices greater than 1.

The MAPE restriction avoids unstable percentage errors for options whose
theoretical price is close to zero.

Diagnostic Plots
----------------

The project produces the following plots:

* training and validation loss;
* Black-Scholes price versus neural network prediction;
* distribution of prediction errors;
* absolute error versus moneyness;
* absolute error versus maturity;
* absolute error versus volatility.

Final Experiment
----------------

The final experiment artifacts selected for the report are stored in
``results/final/``. This directory tracks the configuration snapshot, metrics,
diagnostic figures, and verification notes, while generated datasets and model
checkpoints remain reproducible but untracked in ``data/final/`` and
``outputs/final/``.

The final run uses:

* 100,000 synthetic option contracts;
* up to 200 training epochs with early stopping;
* 50,000 Monte Carlo paths;
* a deterministic Monte Carlo evaluation subset of 512 test options.

The final neural network metrics against analytical Black-Scholes prices are:

.. list-table::
   :header-rows: 1

   * - Metric
     - Value
   * - MAE
     - 0.0619787835
   * - RMSE
     - 0.0821850306
   * - :math:`R^2`
     - 0.9999874234
   * - MAPE, price > 1
     - 0.6746048331%

Further Experiment Ideas
------------------------

The codebase already supports two controlled experimental switches:

* ``--feature-set with_moneyness`` to include ``s0 / k`` as an engineered
  model input;
* ``--activation`` to compare hidden-layer activation functions such as ReLU,
  Tanh, LeakyReLU, SiLU, and GELU.

For the final project report, additional useful experiments include:

* increasing the synthetic dataset size;
* comparing different hidden layer widths and depths;
* measuring how the error changes with moneyness, maturity, and volatility;
* comparing inference time against Monte Carlo simulation.
