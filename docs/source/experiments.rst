Experiments
===========

The experiments verify that the end-to-end pipeline works and that the neural
network can learn the Black-Scholes pricing function.

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
checkpoints remain reproducible but untracked in ``data/final_improved/`` and
``outputs/final_improved/``.

The final run uses:

* 100,000 synthetic option contracts;
* the engineered feature set ``with_moneyness``;
* the SiLU hidden-layer activation function;
* up to 200 training epochs with early stopping;
* 50,000 Monte Carlo paths;
* a deterministic Monte Carlo evaluation subset of 512 test options.

The final neural network metrics against analytical Black-Scholes prices are:

.. list-table::
   :header-rows: 1

   * - Metric
     - Value
   * - MAE
     - 0.0428955592
   * - RMSE
     - 0.0620807954
   * - :math:`R^2`
     - 0.9999927878
   * - MAPE, price > 1
     - 0.4802504182%

Additional Experiments
----------------------

The codebase already supports two controlled experimental switches:

* ``--feature-set with_moneyness`` to include ``s0 / k`` as an engineered
  model input;
* ``--activation`` to compare hidden-layer activation functions such as ReLU,
  Tanh, LeakyReLU, SiLU, and GELU.

The repository also includes a reduced-scale Support Vector Regression
benchmark through ``scripts/run_svr_benchmark.py``. SVR is evaluated on smaller
synthetic datasets and over multiple random seeds because kernel methods do not
scale as naturally to the full 100,000-sample setting used by the neural
network experiment.

The noisy-target robustness experiment is available through
``scripts/run_noisy_targets_experiment.py``. It perturbs only the training
labels with controlled Gaussian noise and evaluates predictions against the
clean analytical Black-Scholes prices. This keeps the experiment focused on
robustness to imperfect labels rather than on market-price modeling.

The same controlled noisy-target setting is also repeated for the reduced-scale
SVR baseline through ``scripts/run_noisy_svr_benchmark.py``. This provides a
classical ML robustness reference while keeping SVR experiments computationally
bounded.

The runtime comparison is available through ``scripts/benchmark_runtime.py``.
It separates neural-network training time from pricing time and compares
analytical Black-Scholes evaluation, neural-network inference, and Monte Carlo
simulation.

The figures used in the report are generated with
``scripts/generate_report_figures.py`` from tracked experiment summaries. This
keeps the report figures reproducible without committing intermediate plotting
work.

Future Extensions
-----------------

Possible future extensions include:

* increasing the synthetic dataset size;
* comparing different hidden layer widths and depths;
* measuring how the error changes with moneyness, maturity, and volatility;
* estimating Greeks through automatic differentiation;
* extending the pricing setup to stochastic-volatility, path-dependent, or
  American-style derivatives;
* using real exchange-traded option data, which would change the research
  question from approximating Black-Scholes to modeling market prices.
