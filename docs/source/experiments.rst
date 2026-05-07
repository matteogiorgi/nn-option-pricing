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

Suggested Final Experiments
---------------------------

For the final project report, useful experiments include:

* increasing the synthetic dataset size;
* comparing different hidden layer widths and depths;
* comparing activation functions such as ReLU, Tanh, and LeakyReLU;
* measuring how the error changes with moneyness, maturity, and volatility;
* comparing inference time against Monte Carlo simulation.

