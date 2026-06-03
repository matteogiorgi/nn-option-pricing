Methodology
===========

Black-Scholes Dynamics
----------------------

Under the risk-neutral Black-Scholes model, the underlying asset follows

.. math::

   dS_t = r S_t dt + \sigma S_t dW_t,

where :math:`S_t` is the underlying price, :math:`r` is the risk-free rate,
:math:`\sigma` is constant volatility, and :math:`W_t` is a Brownian motion.

The terminal distribution is

.. math::

   S_T =
   S_0 \exp\left[
      \left(r - \frac{1}{2}\sigma^2\right)T
      + \sigma \sqrt{T} Z
   \right],
   \qquad Z \sim \mathcal{N}(0,1).

European Call Pricing
---------------------

The European call payoff is

.. math::

   \max(S_T - K, 0).

The analytical Black-Scholes price is used as the ground-truth label:

.. math::

   C_{BS} = S_0 N(d_1) - K e^{-rT} N(d_2),

with

.. math::

   d_1 =
   \frac{
      \ln(S_0/K) + \left(r + \frac{1}{2}\sigma^2\right)T
   }{
      \sigma\sqrt{T}
   },
   \qquad
   d_2 = d_1 - \sigma\sqrt{T}.

Supervised Learning Task
------------------------

The neural network is trained as a regression model:

.. math::

   f_\theta(S_0, K, T, r, \sigma) \approx C_{BS}.

Inputs are standardized and the target is scaled during training. Target
scaling improves numerical conditioning because option prices may span a wider
range than the normalized input features.

The final configuration uses the engineered moneyness feature
:math:`S_0/K`, so the learned map becomes

.. math::

   f_\theta(S_0, K, T, r, \sigma, S_0/K) \approx C_{BS}.

Moneyness is included because it summarizes the relative position of the
underlying price with respect to the strike, which is financially informative
for option-pricing regression.

Monte Carlo Benchmark
---------------------

Monte Carlo pricing estimates

.. math::

   \hat{C}_{MC}
   =
   e^{-rT}
   \frac{1}{M}
   \sum_{i=1}^{M}
   \max(S_T^{(i)} - K, 0).

Because the option is European and path-independent, the implementation samples
the exact terminal distribution instead of simulating intermediate time steps.

Controlled Noisy Targets
------------------------

The noisy-target experiments perturb only the training labels:

.. math::

   \widetilde{C}_{BS}
   =
   \max(C_{BS} + \varepsilon, 0),
   \qquad
   \varepsilon \sim \mathcal{N}(0, \tau^2).

The validation and test targets remain clean analytical Black-Scholes prices.
This design measures robustness to imperfect training labels while preserving a
known ground truth for evaluation.

SVR Baseline
------------

Support Vector Regression is used as a classical machine learning reference.
It is evaluated on reduced synthetic datasets because kernel SVR scales less
favorably than the neural-network pipeline when the number of training samples
becomes large. The purpose is therefore comparative and diagnostic, not to
replace the final neural-network experiment.
