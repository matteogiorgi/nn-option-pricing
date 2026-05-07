"""Analytical Black-Scholes pricing utilities.

This module contains the closed-form European call pricing formula used as the
ground-truth target for the supervised learning problem. Functions are written
to accept both scalars and NumPy arrays, so they can be used for single-option
checks and for vectorized dataset generation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def call_price(
    s0: np.ndarray | float,
    k: np.ndarray | float,
    t: np.ndarray | float,
    r: np.ndarray | float,
    sigma: np.ndarray | float,
) -> np.ndarray:
    """Compute European call prices with the Black-Scholes formula.

    Parameters
    ----------
    s0
        Initial price of the underlying asset.
    k
        Strike price.
    t
        Time to maturity, expressed in years.
    r
        Continuously compounded risk-free rate.
    sigma
        Constant volatility of the underlying asset.

    Returns
    -------
    np.ndarray
        Black-Scholes European call prices. The returned value is always
        non-negative and follows NumPy broadcasting rules.
    """
    s0_arr = np.asarray(s0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    # Avoid division by zero in edge cases. The configured dataset ranges keep
    # T and sigma strictly positive, but this makes the formula robust for
    # ad-hoc checks and future experiments.
    eps = np.finfo(np.float64).eps
    safe_t = np.maximum(t_arr, eps)
    safe_sigma = np.maximum(sigma_arr, eps)
    sqrt_t = np.sqrt(safe_t)

    d1 = (np.log(s0_arr / k_arr) + (r_arr + 0.5 * safe_sigma**2) * safe_t) / (
        safe_sigma * sqrt_t
    )
    d2 = d1 - safe_sigma * sqrt_t

    price = s0_arr * norm.cdf(d1) - k_arr * np.exp(-r_arr * safe_t) * norm.cdf(d2)
    return np.maximum(price, 0.0)


def call_payoff(terminal_price: np.ndarray, strike: np.ndarray | float) -> np.ndarray:
    """Compute the European call payoff at maturity.

    Parameters
    ----------
    terminal_price
        Simulated or observed underlying price at maturity.
    strike
        Option strike price.

    Returns
    -------
    np.ndarray
        Payoff ``max(S_T - K, 0)``.
    """
    return np.maximum(terminal_price - strike, 0.0)
