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
    """Vectorized Black-Scholes price for European call options."""
    s0_arr = np.asarray(s0, dtype=np.float64)
    k_arr = np.asarray(k, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    r_arr = np.asarray(r, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    eps = np.finfo(np.float64).eps
    safe_t = np.maximum(t_arr, eps)
    safe_sigma = np.maximum(sigma_arr, eps)
    sqrt_t = np.sqrt(safe_t)

    d1 = (
        np.log(s0_arr / k_arr)
        + (r_arr + 0.5 * safe_sigma**2) * safe_t
    ) / (safe_sigma * sqrt_t)
    d2 = d1 - safe_sigma * sqrt_t

    price = s0_arr * norm.cdf(d1) - k_arr * np.exp(-r_arr * safe_t) * norm.cdf(d2)
    return np.maximum(price, 0.0)


def call_payoff(terminal_price: np.ndarray, strike: np.ndarray | float) -> np.ndarray:
    """European call payoff."""
    return np.maximum(terminal_price - strike, 0.0)

