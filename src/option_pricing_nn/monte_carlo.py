"""Monte Carlo estimators for European option pricing.

The current estimator prices European calls under Black-Scholes dynamics by
sampling the exact terminal distribution of geometric Brownian motion. Since the
option is European and path-independent, simulating intermediate time steps is
unnecessary.
"""

from __future__ import annotations

import numpy as np

from option_pricing_nn.black_scholes import call_payoff


def call_price_mc(
    s0: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    n_paths: int = 100_000,
    option_batch_size: int = 512,
    path_batch_size: int = 10_000,
    seed: int = 123,
) -> np.ndarray:
    """Estimate European call prices with Monte Carlo simulation.

    Parameters
    ----------
    s0, k, t, r, sigma
        Arrays containing Black-Scholes parameters for multiple options.
    n_paths
        Number of simulated terminal prices per option.
    option_batch_size
        Number of options processed at the same time.
    path_batch_size
        Number of Monte Carlo paths simulated at the same time.
    seed
        Random seed used by NumPy's random number generator.

    Returns
    -------
    np.ndarray
        Monte Carlo estimates of the discounted expected call payoff.

    The implementation batches over both options and simulation paths, so
    memory usage is bounded by roughly `option_batch_size * path_batch_size`.
    """
    s0 = np.asarray(s0, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    rng = np.random.default_rng(seed)
    estimates = np.empty_like(s0, dtype=np.float64)

    # Outer batching controls how many distinct options are priced at once.
    for start in range(0, len(s0), option_batch_size):
        end = min(start + option_batch_size, len(s0))
        sl = slice(start, end)
        payoff_sum = np.zeros(end - start, dtype=np.float64)
        sampled_paths = 0

        # Inner batching controls the number of simulated paths in memory.
        # This is important because a fully vectorized matrix with shape
        # (n_options, n_paths) can become very large in final experiments.
        for path_start in range(0, n_paths, path_batch_size):
            current_paths = min(path_batch_size, n_paths - path_start)
            z = rng.standard_normal(size=(end - start, current_paths))
            # Exact terminal GBM sampling:
            # S_T = S_0 exp((r - 0.5 sigma^2)T + sigma sqrt(T) Z).
            drift = (r[sl, None] - 0.5 * sigma[sl, None] ** 2) * t[sl, None]
            diffusion = sigma[sl, None] * np.sqrt(t[sl, None]) * z
            terminal_price = s0[sl, None] * np.exp(drift + diffusion)
            payoff_sum += call_payoff(terminal_price, k[sl, None]).sum(axis=1)
            sampled_paths += current_paths

        estimates[sl] = np.exp(-r[sl] * t[sl]) * payoff_sum / sampled_paths

    return estimates
