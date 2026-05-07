import numpy as np

from nn_option_pricing.black_scholes import call_price
from nn_option_pricing.monte_carlo import call_price_mc


def test_monte_carlo_output_shape_and_nonnegative_prices():
    s0 = np.array([80.0, 100.0, 120.0])
    k = np.array([100.0, 100.0, 100.0])
    t = np.array([0.5, 1.0, 1.5])
    r = np.array([0.01, 0.03, 0.05])
    sigma = np.array([0.2, 0.25, 0.3])

    prices = call_price_mc(
        s0,
        k,
        t,
        r,
        sigma,
        n_paths=2_000,
        option_batch_size=2,
        path_batch_size=500,
        seed=11,
    )

    assert prices.shape == s0.shape
    assert np.all(np.isfinite(prices))
    assert np.all(prices >= 0.0)


def test_monte_carlo_is_reproducible_with_seed():
    args = (
        np.array([100.0, 110.0]),
        np.array([100.0, 95.0]),
        np.array([1.0, 0.75]),
        np.array([0.03, 0.02]),
        np.array([0.2, 0.35]),
    )

    first = call_price_mc(*args, n_paths=1_000, path_batch_size=250, seed=123)
    second = call_price_mc(*args, n_paths=1_000, path_batch_size=250, seed=123)

    assert np.array_equal(first, second)


def test_monte_carlo_converges_towards_black_scholes():
    s0 = np.array([100.0])
    k = np.array([100.0])
    t = np.array([1.0])
    r = np.array([0.05])
    sigma = np.array([0.2])
    analytical = call_price(s0, k, t, r, sigma)

    estimate = call_price_mc(
        s0,
        k,
        t,
        r,
        sigma,
        n_paths=80_000,
        option_batch_size=1,
        path_batch_size=10_000,
        seed=321,
    )

    assert np.allclose(estimate, analytical, atol=0.15)
