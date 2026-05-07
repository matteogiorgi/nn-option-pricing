import numpy as np

from option_pricing_nn.black_scholes import call_price


def test_black_scholes_known_atm_case():
    price = call_price(100.0, 100.0, 1.0, 0.05, 0.2)
    assert np.isclose(price, 10.45058357, atol=1e-6)


def test_black_scholes_vectorized_output_shape():
    price = call_price(
        np.array([100.0, 120.0]),
        np.array([100.0, 100.0]),
        np.array([1.0, 0.5]),
        np.array([0.05, 0.03]),
        np.array([0.2, 0.25]),
    )
    assert price.shape == (2,)
    assert np.all(price >= 0.0)
