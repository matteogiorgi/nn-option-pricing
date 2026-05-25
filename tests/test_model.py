"""Tests for neural network model construction."""

import pytest
import torch
from torch import nn

from nn_option_pricing.model import PricingMLP, make_activation


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("relu", nn.ReLU),
        ("tanh", nn.Tanh),
        ("leaky_relu", nn.LeakyReLU),
        ("silu", nn.SiLU),
        ("gelu", nn.GELU),
    ],
)
def test_make_activation_returns_expected_module(name, expected_type):
    assert isinstance(make_activation(name), expected_type)


def test_pricing_mlp_supports_configurable_input_dim_and_activation():
    model = PricingMLP(input_dim=6, hidden_layers=(8, 4), activation="silu")
    x = torch.zeros((3, 6), dtype=torch.float32)

    y = model(x)

    assert y.shape == (3,)


def test_make_activation_rejects_unknown_activation():
    with pytest.raises(ValueError, match="Unknown activation"):
        make_activation("not_an_activation")
