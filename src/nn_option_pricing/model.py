"""PyTorch model definitions for option pricing."""

from __future__ import annotations

import torch
from torch import nn


def make_activation(name: str) -> nn.Module:
    """Create an activation layer from a stable configuration name."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "leaky_relu":
        return nn.LeakyReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(
        f"Unknown activation={name!r}. "
        "Expected one of: relu, tanh, leaky_relu, silu, gelu."
    )


class PricingMLP(nn.Module):
    """Feed-forward neural network for scalar option price regression.

    Parameters
    ----------
    input_dim
        Number of input features. The default corresponds to
        ``(S0, K, T, r, sigma)``.
    hidden_layers
        Width of each hidden layer.
    activation
        Activation function inserted after each hidden linear layer.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_layers: tuple[int, ...] = (64, 64, 32),
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(make_activation(activation))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one predicted option price per input row."""
        return self.net(x).squeeze(-1)
