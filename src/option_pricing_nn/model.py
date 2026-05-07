"""PyTorch model definitions for option pricing."""

from __future__ import annotations

import torch
from torch import nn


class PricingMLP(nn.Module):
    """Feed-forward neural network for scalar option price regression.

    Parameters
    ----------
    input_dim
        Number of input features. The default corresponds to
        ``(S0, K, T, r, sigma)``.
    hidden_layers
        Width of each hidden layer. ReLU activations are inserted between
        linear layers.
    """

    def __init__(
        self, input_dim: int = 5, hidden_layers: tuple[int, ...] = (64, 64, 32)
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one predicted option price per input row."""
        return self.net(x).squeeze(-1)
