from __future__ import annotations

import torch
from torch import nn


class PricingMLP(nn.Module):
    """Feed-forward neural network for scalar option price regression."""

    def __init__(self, input_dim: int = 5, hidden_layers: tuple[int, ...] = (64, 64, 32)):
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
        return self.net(x).squeeze(-1)

