"""MLP policy network for behavioral cloning."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """Multilayer perceptron policy.

    Supports continuous (outputs mean action) and discrete (outputs logits)
    action spaces.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (256, 256),
        continuous: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous

        layers: list[nn.Module] = []
        prev_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action mean (continuous) or logits (discrete)."""
        return self.net(obs)

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action for inference."""
        with torch.no_grad():
            out = self.forward(obs)
            if not self.continuous:
                out = out.argmax(dim=-1)
            return out
