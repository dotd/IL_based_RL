"""Actor-Critic policy network for PPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int) -> nn.Sequential:
    """Build an MLP with Tanh activations."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.Tanh())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCriticPolicy(nn.Module):
    """Separate actor/critic MLP policy with learnable log_std for continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] = (64, 64),
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.actor = _build_mlp(obs_dim, list(hidden_dims), action_dim)
        self.critic = _build_mlp(obs_dim, list(hidden_dims), 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (action_mean, value)."""
        action_mean = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return action_mean, value

    def get_distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return (Normal distribution, value)."""
        action_mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = Normal(action_mean, std)
        return dist, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions: return (log_prob, value, entropy)."""
        dist, value = self.get_distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic mean action for inference."""
        with torch.no_grad():
            action_mean = self.actor(obs)
            return action_mean
