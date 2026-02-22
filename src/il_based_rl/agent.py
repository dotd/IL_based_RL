"""Behavioral Cloning agent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from il_based_rl.dataset import DemoDataset
from il_based_rl.policy import MLPPolicy


class BCAgent:
    """Behavioral Cloning agent: wraps a policy for training and inference."""

    def __init__(
        self,
        policy: MLPPolicy,
        lr: float = 1e-3,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        if policy.continuous:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        dataset: DemoDataset,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> list[float]:
        """Train the policy via supervised learning. Returns per-epoch losses."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.policy.train()
        epoch_losses: list[float] = []

        for _ in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for obs, actions in loader:
                pred = self.policy(obs)
                if not self.policy.continuous:
                    actions = actions.long().squeeze(-1)
                loss = self.loss_fn(pred, actions)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            epoch_losses.append(total_loss / max(n_batches, 1))

        return epoch_losses

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return action for a single observation."""
        self.policy.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = self.policy.predict(obs_t)
        return action_t.squeeze(0).numpy()

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "obs_dim": self.policy.obs_dim,
                "action_dim": self.policy.action_dim,
                "continuous": self.policy.continuous,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, hidden_dims: list[int] = (256, 256), lr: float = 1e-3) -> BCAgent:
        """Load agent from checkpoint."""
        ckpt = torch.load(path, weights_only=True)
        policy = MLPPolicy(
            obs_dim=ckpt["obs_dim"],
            action_dim=ckpt["action_dim"],
            hidden_dims=hidden_dims,
            continuous=ckpt["continuous"],
        )
        policy.load_state_dict(ckpt["policy_state_dict"])
        return cls(policy, lr=lr)
