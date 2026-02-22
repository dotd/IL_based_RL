"""Rollout buffer for on-policy algorithms (PPO)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Stores rollout data and computes GAE advantages."""

    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    returns: np.ndarray | None = field(default=None, init=False)
    advantages: np.ndarray | None = field(default=None, init=False)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Add a single transition."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self) -> int:
        return len(self.obs)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and returns."""
        n = len(self)
        advantages = np.zeros(n, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_non_terminal = 1.0 - (dones[t] if t < n - 1 else 0.0)
            # If t is the last step, use last_value as bootstrap; non-terminal mask
            # applies to the *current* done flag only when stepping to next
            if t == n - 1:
                next_non_terminal = 1.0 - dones[t]
            else:
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def get_batches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Yield shuffled minibatches as dicts of tensors."""
        n = len(self)
        indices = np.random.permutation(n)

        obs_arr = np.array(self.obs, dtype=np.float32)
        actions_arr = np.array(self.actions, dtype=np.float32)
        log_probs_arr = np.array(self.log_probs, dtype=np.float32)

        batches: list[dict[str, torch.Tensor]] = []
        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batches.append(
                {
                    "obs": torch.tensor(obs_arr[idx]),
                    "actions": torch.tensor(actions_arr[idx]),
                    "old_log_probs": torch.tensor(log_probs_arr[idx]),
                    "returns": torch.tensor(self.returns[idx]),
                    "advantages": torch.tensor(self.advantages[idx]),
                }
            )
        return batches
