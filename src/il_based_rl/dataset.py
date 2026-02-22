"""Dataset for storing expert demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    """Stores (observation, action) pairs from expert demonstrations."""

    def __init__(self, observations: np.ndarray, actions: np.ndarray) -> None:
        assert len(observations) == len(actions), "obs and actions must have same length"
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.observations[idx]),
            torch.tensor(self.actions[idx]),
        )

    @classmethod
    def from_trajectories(
        cls,
        observations: list[np.ndarray],
        actions: list[np.ndarray],
    ) -> DemoDataset:
        """Create dataset from lists of per-episode arrays."""
        all_obs = np.concatenate(observations, axis=0)
        all_act = np.concatenate(actions, axis=0)
        return cls(all_obs, all_act)

    def save(self, path: str | Path) -> None:
        """Save dataset to .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, observations=self.observations, actions=self.actions)

    @classmethod
    def load(cls, path: str | Path) -> DemoDataset:
        """Load dataset from .npz file."""
        data = np.load(path)
        return cls(data["observations"], data["actions"])
