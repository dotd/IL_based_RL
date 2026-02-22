"""Collect expert demonstrations from a Gymnasium environment."""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from il_based_rl.agent import BCAgent
from il_based_rl.dataset import DemoDataset


def collect_demos(
    env_id: str,
    agent: BCAgent | None = None,
    num_episodes: int = 10,
    seed: int | None = None,
) -> DemoDataset:
    """Collect demonstrations by rolling out a policy (or random if agent is None)."""
    env = gym.make(env_id)
    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []

    for ep in range(num_episodes):
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []

        obs, _ = env.reset(seed=seed + ep if seed is not None else None)
        done = False
        while not done:
            if agent is not None:
                action = agent.predict(obs)
            else:
                action = env.action_space.sample()
            ep_obs.append(obs)
            ep_actions.append(np.asarray(action, dtype=np.float32))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        all_obs.append(np.array(ep_obs))
        all_actions.append(np.array(ep_actions))

    env.close()
    return DemoDataset.from_trajectories(all_obs, all_actions)
