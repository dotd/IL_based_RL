"""PPO (Proximal Policy Optimization) agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from il_based_rl.actor_critic_policy import ActorCriticPolicy
from il_based_rl.buffer import RolloutBuffer


class PPOAgent:
    """PPO agent wrapping an ActorCriticPolicy."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return deterministic action for a single observation."""
        self.policy.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_t = self.policy.predict(obs_t)
        return action_t.squeeze(0).numpy()

    def collect_rollouts(self, env: gym.Env) -> tuple[RolloutBuffer, np.ndarray]:
        """Collect n_steps of experience. Returns (buffer, last_obs).

        Handles mid-rollout resets by bootstrapping with the value estimate.
        """
        self.policy.eval()
        buffer = RolloutBuffer()

        obs, _ = env.reset()
        for _ in range(self.n_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value = self.policy.get_distribution(obs_t)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs=obs,
                action=action_np,
                reward=float(reward),
                done=done,
                log_prob=log_prob.item(),
                value=value.item(),
            )

            if done:
                obs, _ = env.reset()
            else:
                obs = next_obs

        # Bootstrap last value
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            _, last_value = self.policy.forward(obs_t)
        buffer.compute_returns_and_advantages(
            last_value.item(), gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        return buffer, obs

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run PPO update for n_epochs over the buffer. Returns mean losses."""
        self.policy.train()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                log_prob, value, entropy = self.policy.evaluate_actions(
                    batch["obs"], batch["actions"]
                )

                # Normalize advantages
                adv = batch["advantages"]
                if len(adv) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped surrogate objective
                ratio = (log_prob - batch["old_log_probs"]).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((value - batch["returns"]) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def train(
        self,
        env_id: str,
        total_timesteps: int,
        eval_interval: int = 10_000,
        eval_episodes: int = 5,
        seed: int | None = None,
        wandb_log: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> list[dict[str, float]]:
        """Full PPO training loop. Returns list of update info dicts."""
        # --- Optional W&B initialisation ---
        wb = None
        if wandb_log:
            import wandb

            wb = wandb
            wb.init(
                project=wandb_project or "il-based-rl",
                name=wandb_run_name,
                config={
                    "env_id": env_id,
                    "total_timesteps": total_timesteps,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_range": self.clip_range,
                    "n_steps": self.n_steps,
                    "n_epochs": self.n_epochs,
                    "batch_size": self.batch_size,
                    "entropy_coef": self.entropy_coef,
                    "max_grad_norm": self.max_grad_norm,
                    "obs_dim": self.policy.obs_dim,
                    "action_dim": self.policy.action_dim,
                    "seed": seed,
                },
            )

        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)

        logs: list[dict[str, float]] = []
        timesteps_so_far = 0

        while timesteps_so_far < total_timesteps:
            buffer, _ = self.collect_rollouts(env)
            info = self.update(buffer)
            timesteps_so_far += self.n_steps
            info["timesteps"] = timesteps_so_far
            logs.append(info)

            if wb is not None:
                wb.log(
                    {
                        "policy_loss": info["policy_loss"],
                        "value_loss": info["value_loss"],
                        "entropy": info["entropy"],
                    },
                    step=timesteps_so_far,
                )

            # Periodic evaluation
            if timesteps_so_far % eval_interval < self.n_steps:
                mean_reward = self._evaluate(env_id, eval_episodes, seed)
                info["eval_reward"] = mean_reward
                print(
                    f"[{timesteps_so_far:>8d}] "
                    f"policy_loss={info['policy_loss']:.4f}  "
                    f"value_loss={info['value_loss']:.4f}  "
                    f"eval_reward={mean_reward:.2f}"
                )
                if wb is not None:
                    wb.log({"eval_reward": mean_reward}, step=timesteps_so_far)

        env.close()
        if wb is not None:
            wb.finish()
        return logs

    def _evaluate(self, env_id: str, num_episodes: int, seed: int | None) -> float:
        """Run evaluation episodes and return mean reward."""
        env = gym.make(env_id)
        rewards: list[float] = []
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep if seed is not None else None)
            total_reward = 0.0
            done = False
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            rewards.append(total_reward)
        env.close()
        return float(np.mean(rewards))

    @staticmethod
    def timestamped_path(path: str | Path) -> Path:
        """Insert a timestamp before the file extension, e.g. ppo_agent_20260223_0910.pt."""
        path = Path(path)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        return path.with_stem(f"{path.stem}_{stamp}")

    def save(self, path: str | Path, timestamp: bool = False) -> Path:
        """Save model checkpoint. Returns the actual path written.

        If *timestamp* is True, a ``_YYYYMMDD_HHMM`` suffix is appended to the
        filename so that successive runs never overwrite each other.
        """
        path = Path(path)
        if timestamp:
            path = self.timestamped_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "obs_dim": self.policy.obs_dim,
                "action_dim": self.policy.action_dim,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        hidden_dims: list[int] = (64, 64),
        **kwargs,
    ) -> PPOAgent:
        """Load agent from checkpoint."""
        ckpt = torch.load(path, weights_only=True)
        policy = ActorCriticPolicy(
            obs_dim=ckpt["obs_dim"],
            action_dim=ckpt["action_dim"],
            hidden_dims=hidden_dims,
        )
        policy.load_state_dict(ckpt["policy_state_dict"])
        return cls(policy, **kwargs)
