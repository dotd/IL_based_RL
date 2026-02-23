"""End-to-end pipeline: Train PPO expert → Collect demos → Train BC student."""

from __future__ import annotations

import argparse
from datetime import datetime

import gymnasium as gym
import numpy as np

from il_based_rl.actor_critic_policy import ActorCriticPolicy
from il_based_rl.agent import BCAgent
from il_based_rl.collect import collect_demos
from il_based_rl.dataset import DemoDataset
from il_based_rl.policy import MLPPolicy
from il_based_rl.ppo_agent import PPOAgent


def run_pipeline(
    env_id: str = "Pendulum-v1",
    ppo_timesteps: int = 200_000,
    ppo_hidden_dims: list[int] | None = None,
    ppo_lr: float = 3e-4,
    num_demo_episodes: int = 50,
    bc_epochs: int = 100,
    bc_hidden_dims: list[int] | None = None,
    bc_lr: float = 1e-3,
    bc_batch_size: int = 64,
    seed: int | None = None,
    ppo_save_path: str = "checkpoints/ppo_expert.pt",
    demo_save_path: str = "demos/expert_demos.npz",
    bc_save_path: str = "checkpoints/bc_student.pt",
    eval_episodes: int = 20,
    wandb_log: bool = False,
    wandb_project: str | None = None,
) -> None:
    """Run the full RL-to-IL pipeline."""
    if ppo_hidden_dims is None:
        ppo_hidden_dims = [64, 64]
    if bc_hidden_dims is None:
        bc_hidden_dims = [256, 256]

    # --- Infer environment dimensions ---
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    # =====================================================================
    # Stage 1: Train PPO expert
    # =====================================================================
    print("=" * 60)
    print("STAGE 1: Training PPO expert")
    print("=" * 60)

    ppo_policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=ppo_hidden_dims,
    )
    ppo_agent = PPOAgent(ppo_policy, lr=ppo_lr)

    wandb_run_name = None
    if wandb_log:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        wandb_run_name = f"run_{stamp}_{env_id}_{ppo_timesteps}"

    ppo_agent.train(
        env_id,
        total_timesteps=ppo_timesteps,
        seed=seed,
        wandb_log=wandb_log,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    ppo_save_path = ppo_agent.save(ppo_save_path, timestamp=True)
    print(f"PPO expert saved to {ppo_save_path}\n")

    # =====================================================================
    # Stage 2: Collect expert demonstrations
    # =====================================================================
    print("=" * 60)
    print("STAGE 2: Collecting expert demonstrations")
    print("=" * 60)

    dataset = collect_demos(env_id, agent=ppo_agent, num_episodes=num_demo_episodes, seed=seed)
    dataset.save(demo_save_path)
    print(f"Collected {len(dataset)} transitions from {num_demo_episodes} episodes")
    print(f"Demos saved to {demo_save_path}\n")

    # =====================================================================
    # Stage 3: Train BC student
    # =====================================================================
    print("=" * 60)
    print("STAGE 3: Training BC student on expert demos")
    print("=" * 60)

    bc_policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=bc_hidden_dims,
        continuous=True,
    )
    bc_agent = BCAgent(bc_policy, lr=bc_lr)
    losses = bc_agent.train(dataset, epochs=bc_epochs, batch_size=bc_batch_size)

    print(f"  Epoch   1: loss = {losses[0]:.6f}")
    if len(losses) > 2:
        mid = len(losses) // 2
        print(f"  Epoch {mid + 1:3d}: loss = {losses[mid]:.6f}")
    print(f"  Epoch {len(losses):3d}: loss = {losses[-1]:.6f}")

    bc_agent.save(bc_save_path)
    print(f"BC student saved to {bc_save_path}\n")

    # =====================================================================
    # Evaluation: Compare random, PPO expert, and BC student
    # =====================================================================
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    def evaluate_agent(agent, label: str) -> float:
        env = gym.make(env_id)
        rewards: list[float] = []
        for ep in range(eval_episodes):
            obs, _ = env.reset(seed=seed + ep if seed is not None else None)
            total_reward = 0.0
            done = False
            while not done:
                if agent is not None:
                    action = agent.predict(obs)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                done = terminated or truncated
            rewards.append(total_reward)
        env.close()
        mean = float(np.mean(rewards))
        std = float(np.std(rewards))
        print(f"  {label:20s}  reward = {mean:8.2f} +/- {std:.2f}")
        return mean

    evaluate_agent(None, "Random policy")
    evaluate_agent(ppo_agent, "PPO expert")
    evaluate_agent(bc_agent, "BC student")

    print("\nPipeline complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: PPO expert -> collect demos -> BC student"
    )
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Gymnasium environment ID")
    parser.add_argument("--ppo-timesteps", type=int, default=200_000, help="PPO training timesteps")
    parser.add_argument("--ppo-lr", type=float, default=3e-4, help="PPO learning rate")
    parser.add_argument("--ppo-hidden-dims", type=int, nargs="+", default=[64, 64], help="PPO hidden layers")
    parser.add_argument("--num-demo-episodes", type=int, default=50, help="Number of demo episodes to collect")
    parser.add_argument("--bc-epochs", type=int, default=100, help="BC training epochs")
    parser.add_argument("--bc-lr", type=float, default=1e-3, help="BC learning rate")
    parser.add_argument("--bc-hidden-dims", type=int, nargs="+", default=[256, 256], help="BC hidden layers")
    parser.add_argument("--bc-batch-size", type=int, default=64, help="BC training batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--ppo-save-path", type=str, default="checkpoints/ppo_expert.pt")
    parser.add_argument("--demo-save-path", type=str, default="demos/expert_demos.npz")
    parser.add_argument("--bc-save-path", type=str, default="checkpoints/bc_student.pt")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="il-based-rl", help="W&B project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        env_id=args.env_id,
        ppo_timesteps=args.ppo_timesteps,
        ppo_hidden_dims=args.ppo_hidden_dims,
        ppo_lr=args.ppo_lr,
        num_demo_episodes=args.num_demo_episodes,
        bc_epochs=args.bc_epochs,
        bc_hidden_dims=args.bc_hidden_dims,
        bc_lr=args.bc_lr,
        bc_batch_size=args.bc_batch_size,
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        ppo_save_path=args.ppo_save_path,
        demo_save_path=args.demo_save_path,
        bc_save_path=args.bc_save_path,
        wandb_log=args.wandb,
        wandb_project=args.wandb_project,
    )
