"""CLI entry point for training a PPO agent."""

from __future__ import annotations

import argparse

from il_based_rl.actor_critic_policy import ActorCriticPolicy
from il_based_rl.ppo_agent import PPOAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1", help="Gymnasium environment ID")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 64], help="Hidden layer dimensions")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per update")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--entropy-coef", type=float, default=0.0, help="Entropy bonus coefficient")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save-path", type=str, default="checkpoints/ppo_agent.pt", help="Path to save checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym

    # Infer obs/action dims from the environment
    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    policy = ActorCriticPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
    )
    agent = PPOAgent(
        policy,
        lr=args.lr,
        gamma=args.gamma,
        clip_range=args.clip_range,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef,
    )

    print(f"Training PPO on {args.env_id} for {args.total_timesteps} timesteps...")
    agent.train(args.env_id, total_timesteps=args.total_timesteps, seed=args.seed)

    saved_path = agent.save(args.save_path, timestamp=True)
    print(f"Checkpoint saved to {saved_path}")


if __name__ == "__main__":
    main()
