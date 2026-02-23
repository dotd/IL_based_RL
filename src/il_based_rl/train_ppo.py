"""CLI entry point for training a PPO agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from il_based_rl.actor_critic_policy import ActorCriticPolicy
from il_based_rl.ppo_agent import PPOAgent


def _find_latest_checkpoint(save_path: str) -> Path | None:
    """Find the most recently modified checkpoint matching the save_path stem."""
    base = Path(save_path)
    directory = base.parent
    if not directory.is_dir():
        return None
    # Match files like ppo_agent_20260223_0910.pt (same stem prefix + .pt)
    candidates = sorted(
        directory.glob(f"{base.stem}*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


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
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from checkpoint. Use --resume to load the latest, or --resume PATH to load a specific file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym

    # Infer obs/action dims from the environment
    env = gym.make(args.env_id)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    # --- Resume from checkpoint or create fresh agent ---
    if args.resume is not None:
        if args.resume == "latest":
            ckpt_path = _find_latest_checkpoint(args.save_path)
            if ckpt_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found matching '{args.save_path}' stem in "
                    f"{Path(args.save_path).parent}"
                )
        else:
            ckpt_path = Path(args.resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Resuming from {ckpt_path}")
        agent = PPOAgent.load(
            ckpt_path,
            hidden_dims=args.hidden_dims,
            lr=args.lr,
            gamma=args.gamma,
            clip_range=args.clip_range,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            entropy_coef=args.entropy_coef,
        )
    else:
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
