"""CLI entry point for evaluating a trained BC agent."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from il_based_rl.agent import BCAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BC agent")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium environment ID")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256], help="Hidden layer dimensions")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    agent = BCAgent.load(args.checkpoint, hidden_dims=args.hidden_dims)

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)

    rewards: list[float] = []
    for ep in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + ep if args.seed is not None else None)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
        rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()
    print(f"\nMean reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")


if __name__ == "__main__":
    main()
