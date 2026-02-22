"""CLI entry point for training a Behavioral Cloning agent."""

from __future__ import annotations

import argparse

from il_based_rl.agent import BCAgent
from il_based_rl.dataset import DemoDataset
from il_based_rl.policy import MLPPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Behavioral Cloning agent")
    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4", help="Gymnasium environment ID")
    parser.add_argument("--demo-path", type=str, required=True, help="Path to demonstrations .npz file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256], help="Hidden layer dimensions")
    parser.add_argument("--save-path", type=str, default="checkpoints/bc_agent.pt", help="Path to save checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DemoDataset.load(args.demo_path)
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]

    policy = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.hidden_dims,
        continuous=True,
    )
    agent = BCAgent(policy, lr=args.lr)

    print(f"Training BC agent on {len(dataset)} samples for {args.epochs} epochs...")
    losses = agent.train(dataset, epochs=args.epochs, batch_size=args.batch_size)

    for i, loss in enumerate(losses):
        print(f"  Epoch {i + 1:3d}: loss = {loss:.6f}")

    agent.save(args.save_path)
    print(f"Checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
