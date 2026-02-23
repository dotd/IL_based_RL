# IL-based RL

A from-scratch framework that combines **Reinforcement Learning** (PPO) with **Imitation Learning** (Behavioral Cloning) using PyTorch and Gymnasium.

## Why this repo?

Training an RL agent directly on a task can be sample-expensive and unstable. A common alternative is **Behavioral Cloning (BC)** — learning a policy by imitating expert demonstrations via supervised learning. But where do expert demonstrations come from?

This repo closes the loop with a three-stage pipeline:

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   1. Train expert    2. Collect demos    3. Train BC    │
│                                                         │
│   PPO agent  ───►  Expert trajectories  ───►  BC agent  │
│   (RL)              (obs, action) pairs       (IL)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

1. **Train an RL expert** — Use PPO (Proximal Policy Optimization) to train a strong agent on a Gymnasium environment.
2. **Collect demonstrations** — Roll out the trained PPO agent to gather (observation, action) pairs.
3. **Train a BC student** — Use the collected demonstrations to train a lightweight BC policy via supervised learning.

The BC student learns to mimic the expert without ever interacting with the reward function, making it useful for sim-to-real transfer, offline settings, or as a warm-start for further fine-tuning.

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Stage 1 — Train a PPO Expert

Train a PPO agent on `Pendulum-v1` (or any continuous-action Gymnasium environment):

```bash
python -m il_based_rl.train_ppo \
    --env-id Pendulum-v1 \
    --total-timesteps 200000 \
    --lr 3e-4 \
    --n-steps 2048 \
    --n-epochs 10 \
    --batch-size 64 \
    --gamma 0.99 \
    --clip-range 0.2 \
    --seed 42 \
    --save-path checkpoints/ppo_expert.pt
```

You will see periodic evaluation logs showing reward improvement:

```
Training PPO on Pendulum-v1 for 200000 timesteps...
[   10240] policy_loss=-0.0013  value_loss=3451.60  eval_reward=-1135.60
[   51200] policy_loss=-0.0009  value_loss=2230.43  eval_reward=-1177.05
[  100352] policy_loss=-0.0027  value_loss=1485.72  eval_reward=-961.27
...
Checkpoint saved to checkpoints/ppo_expert.pt
```

All CLI options:

| Flag | Default | Description |
|---|---|---|
| `--env-id` | `Pendulum-v1` | Gymnasium environment ID |
| `--total-timesteps` | `200000` | Total training timesteps |
| `--lr` | `3e-4` | Learning rate |
| `--hidden-dims` | `64 64` | Hidden layer sizes |
| `--gamma` | `0.99` | Discount factor |
| `--clip-range` | `0.2` | PPO clip range |
| `--n-steps` | `2048` | Rollout length per update |
| `--n-epochs` | `10` | PPO epochs per update |
| `--batch-size` | `64` | Minibatch size |
| `--entropy-coef` | `0.0` | Entropy bonus coefficient |
| `--seed` | `None` | Random seed |
| `--save-path` | `checkpoints/ppo_agent.pt` | Checkpoint path |
| `--resume` | — | Resume from latest checkpoint, or `--resume PATH` for a specific file |
| `--wandb` | off | Enable Weights & Biases logging |
| `--wandb-project` | `il-based-rl` | W&B project name |
| `--wandb-run-name` | auto | W&B run name (auto: `run_YYYYMMDD_HHMM_<env>_<steps>`) |

## Weights & Biases Integration

Training metrics can be logged to [Weights & Biases](https://wandb.ai) for live visualization of losses and reward curves.

### Setup

```bash
pip install -e ".[wandb]"
wandb login
```

### Usage

Add `--wandb` to any training command:

```bash
python -m il_based_rl.train_ppo \
    --env-id Pendulum-v1 \
    --total-timesteps 200000 \
    --seed 42 \
    --wandb
```

A run name is auto-generated (e.g. `run_20260223_0951_Pendulum-v1_200000`). To set a custom name:

```bash
python -m il_based_rl.train_ppo --wandb --wandb-run-name "my-experiment"
```

The full pipeline also supports wandb:

```bash
python scripts/run_pipeline.py --wandb --ppo-timesteps 200000
```

### Logged metrics

| Metric | Logged every | Description |
|---|---|---|
| `policy_loss` | update | PPO clipped surrogate loss |
| `value_loss` | update | Critic MSE loss |
| `entropy` | update | Policy entropy |
| `eval_reward` | eval interval | Mean reward over evaluation episodes |

All hyperparameters (lr, gamma, clip_range, etc.) are saved as the run config.

## Stage 2 — Collect Expert Demonstrations

Use the trained PPO agent to collect trajectories:

```python
from il_based_rl.ppo_agent import PPOAgent
from il_based_rl.collect import collect_demos

# Load the trained expert
expert = PPOAgent.load("checkpoints/ppo_expert.pt", hidden_dims=[64, 64])

# Roll out the expert to gather demonstrations
dataset = collect_demos("Pendulum-v1", agent=expert, num_episodes=50, seed=0)
print(f"Collected {len(dataset)} transitions")

# Save to disk
dataset.save("demos/ppo_expert_demos.npz")
```

The `collect_demos` function accepts any agent with a `predict(obs) -> action` method — both `PPOAgent` and `BCAgent` work out of the box.

## Stage 3 — Train a BC Student

Train a Behavioral Cloning agent on the expert demonstrations:

```bash
python -m il_based_rl.train --demo-path demos/ppo_expert_demos.npz \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --hidden-dims 256 256 \
    --save-path checkpoints/bc_student.pt
```

Then evaluate:

```bash
python -m il_based_rl.evaluate \
    --env-id Pendulum-v1 \
    --checkpoint checkpoints/bc_student.pt \
    --hidden-dims 256 256 \
    --num-episodes 20
```

## Full Pipeline Script

The script below runs all three stages end-to-end:

```bash
python scripts/run_pipeline.py \
    --env-id Pendulum-v1 \
    --ppo-timesteps 200000 \
    --num-demo-episodes 50 \
    --bc-epochs 100
```

Or from Python:

```python
from scripts.run_pipeline import run_pipeline

run_pipeline(
    env_id="Pendulum-v1",
    ppo_timesteps=200_000,
    num_demo_episodes=50,
    bc_epochs=100,
    seed=42,
)
```

See [`scripts/run_pipeline.py`](scripts/run_pipeline.py) for the full source.

## Project Structure

```
src/il_based_rl/
├── policy.py                # MLPPolicy — MLP for BC (continuous + discrete)
├── dataset.py               # DemoDataset — (obs, action) pairs, save/load .npz
├── agent.py                 # BCAgent — supervised training + inference
├── collect.py               # collect_demos() — rollout any Predictable agent
├── train.py                 # CLI: python -m il_based_rl.train
├── evaluate.py              # CLI: python -m il_based_rl.evaluate
├── actor_critic_policy.py   # ActorCriticPolicy — actor/critic MLPs + log_std
├── buffer.py                # RolloutBuffer — GAE computation + minibatches
├── ppo_agent.py             # PPOAgent — PPO training loop
└── train_ppo.py             # CLI: python -m il_based_rl.train_ppo

scripts/
└── run_pipeline.py          # End-to-end: PPO → demos → BC

tests/
├── test_policy.py
├── test_dataset.py
├── test_agent.py
├── test_actor_critic_policy.py
├── test_buffer.py
└── test_ppo_agent.py
```

## Testing

```bash
pytest tests/
```
