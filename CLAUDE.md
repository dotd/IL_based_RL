# IL-based RL (Behavioral Cloning)

## Build & Install
```bash
pip install -e ".[dev]"
```

## Run Tests
```bash
pytest tests/
```

## Architecture

### Behavioral Cloning
- **src/il_based_rl/policy.py** — `MLPPolicy`: PyTorch MLP supporting continuous and discrete action spaces
- **src/il_based_rl/dataset.py** — `DemoDataset`: stores (obs, action) pairs, saves/loads as `.npz`
- **src/il_based_rl/agent.py** — `BCAgent`: wraps policy + optimizer for supervised training and inference
- **src/il_based_rl/collect.py** — `collect_demos()`: roll out any `Predictable` agent in a Gymnasium env to gather demonstrations
- **src/il_based_rl/train.py** — CLI: `python -m il_based_rl.train --demo-path <path> [options]`
- **src/il_based_rl/evaluate.py** — CLI: `python -m il_based_rl.evaluate --checkpoint <path> [options]`

### PPO (Proximal Policy Optimization)
- **src/il_based_rl/actor_critic_policy.py** — `ActorCriticPolicy`: separate actor/critic MLPs with learnable `log_std`
- **src/il_based_rl/buffer.py** — `RolloutBuffer`: stores rollout data, computes GAE advantages, yields minibatches
- **src/il_based_rl/ppo_agent.py** — `PPOAgent`: PPO training loop with clipped surrogate loss, collect/update/train/predict/save/load
- **src/il_based_rl/train_ppo.py** — CLI: `python -m il_based_rl.train_ppo --env-id Pendulum-v1 [options]`

### Pipeline: RL → Expert Demos → BC
1. Train PPO expert: `python -m il_based_rl.train_ppo --env-id Pendulum-v1`
2. Collect demos: use `collect_demos(env_id, ppo_agent)` (PPOAgent satisfies `Predictable` protocol)
3. Train BC on expert demos: `python -m il_based_rl.train --demo-path demos.npz`

## Code Style
- Type hints on all public functions
- `from __future__ import annotations` in all modules
- numpy for data, PyTorch for models
