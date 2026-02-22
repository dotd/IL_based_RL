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
- **src/il_based_rl/policy.py** — `MLPPolicy`: PyTorch MLP supporting continuous and discrete action spaces
- **src/il_based_rl/dataset.py** — `DemoDataset`: stores (obs, action) pairs, saves/loads as `.npz`
- **src/il_based_rl/agent.py** — `BCAgent`: wraps policy + optimizer for supervised training and inference
- **src/il_based_rl/collect.py** — `collect_demos()`: roll out a policy in a Gymnasium env to gather demonstrations
- **src/il_based_rl/train.py** — CLI: `python -m il_based_rl.train --demo-path <path> [options]`
- **src/il_based_rl/evaluate.py** — CLI: `python -m il_based_rl.evaluate --checkpoint <path> [options]`

## Code Style
- Type hints on all public functions
- `from __future__ import annotations` in all modules
- numpy for data, PyTorch for models
