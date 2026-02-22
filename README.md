# IL-based RL: Behavioral Cloning

Minimal Behavioral Cloning framework built with PyTorch and Gymnasium.

## Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Collect demonstrations
```python
from il_based_rl.collect import collect_demos
dataset = collect_demos("HalfCheetah-v4", agent=None, num_episodes=20)
dataset.save("demos/halfcheetah.npz")
```

### 2. Train
```bash
python -m il_based_rl.train --demo-path demos/halfcheetah.npz --epochs 100
```

### 3. Evaluate
```bash
python -m il_based_rl.evaluate --checkpoint checkpoints/bc_agent.pt --num-episodes 10
```

## Testing
```bash
pytest tests/
```
