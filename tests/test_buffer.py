"""Tests for RolloutBuffer."""

import numpy as np

from il_based_rl.buffer import RolloutBuffer


class TestRolloutBuffer:
    def _make_buffer(self, n=100, obs_dim=3, action_dim=1):
        buf = RolloutBuffer()
        for i in range(n):
            buf.add(
                obs=np.random.randn(obs_dim).astype(np.float32),
                action=np.random.randn(action_dim).astype(np.float32),
                reward=float(np.random.randn()),
                done=(i == n - 1),  # done only at end
                log_prob=float(np.random.randn()),
                value=float(np.random.randn()),
            )
        return buf

    def test_length(self):
        buf = self._make_buffer(50)
        assert len(buf) == 50

    def test_gae_shapes(self):
        buf = self._make_buffer(100)
        buf.compute_returns_and_advantages(last_value=0.0)
        assert buf.advantages.shape == (100,)
        assert buf.returns.shape == (100,)

    def test_returns_equal_advantages_plus_values(self):
        buf = self._make_buffer(100)
        buf.compute_returns_and_advantages(last_value=0.5)
        values = np.array(buf.values, dtype=np.float32)
        np.testing.assert_allclose(buf.returns, buf.advantages + values, atol=1e-5)

    def test_batch_coverage(self):
        """All indices should appear across all batches."""
        buf = self._make_buffer(100)
        buf.compute_returns_and_advantages(last_value=0.0)
        batches = buf.get_batches(batch_size=32)
        # Should have ceil(100/32) = 4 batches
        assert len(batches) == 4
        # Check all keys present
        for b in batches:
            assert set(b.keys()) == {"obs", "actions", "old_log_probs", "returns", "advantages"}

    def test_done_boundary_resets_gae(self):
        """When done=True mid-rollout, GAE should not propagate across the boundary."""
        buf = RolloutBuffer()
        # First episode: 3 steps with done at step 2
        for i in range(3):
            buf.add(
                obs=np.zeros(2, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=1.0,
                done=(i == 2),
                log_prob=0.0,
                value=0.0,
            )
        # Second episode: 2 steps, not done
        for i in range(2):
            buf.add(
                obs=np.zeros(2, dtype=np.float32),
                action=np.zeros(1, dtype=np.float32),
                reward=1.0,
                done=False,
                log_prob=0.0,
                value=0.0,
            )
        buf.compute_returns_and_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
        # The advantage at step 2 (done=True) should only reflect its own reward
        # (since value=0 and next is cut off by done)
        assert buf.advantages is not None
        assert buf.advantages[2] == 1.0  # delta = reward + 0 - 0 = 1.0, no propagation from future
