"""Tests for BCAgent."""

import numpy as np

from il_based_rl.agent import BCAgent
from il_based_rl.dataset import DemoDataset
from il_based_rl.policy import MLPPolicy


class TestBCAgent:
    def _make_agent_and_data(self):
        obs_dim, action_dim = 4, 2
        policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[32], continuous=True)
        agent = BCAgent(policy, lr=1e-3)
        obs = np.random.randn(200, obs_dim).astype(np.float32)
        act = np.random.randn(200, action_dim).astype(np.float32)
        dataset = DemoDataset(obs, act)
        return agent, dataset

    def test_train_reduces_loss(self):
        agent, dataset = self._make_agent_and_data()
        losses = agent.train(dataset, epochs=20, batch_size=32)
        assert losses[-1] < losses[0], "Loss should decrease over training"

    def test_predict_shape(self):
        agent, _ = self._make_agent_and_data()
        obs = np.random.randn(4).astype(np.float32)
        action = agent.predict(obs)
        assert action.shape == (2,)

    def test_save_load_roundtrip(self, tmp_path):
        agent, dataset = self._make_agent_and_data()
        agent.train(dataset, epochs=5, batch_size=32)

        path = tmp_path / "agent.pt"
        agent.save(path)
        loaded = BCAgent.load(path, hidden_dims=[32])

        obs = np.random.randn(4).astype(np.float32)
        orig_action = agent.predict(obs)
        loaded_action = loaded.predict(obs)
        np.testing.assert_array_almost_equal(orig_action, loaded_action)
