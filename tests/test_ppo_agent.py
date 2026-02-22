"""Tests for PPOAgent."""

import numpy as np
import gymnasium as gym

from il_based_rl.actor_critic_policy import ActorCriticPolicy
from il_based_rl.ppo_agent import PPOAgent
from il_based_rl.collect import collect_demos


class TestPPOAgent:
    def _make_agent(self, obs_dim=3, action_dim=1):
        policy = ActorCriticPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=[32])
        return PPOAgent(policy, lr=3e-4, n_steps=64, n_epochs=2, batch_size=32)

    def test_predict_shape(self):
        agent = self._make_agent()
        obs = np.random.randn(3).astype(np.float32)
        action = agent.predict(obs)
        assert action.shape == (1,)

    def test_collect_rollouts(self):
        agent = self._make_agent()
        env = gym.make("Pendulum-v1")
        buffer, last_obs = agent.collect_rollouts(env)
        env.close()
        assert len(buffer) == 64
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert last_obs.shape == (3,)

    def test_single_update_step(self):
        agent = self._make_agent()
        env = gym.make("Pendulum-v1")
        buffer, _ = agent.collect_rollouts(env)
        env.close()
        info = agent.update(buffer)
        assert "policy_loss" in info
        assert "value_loss" in info
        assert "entropy" in info
        assert isinstance(info["policy_loss"], float)

    def test_save_load_roundtrip(self, tmp_path):
        agent = self._make_agent()
        obs = np.random.randn(3).astype(np.float32)
        orig_action = agent.predict(obs)

        path = tmp_path / "ppo_agent.pt"
        agent.save(path)
        loaded = PPOAgent.load(path, hidden_dims=[32])

        loaded_action = loaded.predict(obs)
        np.testing.assert_array_almost_equal(orig_action, loaded_action)

    def test_works_with_collect_demos(self):
        """PPOAgent satisfies the Predictable protocol and works with collect_demos."""
        agent = self._make_agent()
        dataset = collect_demos("Pendulum-v1", agent=agent, num_episodes=2)
        assert len(dataset) > 0
        assert dataset.observations.shape[1] == 3
        assert dataset.actions.shape[1] == 1
