"""Tests for ActorCriticPolicy."""

import torch

from il_based_rl.actor_critic_policy import ActorCriticPolicy


class TestActorCriticPolicy:
    def _make_policy(self):
        return ActorCriticPolicy(obs_dim=3, action_dim=1, hidden_dims=[32, 32])

    def test_forward_shapes(self):
        policy = self._make_policy()
        obs = torch.randn(4, 3)
        action_mean, value = policy(obs)
        assert action_mean.shape == (4, 1)
        assert value.shape == (4,)

    def test_get_distribution_sampling(self):
        policy = self._make_policy()
        obs = torch.randn(4, 3)
        dist, value = policy.get_distribution(obs)
        sample = dist.sample()
        assert sample.shape == (4, 1)
        assert value.shape == (4,)

    def test_evaluate_actions_shapes(self):
        policy = self._make_policy()
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 1)
        log_prob, value, entropy = policy.evaluate_actions(obs, actions)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)
        assert entropy.shape == (4,)

    def test_predict_shape(self):
        policy = self._make_policy()
        obs = torch.randn(1, 3)
        action = policy.predict(obs)
        assert action.shape == (1, 1)

    def test_gradient_flow(self):
        policy = self._make_policy()
        obs = torch.randn(4, 3)
        actions = torch.randn(4, 1)
        log_prob, value, entropy = policy.evaluate_actions(obs, actions)
        loss = -log_prob.mean() + value.mean() - entropy.mean()
        loss.backward()
        for name, param in policy.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_log_std_is_learnable(self):
        policy = self._make_policy()
        assert policy.log_std.requires_grad is True
        # log_std should be in parameters
        param_names = [name for name, _ in policy.named_parameters()]
        assert "log_std" in param_names
