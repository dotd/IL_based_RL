"""Tests for MLPPolicy."""

import torch

from il_based_rl.policy import MLPPolicy


class TestMLPPolicyContinuous:
    def test_forward_shape(self):
        policy = MLPPolicy(obs_dim=10, action_dim=3, hidden_dims=[32, 32], continuous=True)
        obs = torch.randn(4, 10)
        out = policy(obs)
        assert out.shape == (4, 3)

    def test_predict_shape(self):
        policy = MLPPolicy(obs_dim=10, action_dim=3, hidden_dims=[32, 32], continuous=True)
        obs = torch.randn(1, 10)
        out = policy.predict(obs)
        assert out.shape == (1, 3)

    def test_gradient_flow(self):
        policy = MLPPolicy(obs_dim=4, action_dim=2, hidden_dims=[16], continuous=True)
        obs = torch.randn(2, 4)
        out = policy(obs)
        loss = out.sum()
        loss.backward()
        for param in policy.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0


class TestMLPPolicyDiscrete:
    def test_forward_shape(self):
        policy = MLPPolicy(obs_dim=8, action_dim=5, hidden_dims=[32], continuous=False)
        obs = torch.randn(3, 8)
        out = policy(obs)
        assert out.shape == (3, 5)

    def test_predict_returns_scalar_action(self):
        policy = MLPPolicy(obs_dim=8, action_dim=5, hidden_dims=[32], continuous=False)
        obs = torch.randn(1, 8)
        out = policy.predict(obs)
        # discrete predict returns argmax → scalar per batch item
        assert out.shape == (1,)
