"""Tests for DemoDataset."""

import numpy as np
import torch

from il_based_rl.dataset import DemoDataset


class TestDemoDataset:
    def test_length(self):
        obs = np.random.randn(100, 4).astype(np.float32)
        act = np.random.randn(100, 2).astype(np.float32)
        ds = DemoDataset(obs, act)
        assert len(ds) == 100

    def test_getitem_returns_tensors(self):
        obs = np.random.randn(10, 4).astype(np.float32)
        act = np.random.randn(10, 2).astype(np.float32)
        ds = DemoDataset(obs, act)
        o, a = ds[0]
        assert isinstance(o, torch.Tensor)
        assert isinstance(a, torch.Tensor)
        assert o.shape == (4,)
        assert a.shape == (2,)

    def test_save_load_roundtrip(self, tmp_path):
        obs = np.random.randn(50, 6).astype(np.float32)
        act = np.random.randn(50, 3).astype(np.float32)
        ds = DemoDataset(obs, act)

        path = tmp_path / "demos.npz"
        ds.save(path)
        loaded = DemoDataset.load(path)

        assert len(loaded) == 50
        np.testing.assert_array_almost_equal(loaded.observations, obs)
        np.testing.assert_array_almost_equal(loaded.actions, act)

    def test_from_trajectories(self):
        obs_list = [np.random.randn(10, 4) for _ in range(3)]
        act_list = [np.random.randn(10, 2) for _ in range(3)]
        ds = DemoDataset.from_trajectories(obs_list, act_list)
        assert len(ds) == 30
