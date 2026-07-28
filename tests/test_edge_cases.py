"""
BAGIAN 19.2 — Edge Case Tests
Test kasus ekstrem yang rawan bug halus tanpa error eksplisit.
"""

import numpy as np
import pytest
import torch
from src.rl.environment import NeuroStockEnv
from src.forecasting.model import NeuroStockForecaster


@pytest.fixture
def env():
    return NeuroStockEnv(n_stores=3, forecast_dim=7, seed=42)


def test_edge_case_zero_inventory_zero_demand(env):
    """Reward harus tetap finite meski demand dan inventori nol."""
    env.reset()
    for agent in env.agents:
        if agent != "distribution_center":
            env.inventory[agent] = 0.0
    reward = env._compute_reward("store_0", inventory_after=0,
                                  demand=0, order_qty=0, realloc=0)
    assert np.isfinite(reward), "Reward harus finite meski demand dan inventori nol"


def test_edge_case_zero_demand_nonzero_inventory(env):
    """Reward hanya komponen holding saat demand nol."""
    env.reset()
    reward = env._compute_reward("store_0", inventory_after=100,
                                  demand=0, order_qty=0, realloc=0)
    assert np.isfinite(reward)
    assert reward < 0  # holding cost tetap ada


def test_edge_case_very_large_order(env):
    """Order sangat besar tidak boleh menyebabkan infinite/NaN."""
    env.reset()
    actions = {agent: np.array([1.0, 0.0]) for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for agent, r in rewards.items():
        assert np.isfinite(r), f"Reward {agent} tidak finite dengan order maksimal"


def test_edge_case_negative_realloc(env):
    """Realokasi negatif (kirim stok keluar) tidak boleh crash."""
    env.reset()
    actions = {agent: np.array([0.0, -1.0]) for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    for agent, r in rewards.items():
        assert np.isfinite(r)


def test_edge_case_very_short_history():
    """Model tidak boleh crash jika input sequence lebih pendek dari biasanya."""
    model = NeuroStockForecaster(
        n_sales_features=10, n_sensor_features=3,
        d_model=16, n_heads=2, n_layers=1, horizon=5
    )
    model.eval()
    short_seq = torch.randn(1, 5, 10)   # hanya 5 timestep
    sensor_seq = torch.randn(1, 5, 3)
    with torch.no_grad():
        mu, log_var, _ = model(short_seq, sensor_seq)
    assert mu.shape == (1, 5)
    assert torch.all(torch.isfinite(mu))


def test_edge_case_single_sample_batch():
    """Batch size 1 tidak boleh crash (BatchNorm issue)."""
    model = NeuroStockForecaster(
        n_sales_features=10, n_sensor_features=3,
        d_model=16, n_heads=2, n_layers=1, horizon=5
    )
    model.eval()
    with torch.no_grad():
        mu, log_var, _ = model(
            torch.randn(1, 20, 10),
            torch.randn(1, 20, 3)
        )
    assert mu.shape == (1, 5)


def test_edge_case_episode_truncation(env):
    """Episode harus truncate dengan benar di timestep 365."""
    env.reset()
    env.timestep = 365  # set langsung ke 365
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    _, _, terms, truncs, _ = env.step(actions)
    assert all(truncs.values()), "Semua agent harus truncated di timestep 365"