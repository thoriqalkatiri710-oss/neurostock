import torch
import numpy as np
from src.forecasting.model import NeuroStockForecaster
from src.rl.environment import NeuroStockEnv
from src.joint.joint_env import JointNeuroStockEnv


def test_full_pipeline_runs_without_error():
    forecaster = NeuroStockForecaster(
        n_sales_features=8, n_sensor_features=3,
        d_model=8, n_heads=2, n_layers=1, horizon=5
    )
    forecaster.eval()

    def dummy_input_builder(agent, timestep):
        return torch.randn(1, 20, 8), torch.randn(1, 20, 3)

    env = JointNeuroStockEnv(
        forecaster=forecaster,
        forecaster_input_builder=dummy_input_builder,
        n_stores=2, forecast_dim=5
    )
    obs, infos = env.reset()

    for _ in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

    assert all(isinstance(r, float) for r in rewards.values())
    print("Pipeline end-to-end berjalan tanpa error.")