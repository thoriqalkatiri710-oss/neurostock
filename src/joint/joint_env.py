import torch
import numpy as np
from src.rl.environment import NeuroStockEnv


class JointNeuroStockEnv(NeuroStockEnv):
    """
    Wrapper environment yang menghubungkan forecaster ke RL state.

    Poin kunci: sigma (uncertainty forecast) langsung masuk ke observasi agent.
    Agent belajar secara natural: sigma besar → jaga margin inventori lebih besar.
    Tidak perlu di-hardcode sebagai aturan terpisah.
    """

    def __init__(self, forecaster, forecaster_input_builder, **kwargs):
        super().__init__(**kwargs)
        self.forecaster = forecaster
        self.forecaster.eval()
        self.build_forecaster_input = forecaster_input_builder
        self._cached_forecasts = {}

    def _get_promo_flag(self, agent: str) -> float:
        return self.promo_flag.get(agent, 0.0)

    def _get_days_to_holiday(self, agent: str) -> float:
        return self.days_to_holiday.get(agent, 15.0)

    def _get_obs(self, agent: str) -> np.ndarray:
        if agent == "distribution_center":
            # DC observasi: inventory semua toko + dc_inventory
            store_inventories = np.array([
                self.inventory.get(a, 0) for a in self.possible_agents
                if a != "distribution_center"
            ], dtype=np.float32)
            dc_obs = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
            dc_obs[:len(store_inventories)] = store_inventories / self.max_inventory
            dc_obs[len(store_inventories)] = self.dc_inventory / (self.max_inventory * self.n_stores)
            return dc_obs

        # Dapatkan forecast dari transformer
        sales_seq, sensor_seq = self.build_forecaster_input(agent, self.timestep)
        with torch.no_grad():
            mu, log_var, _ = self.forecaster(sales_seq, sensor_seq)
            sigma = torch.exp(0.5 * log_var)

        # Cache forecast untuk debugging
        self._cached_forecasts[agent] = {
            "mu": mu.squeeze().numpy(),
            "sigma": sigma.squeeze().numpy()
        }

        obs = np.concatenate([
            [self.inventory[agent] / self.max_inventory],
            mu.squeeze().numpy(),
            sigma.squeeze().numpy(),
            [self._get_promo_flag(agent)],
            [self._get_days_to_holiday(agent) / 30.0],
            self._get_dc_signal(),
        ]).astype(np.float32)
        return obs