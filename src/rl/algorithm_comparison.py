import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch


# ── Single-Agent Wrapper untuk SB3 ───────────────────────────────────────────

class SingleStoreEnv(gym.Env):
    """
    Wrapper single-agent dari NeuroStockEnv untuk kompatibilitas SB3.
    Catatan: SAC/TD3 didesain untuk single-agent — perbandingan ini
    dilakukan di level 1 toko sebagai studi tambahan, bukan pengganti
    hasil utama multi-agent PPO.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_inventory: int = 1000, forecast_dim: int = 14,
                 lead_time: int = 3, seed: int = 42):
        super().__init__()
        self.max_inventory = max_inventory
        self.forecast_dim = forecast_dim
        self.lead_time = lead_time
        self.rng = np.random.default_rng(seed)
        self.timestep = 0

        obs_dim = 1 + forecast_dim + forecast_dim + 1 + 1 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.inventory = float(self.rng.uniform(50, 200))
        self.pending_orders = []
        self.timestep = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        mu = self.rng.uniform(10, 50, size=self.forecast_dim).astype(np.float32)
        sigma = self.rng.uniform(1, 10, size=self.forecast_dim).astype(np.float32)
        return np.concatenate([
            [self.inventory / self.max_inventory],
            mu / self.max_inventory,
            sigma / self.max_inventory,
            [0.0],   # promo_flag
            [0.5],   # days_to_holiday
            [0.15, 0.04],  # dc_signal
        ]).astype(np.float32)

    def step(self, action):
        order_qty = (action[0] + 1) / 2 * self.max_inventory
        realloc = action[1] * (self.max_inventory * 0.2)

        # Process pending orders
        arriving = sum(qty for qty, arr in self.pending_orders if arr <= self.timestep)
        self.pending_orders = [(qty, arr) for qty, arr in self.pending_orders
                               if arr > self.timestep]
        self.inventory += arriving

        # Demand
        demand = float(self.rng.uniform(10, 80))
        inventory_after = self.inventory - demand
        self.inventory = max(inventory_after, -demand)

        # Order
        if order_qty > 0:
            self.pending_orders.append((order_qty, self.timestep + self.lead_time))

        # Reward
        r_holding = -0.5 * max(inventory_after, 0)
        r_stockout = -2.0 * max(-inventory_after, 0)
        r_service = min(demand, max(inventory_after + demand, 0)) / (demand + 1e-6)
        reward = 0.3 * r_holding + 0.4 * r_stockout + 0.2 * r_service * 10

        self.timestep += 1
        terminated = False
        truncated = self.timestep >= 365

        return self._get_obs(), float(reward), terminated, truncated, {}


# ── 10.3.1 Algorithm Comparison ──────────────────────────────────────────────

def compare_rl_algorithms(env_fn, algorithms: dict,
                           total_timesteps: int = 10_000) -> dict:
    """
    Bandingkan PPO vs SAC vs TD3 di single-agent environment.
    Jalankan sebagai studi tambahan, bukan pengganti hasil multi-agent utama.
    """
    results = {}

    for name, AlgoClass in algorithms.items():
        print(f"\n── Training {name} ({total_timesteps} timesteps) ──")
        env = env_fn()
        model = AlgoClass("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=total_timesteps)

        eval_rewards = []
        obs, _ = env.reset()
        for _ in range(20):
            episode_reward = 0.0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            eval_rewards.append(episode_reward)
            obs, _ = env.reset()

        results[name] = {
            "mean_reward": round(float(np.mean(eval_rewards)), 2),
            "std_reward": round(float(np.std(eval_rewards)), 2),
        }
        print(f"  {name}: mean={results[name]['mean_reward']:.2f} "
              f"± {results[name]['std_reward']:.2f}")

    return results


if __name__ == "__main__":
    try:
        from stable_baselines3 import SAC, TD3, PPO

        print("── RL Algorithm Comparison (Single-Agent) ──")
        print("Catatan: perbandingan di 1 toko — studi tambahan vs multi-agent utama\n")

        algorithms = {
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
        }

        results = compare_rl_algorithms(
            env_fn=SingleStoreEnv,
            algorithms=algorithms,
            total_timesteps=5_000  # kecil untuk demo
        )

        print("\n── Summary ──")
        for name, r in results.items():
            print(f"  {name}: {r['mean_reward']:.2f} ± {r['std_reward']:.2f}")

        best = max(results, key=lambda k: results[k]["mean_reward"])
        print(f"\n✅ Best algorithm (demo): {best}")

    except ImportError:
        print("stable-baselines3 tidak terinstall. Jalankan: pip install stable-baselines3")