import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class NeuroStockEnv(ParallelEnv):
    """
    Multi-agent inventory management environment.
    - Store agents: order quantity + reallocation request
    - Distribution center agent: alokasi stok ke toko
    - State: inventory + forecast (mu, sigma) + promo flag + days_to_holiday
    """

    metadata = {"name": "neurostock_v1"}

    def __init__(self, n_stores: int = 5, forecast_dim: int = 14,
                 max_inventory: int = 1000, lead_time: int = 3, seed: int = 42):
        super().__init__()
        self.n_stores = n_stores
        self.forecast_dim = forecast_dim
        self.max_inventory = max_inventory
        self.lead_time = lead_time
        self.rng = np.random.default_rng(seed)

        self.possible_agents = [f"store_{i}" for i in range(n_stores)] + ["distribution_center"]
        self.agents = self.possible_agents[:]

        # State per store: [inventory, forecast_mu (H), forecast_sigma (H), promo_flag, days_to_holiday]
        obs_dim = 1 + forecast_dim + forecast_dim + 1 + 1 + 2
        self._observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Action: [order_quantity (0-1), reallocation_request (-1 to 1)]
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ── 4.2.2 Reset ───────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.timestep = 0

        self.inventory = {
            agent: self.rng.uniform(50, 200)
            for agent in self.agents if agent != "distribution_center"
        }
        self.pending_orders = {
            agent: [] for agent in self.agents if agent != "distribution_center"
        }
        self.dc_inventory = self.max_inventory * self.n_stores * 0.5

        # Simulasi forecast awal (nanti diganti output transformer)
        self.forecast_mu = {
            agent: self.rng.uniform(10, 50, size=self.forecast_dim)
            for agent in self.agents if agent != "distribution_center"
        }
        self.forecast_sigma = {
            agent: self.rng.uniform(1, 10, size=self.forecast_dim)
            for agent in self.agents if agent != "distribution_center"
        }
        self.promo_flag = {agent: 0.0 for agent in self.agents}
        self.days_to_holiday = {agent: float(self.rng.integers(1, 30)) for agent in self.agents}

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def _get_obs(self, agent) -> np.ndarray:
        if agent == "distribution_center":
            store_inventories = np.array([
                self.inventory.get(a, 0) for a in self.possible_agents
                if a != "distribution_center"
            ], dtype=np.float32)
            dc_obs = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
            dc_obs[:len(store_inventories)] = store_inventories / self.max_inventory
            dc_obs[len(store_inventories)] = self.dc_inventory / (self.max_inventory * self.n_stores)
            return dc_obs

        inv = np.array([self.inventory[agent] / self.max_inventory], dtype=np.float32)
        mu = (self.forecast_mu[agent] / self.max_inventory).astype(np.float32)
        sigma = (self.forecast_sigma[agent] / self.max_inventory).astype(np.float32)
        promo = np.array([self.promo_flag[agent]], dtype=np.float32)
        holiday = np.array([self.days_to_holiday[agent] / 30.0], dtype=np.float32)
        dc_signal = self._get_dc_signal()  # sinyal koordinasi global
        return np.concatenate([inv, mu, sigma, promo, holiday, dc_signal])

    # ── 4.5.1 Step Function ───────────────────────────────────────────────────

    def step(self, actions: dict):
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}

        # Decode seluruh aksi
        decoded = {
            agent: self._decode_action(agent, actions[agent])
            for agent in self.agents if agent != "distribution_center"
        }

        # Validasi konservasi realokasi (Persamaan 1.4.2)
        total_realloc = sum(d["realloc"] for d in decoded.values())
        if abs(total_realloc) > 1e-3:
            correction = total_realloc / len(decoded)
            for d in decoded.values():
                d["realloc"] -= correction

        # Update setiap store agent
        for agent in self.agents:
            if agent == "distribution_center":
                continue

            demand = self._simulate_demand(agent)
            arriving_order = self._process_pending_orders(agent)

            inventory_after = (
                self.inventory[agent] - demand + arriving_order + decoded[agent]["realloc"]
            )
            # Backlog dibatasi, tidak negatif tak terbatas
            self.inventory[agent] = max(inventory_after, -demand)

            # Catat order baru dengan lead time
            self.pending_orders[agent].append({
                "qty": decoded[agent]["order_qty"],
                "arrival_step": self.timestep + self.lead_time
            })

            rewards[agent] = self._compute_reward(
                agent, inventory_after, demand,
                decoded[agent]["order_qty"], decoded[agent]["realloc"]
            )
            observations[agent] = self._get_obs(agent)
            terminations[agent] = False
            truncations[agent] = self.timestep >= 365
            infos[agent] = {"demand": demand, "inventory": self.inventory[agent]}

        # Distribution center reward: rata-rata store + biaya koordinasi
        rewards["distribution_center"] = float(np.mean(list(rewards.values())))
        observations["distribution_center"] = self._get_obs("distribution_center")
        terminations["distribution_center"] = False
        truncations["distribution_center"] = self.timestep >= 365
        infos["distribution_center"] = {}

        self.timestep += 1
        return observations, rewards, terminations, truncations, infos

    # ── Pending Orders (Lead Time) ────────────────────────────────────────────

    def _process_pending_orders(self, agent: str) -> float:
        """Order yang dibuat hari ini baru tiba L hari kemudian."""
        arriving = 0.0
        remaining = []
        for order in self.pending_orders[agent]:
            if order["arrival_step"] <= self.timestep:
                arriving += order["qty"]
            else:
                remaining.append(order)
        self.pending_orders[agent] = remaining
        return arriving

    def _simulate_demand(self, agent: str) -> float:
        """Simulasi demand harian dengan pola musiman + noise."""
        base_demand = self.rng.uniform(10, 80)
        weekend_boost = 1.3 if self.timestep % 7 >= 5 else 1.0
        promo_boost = 1.5 if self.promo_flag[agent] else 1.0
        noise = self.rng.normal(1.0, 0.1)
        return max(0.0, base_demand * weekend_boost * promo_boost * noise)    
    
    # ── 4.6.1 DC Coordination Signal (CTDE) ──────────────────────────────────

    def _get_dc_signal(self) -> np.ndarray:
        """
        Sinyal koordinasi dari distribution center ke store agents.
        Implementasi CTDE (Persamaan 1.6.2):
        - Training: bisa diperluas ke informasi global penuh
        - Deployment: dipangkas ke ringkasan statistik agregat saja
          (toko tidak butuh data mentah toko lain)
        """
        inventories = np.array(list(self.inventory.values()))
        relative_levels = inventories / (self.max_inventory + 1e-6)
        return np.array([relative_levels.mean(), relative_levels.std()], dtype=np.float32)
    # ── 4.3.1 Action Decoding ─────────────────────────────────────────────────

    def _decode_action(self, agent: str, raw_action: np.ndarray) -> dict:
        """
        Decode aksi mentah [-1,1] menjadi unit fisik.
        - order_qty: map [-1,1] -> [0, max_inventory]
        - realloc: dibatasi maksimum 20% max_inventory
        """
        order_qty = (raw_action[0] + 1) / 2 * self.max_inventory
        realloc_request = raw_action[1] * (self.max_inventory * 0.2)
        return {
            "order_qty": max(0.0, float(order_qty)),
            "realloc": float(realloc_request)
        }

    def _enforce_realloc_conservation(self, realloc_requests: dict) -> dict:
        """
        Constraint konservasi realokasi (Persamaan 1.4.2):
        total realokasi seluruh store harus = 0.
        Kelebihan/kekurangan dikoreksi secara proporsional.
        Ditangani di level environment (butuh koordinasi global).
        """
        store_agents = [a for a in self.agents if a != "distribution_center"]
        total = sum(realloc_requests[a] for a in store_agents)

        if abs(total) < 1e-6:
            return realloc_requests

        # Koreksi proporsional agar jumlah = 0
        correction = total / len(store_agents)
        corrected = {
            a: realloc_requests[a] - correction
            for a in store_agents
        }
        corrected["distribution_center"] = realloc_requests.get("distribution_center", 0.0)
        return corrected

    # ── 4.4.1 Reward Function ─────────────────────────────────────────────────

    def _compute_reward(self, agent: str, inventory_after: float,
                        demand: float, order_qty: float, realloc: float,
                        holding_cost: float = 0.5, stockout_penalty: float = 2.0,
                        transport_cost: float = 0.1) -> float:
        """
        Multi-objective reward (Persamaan 1.4):
        - R_holding:   penalti menyimpan stok berlebih
        - R_stockout:  penalti kehabisan stok (bobot tertinggi)
        - R_service:   reward melayani demand (dinormalisasi per demand)
        - R_transport: penalti biaya realokasi

        Bobot w adalah hyperparameter bisnis — tuning sesuai prioritas.
        """
        # R_holding
        r_holding = -holding_cost * max(inventory_after, 0)

        # R_stockout
        r_stockout = -stockout_penalty * max(-inventory_after, 0)

        # R_service — dinormalisasi agar tidak bias ke toko bervolume besar
        r_service = min(demand, max(inventory_after + demand, 0)) / (demand + 1e-6)

        # R_transport
        r_transport = -transport_cost * abs(realloc)

        w = {"holding": 0.3, "stockout": 0.4, "service": 0.2, "transport": 0.1}
        total_reward = (
            w["holding"] * r_holding +
            w["stockout"] * r_stockout +
            w["service"] * r_service * 10 +  # scaling agar magnitude sebanding
            w["transport"] * r_transport
        )
        return total_reward