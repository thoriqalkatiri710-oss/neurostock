import torch
import torch.nn.functional as F
from torch.optim import Adam

from .networks import ActorCritic


class MAPPOTrainer:
    """
    Multi-Agent PPO Trainer.
    - compute_gae: Persamaan 1.5.2 (Generalized Advantage Estimation)
    - ratio: Persamaan 1.7.2 (probability ratio)
    - policy_loss: Persamaan 1.7.3 (clipped surrogate objective)
    - total_loss: L^total dengan entropy bonus
    """

    def __init__(self, agents: list, obs_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01):
        self.networks = {name: ActorCritic(obs_dim, action_dim) for name in agents}
        self.optimizers = {name: Adam(net.parameters(), lr=lr)
                          for name, net in self.networks.items()}
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards: list, values: list, dones: list) -> tuple:
        """Generalized Advantage Estimation (Persamaan 1.5.2)."""
        advantages = []
        gae = 0
        values_ext = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def update(self, agent_name: str, batch: dict, n_epochs: int = 10,
               batch_size: int = 64) -> dict:
        net = self.networks[agent_name]
        optimizer = self.optimizers[agent_name]

        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Normalisasi advantage — menstabilkan training PPO
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(n_epochs):
            mean, std, values = net(obs)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            # Persamaan 1.7.2: probability ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Persamaan 1.7.3: clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values.squeeze(-1), returns)
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }

    def save(self, path: str = "checkpoints/mappo_best.pt"):
        torch.save({name: net.state_dict() for name, net in self.networks.items()}, path)
        print(f"MAPPO saved: {path}")

    def load(self, path: str = "checkpoints/mappo_best.pt"):
        state_dicts = torch.load(path)
        for name, net in self.networks.items():
            net.load_state_dict(state_dicts[name])
        print(f"MAPPO loaded: {path}")