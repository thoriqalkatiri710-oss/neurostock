import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic network untuk PPO.
    - Shared backbone: ekstrak fitur dari observasi
    - Actor head: output mean + log_std untuk distribusi Gaussian
    - Critic head: estimasi state value V(s)
    - tanh pada actor output memastikan aksi dalam [-1, 1]
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(self.actor_log_std).clamp(1e-3, 1.0)
        value = self.critic(features)
        return mean, std, value

    def get_action(self, obs: torch.Tensor):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1, 1), log_prob, value