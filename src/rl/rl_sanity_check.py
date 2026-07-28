import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .environment import NeuroStockEnv
from .ppo_trainer import MAPPOTrainer

# ── 4.8.2 Logging ─────────────────────────────────────────────────────────────

def log_episode(writer: SummaryWriter, episode: int, rewards: dict, losses: dict):
    for agent, r in rewards.items():
        writer.add_scalar(f"reward/{agent}", r, episode)
    for agent, loss_dict in losses.items():
        for k, v in loss_dict.items():
            writer.add_scalar(f"{k}/{agent}", v, episode)


# ── 4.8.1 Sanity Check Bertahap ───────────────────────────────────────────────

def run_sanity_check(n_episodes: int = 50, max_steps: int = 30):
    writer = SummaryWriter(log_dir="logs/rl_training")
    env = NeuroStockEnv(n_stores=5)
    obs, _ = env.reset()

    obs_dim = env.observation_space("store_0").shape[0]
    action_dim = env.action_space("store_0").shape[0]
    trainer = MAPPOTrainer(agents=env.agents, obs_dim=obs_dim, action_dim=action_dim)

    print("── RL Sanity Check ──")
    print(f"obs_dim={obs_dim}, action_dim={action_dim}, agents={len(env.agents)}")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards = {agent: 0.0 for agent in env.agents}

        # Buffer per agent
        buffers = {agent: {"obs": [], "actions": [], "log_probs": [],
                           "rewards": [], "values": [], "dones": []}
                   for agent in env.agents}

        for step in range(max_steps):
            actions = {}
            log_probs = {}
            values = {}

            for agent in env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0)
                action, log_prob, value = trainer.networks[agent].get_action(obs_tensor)
                actions[agent] = action.squeeze(0).detach().numpy()
                log_probs[agent] = log_prob.item()
                values[agent] = value.item()

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent in env.agents:
                buffers[agent]["obs"].append(obs[agent])
                buffers[agent]["actions"].append(actions[agent])
                buffers[agent]["log_probs"].append(log_probs[agent])
                buffers[agent]["rewards"].append(rewards[agent])
                buffers[agent]["values"].append(values[agent])
                buffers[agent]["dones"].append(float(terminations[agent]))
                episode_rewards[agent] += rewards[agent]

            obs = next_obs
            if all(terminations.values()) or all(truncations.values()):
                break

        # Update PPO per agent
        losses = {}
        for agent in env.agents:
            buf = buffers[agent]
            advs, rets = trainer.compute_gae(buf["rewards"], buf["values"], buf["dones"])

            batch = {
                "obs":        torch.FloatTensor(np.array(buf["obs"])),
                "actions":    torch.FloatTensor(np.array(buf["actions"])),
                "log_probs":  torch.FloatTensor(buf["log_probs"]),
                "advantages": torch.FloatTensor(advs),
                "returns":    torch.FloatTensor(rets),
            }
            losses[agent] = trainer.update(agent, batch, n_epochs=4)

        log_episode(writer, episode, episode_rewards, losses)

        if episode % 10 == 0:
            avg_r = np.mean(list(episode_rewards.values()))
            print(f"Episode {episode:3d} | avg_reward={avg_r:.4f} | "
                  f"entropy={losses['store_0']['entropy']:.4f}")

    writer.close()
    print("\n✅ Sanity check selesai. Jalankan:")
    print("   tensorboard --logdir logs/rl_training")


if __name__ == "__main__":
    run_sanity_check()