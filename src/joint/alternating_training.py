import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── 5.5.1 Joint Training Monitor ─────────────────────────────────────────────

def log_joint_training_step(writer: SummaryWriter, round_idx: int,
                             forecast_loss: float, rl_reward: float,
                             alpha: float):
    """
    Log metrik joint training ke TensorBoard.
    Tanda sehat:
    - forecast_loss menurun bertahap
    - rl_reward meningkat bertahap
    - keduanya tidak saling mengorbankan secara drastis
    """
    writer.add_scalar("joint/forecast_loss", forecast_loss, round_idx)
    writer.add_scalar("joint/rl_reward", rl_reward, round_idx)
    writer.add_scalar("joint/alpha", alpha, round_idx)

# ── Helper: Run RL Episodes ───────────────────────────────────────────────────

def run_rl_episodes(rl_trainer, joint_env, n_episodes: int = 100,
                    max_steps: int = 30) -> list:
    """Jalankan n_episodes dan kembalikan total reward per episode."""
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = joint_env.reset()
        total_rewards = {agent: 0.0 for agent in joint_env.agents}
        buffers = {agent: {"obs": [], "actions": [], "log_probs": [],
                           "rewards": [], "values": [], "dones": []}
                   for agent in joint_env.agents}

        for step in range(max_steps):
            actions = {}
            log_probs = {}
            values = {}

            for agent in joint_env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0)
                action, log_prob, value = rl_trainer.networks[agent].get_action(obs_tensor)
                actions[agent] = action.squeeze(0).detach().numpy()
                log_probs[agent] = log_prob.item()
                values[agent] = value.item()

            next_obs, rewards, terminations, truncations, infos = joint_env.step(actions)

            for agent in joint_env.agents:
                buffers[agent]["obs"].append(obs[agent])
                buffers[agent]["actions"].append(actions[agent])
                buffers[agent]["log_probs"].append(log_probs[agent])
                buffers[agent]["rewards"].append(rewards[agent])
                buffers[agent]["values"].append(values[agent])
                buffers[agent]["dones"].append(float(terminations[agent]))
                total_rewards[agent] += rewards[agent]

            obs = next_obs
            if all(terminations.values()) or all(truncations.values()):
                break

        # Update PPO
        for agent in joint_env.agents:
            buf = buffers[agent]
            advs, rets = rl_trainer.compute_gae(buf["rewards"], buf["values"], buf["dones"])
            batch = {
                "obs":        torch.FloatTensor(np.array(buf["obs"])),
                "actions":    torch.FloatTensor(np.array(buf["actions"])),
                "log_probs":  torch.FloatTensor(buf["log_probs"]),
                "advantages": torch.FloatTensor(advs),
                "returns":    torch.FloatTensor(rets),
            }
            rl_trainer.update(agent, batch, n_epochs=4)

        episode_rewards.append(np.mean(list(total_rewards.values())))

    return episode_rewards


# ── 5.2.1 Alternating Training ────────────────────────────────────────────────

def alternating_training(forecaster, forecaster_optimizer, forecaster_loss_fn,
                         forecaster_loader: DataLoader,
                         rl_trainer, joint_env,
                         n_rounds: int = 20,
                         forecaster_epochs_per_round: int = 5,
                         rl_episodes_per_round: int = 100,
                         log_dir: str = "logs/joint_training"):
    """
    Alternating training: forecaster dan RL saling beradaptasi bertahap.
    - Fase A: latih forecaster, RL dibekukan
    - Fase B: latih RL, forecaster dibekukan
    Lebih stabil dari end-to-end karena dua loss tidak saling mengganggu.
    """
    writer = SummaryWriter(log_dir=log_dir)

    for round_idx in range(n_rounds):
        print(f"\n=== Round {round_idx+1}/{n_rounds} ===")

        # ── Fase A: Latih Forecaster ──────────────────────────────────────────
        for param in forecaster.parameters():
            param.requires_grad = True
        forecaster.train()

        for epoch in range(forecaster_epochs_per_round):
            for batch in forecaster_loader:
                sales_seq, sensor_seq, target = batch
                forecaster_optimizer.zero_grad()
                mu, log_var, _ = forecaster(sales_seq, sensor_seq)
                loss = forecaster_loss_fn(mu, log_var, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(forecaster.parameters(), max_norm=1.0)
                forecaster_optimizer.step()

        forecaster_loss_val = loss.item()
        print(f"  Forecaster loss akhir fase A: {forecaster_loss_val:.4f}")
        writer.add_scalar("forecaster/loss", forecaster_loss_val, round_idx)

        # ── Fase B: Latih RL ──────────────────────────────────────────────────
        for param in forecaster.parameters():
            param.requires_grad = False
        forecaster.eval()

        episode_rewards = run_rl_episodes(
            rl_trainer, joint_env, n_episodes=rl_episodes_per_round
        )
        avg_reward = np.mean(episode_rewards)
        print(f"  Rata-rata reward RL fase B: {avg_reward:.2f}")
        writer.add_scalar("rl/avg_reward", avg_reward, round_idx)

    writer.close()
    print("\n✅ Alternating training selesai.")
    return forecaster, rl_trainer