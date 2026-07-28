import torch

from src.forecasting.losses import GaussianNLLLoss

# ── 5.3.1 End-to-End Differentiable Training ─────────────────────────────────

def compute_rl_loss(actor_critic_dict: dict, rl_batch: dict,
                    mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Hitung RL loss menggunakan mu/sigma dari forecaster sebagai bagian observasi.
    Graph komputasi tersambung (bukan dari cache) agar gradient bisa mengalir balik.

    PERINGATAN: PPO tidak didesain untuk backprop lewat observasi.
    Ini adalah starting point eksperimen — bukan resep baku.
    Gunakan reparametrization trick jika ingin gradient stabil.
    """
    sigma = torch.exp(0.5 * log_var)
    total_loss = torch.tensor(0.0, requires_grad=True)

    for agent_name, ac_net in actor_critic_dict.items():
        if agent_name not in rl_batch:
            continue
        batch = rl_batch[agent_name]

        # Rekonstruksi observasi dengan mu/sigma terhubung ke graph
        obs = batch["obs"]
        # Sisipkan mu dan sigma ke dalam observasi (posisi 1:1+H dan 1+H:1+2H)
        horizon = mu.shape[-1]
        obs_with_forecast = obs.clone()
        obs_with_forecast[:, 1:1+horizon] = mu.detach()         # mu
        obs_with_forecast[:, 1+horizon:1+2*horizon] = sigma      # sigma — gradient mengalir di sini

        mean, std, values = ac_net(obs_with_forecast)
        dist = torch.distributions.Normal(mean, std)
        new_log_probs = dist.log_prob(batch["actions"]).sum(dim=-1)

        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_log_probs = batch["log_probs"]
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), batch["returns"])

        agent_loss = policy_loss + 0.5 * value_loss
        total_loss = total_loss + agent_loss

    return total_loss / max(len(actor_critic_dict), 1)


def joint_loss_step(forecaster, actor_critic_dict: dict,
                    forecaster_optimizer, rl_optimizers: dict,
                    batch: tuple, alpha: float = 0.6) -> dict:
    """
    Single joint training step (Persamaan 1.5).
    alpha mengontrol bobot forecasting vs RL:
    - alpha=1.0: pure forecasting
    - alpha=0.0: pure RL
    - alpha=0.6: default — forecasting lebih dominan di awal
    """
    sales_seq, sensor_seq, target, rl_batch = batch
    gaussian_nll = GaussianNLLLoss()

    # Forward forecaster TANPA no_grad — gradient harus bisa mengalir
    mu, log_var, _ = forecaster(sales_seq, sensor_seq)
    forecast_loss = gaussian_nll(mu, log_var, target)

    # Forward RL dengan observasi yang terhubung ke graph forecaster
    rl_loss = compute_rl_loss(actor_critic_dict, rl_batch, mu, log_var)

    # Joint loss (Persamaan 1.5)
    total_loss = alpha * forecast_loss + (1 - alpha) * rl_loss

    # Zero grad semua optimizer
    forecaster_optimizer.zero_grad()
    for opt in rl_optimizers.values():
        opt.zero_grad()

    total_loss.backward()

    # Clip gradient
    torch.nn.utils.clip_grad_norm_(forecaster.parameters(), max_norm=1.0)
    for ac_net in actor_critic_dict.values():
        torch.nn.utils.clip_grad_norm_(ac_net.parameters(), max_norm=0.5)

    forecaster_optimizer.step()
    for opt in rl_optimizers.values():
        opt.step()

    return {
        "forecast_loss": forecast_loss.item(),
        "rl_loss": rl_loss.item(),
        "total_loss": total_loss.item()
    }