import numpy as np
import optuna
import torch

from src.rl.environment import NeuroStockEnv
from src.rl.ppo_trainer import MAPPOTrainer

# ── Helper: Train & Evaluate ──────────────────────────────────────────────────

def run_training_and_evaluate(trainer: MAPPOTrainer, env: NeuroStockEnv,
                               n_episodes: int = 500,
                               eval_episodes: int = 20,
                               max_steps: int = 30) -> float:
    """Train n_episodes lalu evaluasi di eval_episodes terpisah."""

    # Training
    for episode in range(n_episodes):
        obs, _ = env.reset()
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

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            for agent in env.agents:
                buffers[agent]["obs"].append(obs[agent])
                buffers[agent]["actions"].append(actions[agent])
                buffers[agent]["log_probs"].append(log_probs[agent])
                buffers[agent]["rewards"].append(rewards[agent])
                buffers[agent]["values"].append(values[agent])
                buffers[agent]["dones"].append(float(terminations[agent]))

            obs = next_obs
            if all(terminations.values()):
                break

        # Update PPO
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
            trainer.update(agent, batch, n_epochs=4)

    # Evaluation
    eval_rewards = []
    for _ in range(eval_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            actions = {}
            for agent in env.agents:
                obs_tensor = torch.FloatTensor(obs[agent]).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = trainer.networks[agent].get_action(obs_tensor)
                actions[agent] = action.squeeze(0).detach().numpy()
            obs, rewards, terminations, truncations, _ = env.step(actions)
            total_reward += np.mean(list(rewards.values()))
            if all(terminations.values()):
                break
        eval_rewards.append(total_reward)

    return float(np.mean(eval_rewards))


# ── 10.2.1 RL Objective ───────────────────────────────────────────────────────

def rl_objective(trial: optuna.Trial, env_fn, obs_dim, action_dim):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    env = env_fn()
    trainer = MAPPOTrainer(
        agents=env.possible_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=lr,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        gae_lambda=gae_lambda
    )

    avg_reward = run_training_and_evaluate(
        trainer, env, n_episodes=500, eval_episodes=20
    )
    return avg_reward


def run_rl_hyperparameter_search(env_fn, obs_dim, action_dim,
                                  n_trials: int = 20):
    """
    Search hyperparameter RL. Gunakan n_trials kecil (10-20)
    karena tiap trial jauh lebih mahal dari forecasting.
    Jika terlalu lama, kurangi n_episodes di rl_objective.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: rl_objective(trial, env_fn, obs_dim, action_dim),
        n_trials=n_trials
    )
    print("Best RL hyperparameters:", study.best_params)
    print("Best avg reward:", study.best_value)
    return study


if __name__ == "__main__":
    print("── Demo RL Hyperparameter Search (2 trials, 5 episodes) ──")

    def env_fn():
        return NeuroStockEnv(n_stores=3, forecast_dim=7)

    env = env_fn()
    obs_dim = env.observation_space("store_0").shape[0]
    action_dim = env.action_space("store_0").shape[0]

    # Patch untuk demo cepat
    import src.rl.tune_rl as tune_module
    original_fn = tune_module.run_training_and_evaluate

    def fast_eval(trainer, env, **kwargs):
        obs, _ = env.reset()
        total = 0.0
        for _ in range(5):
            actions = {a: env.action_space(a).sample() for a in env.agents}
            _, rewards, _, _, _ = env.step(actions)
            total += np.mean(list(rewards.values()))
        return total / 5

    tune_module.run_training_and_evaluate = fast_eval

    study = run_rl_hyperparameter_search(env_fn, obs_dim, action_dim, n_trials=2)
    print(f"\nBest params: {study.best_params}")

    tune_module.run_training_and_evaluate = original_fn