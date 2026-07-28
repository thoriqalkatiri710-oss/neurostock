"""
BAGIAN 13 — ANALISIS KOMPLEKSITAS & SKALABILITAS
Analisis teoritis + empiris kompleksitas komputasi sistem NeuroStock.
"""

import time

from src.rl.environment import NeuroStockEnv

# ── 13.1.1 Transformer Complexity ────────────────────────────────────────────

def analyze_attention_complexity(sequence_lengths: list, d_model: int = 64):
    """
    Analisis kompleksitas self-attention: O(T² · d)
    Space complexity: O(T²) untuk attention matrix.
    """
    print("── 13.1.1 Self-Attention Complexity ──")
    print(f"{'T (seq_len)':<15} {'T²':<12} {'T²·d':<15} {'Memory (KB)':<15}")
    print("-" * 55)
    for T in sequence_lengths:
        t_sq = T ** 2
        t_sq_d = T ** 2 * d_model
        memory_kb = t_sq * 4 / 1024  # float32 = 4 bytes
        print(f"{T:<15} {t_sq:<12,} {t_sq_d:<15,} {memory_kb:<15.2f}")

    print(f"\nRasio T=365 vs T=90: {365**2 / 90**2:.1f}x lebih besar")
    print("Mitigasi: sparse attention atau sliding window O(T·w)")


# ── 13.1.2 Model Parameter Estimation ────────────────────────────────────────

def estimate_model_params(d_model: int, n_layers: int, n_heads: int,
                           d_ff_multiplier: int = 4) -> int:
    """Estimasi jumlah parameter transformer forecaster."""
    attn_params = 4 * d_model * d_model          # Q, K, V, O
    ff_params = 2 * d_model * (d_model * d_ff_multiplier)
    per_layer = attn_params + ff_params
    total = per_layer * n_layers
    return total


def analyze_model_size():
    print("\n── 13.1.2 Model Parameter Estimation ──")
    configs = [
        {"d_model": 32,  "n_layers": 2, "n_heads": 2},
        {"d_model": 64,  "n_layers": 3, "n_heads": 4},
        {"d_model": 128, "n_layers": 4, "n_heads": 8},
    ]
    print(f"{'d_model':<10} {'n_layers':<10} {'Params':<15} {'Size (KB)':<12}")
    print("-" * 47)
    for cfg in configs:
        params = estimate_model_params(**cfg)
        size_kb = params * 4 / 1024  # float32
        print(f"{cfg['d_model']:<10} {cfg['n_layers']:<10} {params:<15,} {size_kb:<12.1f}")

    default_params = estimate_model_params(d_model=64, n_layers=3, n_heads=4)
    print(f"\nDefault config: {default_params:,} parameter — training bisa di CPU ✅")


# ── 13.2.1 Multi-Agent Joint Action Space ────────────────────────────────────

def analyze_marl_complexity(agent_counts: list, action_dim: int = 2):
    print("\n── 13.2.1 Multi-Agent Action Space ──")
    print(f"{'N agents':<12} {'Joint dim':<15} {'CTDE policy dim':<20}")
    print("-" * 47)
    for N in agent_counts:
        joint_dim = N * action_dim
        ctde_dim = action_dim  # setiap agent tetap policy individual
        print(f"{N:<12} {joint_dim:<15} {ctde_dim:<20}")
    print("\nCTDE: setiap agent belajar π_i(a_i|o_i) — scalable vs fully centralized")


# ── 13.2.2 Training Speed Benchmark ─────────────────────────────────────────

def benchmark_training_speed(env, n_episodes_sample: int = 50) -> float:
    """Benchmark empiris waktu per episode."""
    start = time.time()
    for _ in range(n_episodes_sample):
        obs, _ = env.reset()
        done = False
        while not done:
            actions = {agent: env.action_space(agent).sample()
                       for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)
            done = all(truncs.values())
    elapsed = time.time() - start
    per_episode = elapsed / n_episodes_sample

    print("\n── 13.2.2 Training Speed Benchmark ──")
    print(f"Sample episodes     : {n_episodes_sample}")
    print(f"Total elapsed       : {elapsed:.2f} detik")
    print(f"Per episode         : {per_episode:.3f} detik")
    print(f"Estimasi 5000 ep    : {per_episode * 5000 / 60:.1f} menit")
    print(f"Estimasi 1000 ep    : {per_episode * 1000 / 60:.1f} menit")
    return per_episode


# ── 13.3.1 Scalability Study ─────────────────────────────────────────────────

def scalability_study(store_counts: list, n_episodes: int = 20) -> dict:
    """
    Ukur waktu training vs jumlah store agent.
    Jika mendekati linear → bukti empiris CTDE scalable.
    Jika super-linear → dokumentasikan sebagai keterbatasan.
    """
    print("\n── 13.3.1 Scalability Study ──")
    print(f"{'N stores':<12} {'Total (s)':<12} {'Per store (s)':<15} {'Episodes'}")
    print("-" * 50)

    results = {}
    for n_stores in store_counts:
        env = NeuroStockEnv(n_stores=n_stores)
        start = time.time()

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                actions = {agent: env.action_space(agent).sample()
                           for agent in env.agents}
                obs, rewards, terms, truncs, infos = env.step(actions)
                done = all(truncs.values())

        elapsed = time.time() - start
        per_store = elapsed / n_stores
        results[n_stores] = {
            "training_time_sec": round(elapsed, 2),
            "time_per_store": round(per_store, 3)
        }
        print(f"{n_stores:<12} {elapsed:<12.2f} {per_store:<15.3f} {n_episodes}")

    # Analisis linearitas
    times = [results[n]["training_time_sec"] for n in store_counts]
    ratios = [times[i] / times[0] for i in range(len(store_counts))]
    store_ratios = [store_counts[i] / store_counts[0] for i in range(len(store_counts))]

    print("\n── Linearitas ──")
    for i, n in enumerate(store_counts):
        linearity = ratios[i] / store_ratios[i] if store_ratios[i] > 0 else 1
        status = "✅ linear" if linearity < 1.3 else "⚠️ super-linear"
        print(f"  N={n}: time_ratio={ratios[i]:.2f}, store_ratio={store_ratios[i]:.2f} → {status}")

    return results


if __name__ == "__main__":
    # 13.1 Transformer complexity
    analyze_attention_complexity([30, 90, 180, 365], d_model=64)
    analyze_model_size()

    # 13.2 MARL complexity
    analyze_marl_complexity([2, 5, 10, 20, 50])

    # 13.2.2 Benchmark
    env = NeuroStockEnv(n_stores=5)
    benchmark_training_speed(env, n_episodes_sample=20)

    # 13.3 Scalability study
    scalability_study(store_counts=[2, 5, 10], n_episodes=10)