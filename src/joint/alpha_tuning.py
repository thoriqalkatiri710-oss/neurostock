import numpy as np
from typing import Callable


# ── 5.4.1 Alpha Sensitivity Study ────────────────────────────────────────────

def alpha_sensitivity_study(alphas: list, train_fn: Callable,
                             eval_fn: Callable) -> dict:
    """
    Grid search kecil pada alpha untuk temukan keseimbangan optimal
    antara akurasi forecast dan kualitas keputusan inventori.

    Interpretasi:
    - alpha → 1.0: prioritas forecast accuracy (MAPE rendah)
    - alpha → 0.0: prioritas inventory decision quality (stockout rendah)
    - Pilih alpha sesuai prioritas bisnis
    """
    results = {}

    for alpha in alphas:
        print(f"\n── Testing alpha={alpha} ──")
        model_state = train_fn(alpha=alpha)
        metrics = eval_fn(model_state)
        results[alpha] = metrics
        print(f"  alpha={alpha}: "
              f"forecast_mape={metrics.get('mape', 0):.2f}, "
              f"stockout_rate={metrics.get('stockout_rate', 0):.2f}, "
              f"avg_reward={metrics.get('avg_reward', 0):.2f}")

    # Ringkasan
    print("\n── Alpha Sensitivity Summary ──")
    print(f"{'Alpha':<8} {'MAPE':<10} {'Stockout':<12} {'Avg Reward'}")
    print("-" * 45)
    for alpha, metrics in sorted(results.items()):
        print(f"{alpha:<8} "
              f"{metrics.get('mape', 0):<10.2f} "
              f"{metrics.get('stockout_rate', 0):<12.2f} "
              f"{metrics.get('avg_reward', 0):.2f}")

    best_alpha = _select_best_alpha(results)
    print(f"\n✅ Recommended alpha: {best_alpha}")
    return results


def _select_best_alpha(results: dict) -> float:
    """
    Pilih alpha dengan skor gabungan terbaik:
    score = -mape_normalized - stockout_normalized + reward_normalized
    """
    alphas = list(results.keys())
    mapes = np.array([results[a].get("mape", 0) for a in alphas])
    stockouts = np.array([results[a].get("stockout_rate", 0) for a in alphas])
    rewards = np.array([results[a].get("avg_reward", 0) for a in alphas])

    # Normalisasi ke [0, 1]
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / (r + 1e-8)

    scores = -norm(mapes) - norm(stockouts) + norm(rewards)
    best_idx = np.argmax(scores)
    return alphas[best_idx]


def dummy_alpha_study_demo():
    """Demo alpha study dengan hasil simulasi — untuk verifikasi struktur kode."""

    def mock_train_fn(alpha: float):
        return {"alpha": alpha}

    def mock_eval_fn(model_state):
        alpha = model_state["alpha"]
        # Simulasi trade-off: alpha tinggi → mape rendah, stockout tinggi
        return {
            "mape": 10.0 * (1 - alpha) + 5.0,
            "stockout_rate": 0.1 * alpha + 0.05,
            "avg_reward": -1000 * (1 - alpha) - 500 * alpha
        }

    alphas = [0.3, 0.5, 0.6, 0.7, 0.9]
    return alpha_sensitivity_study(alphas, mock_train_fn, mock_eval_fn)


if __name__ == "__main__":
    dummy_alpha_study_demo()