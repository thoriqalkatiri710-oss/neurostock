import numpy as np
import pandas as pd
from scipy import stats

# ── 6.3.1 Paired t-test ───────────────────────────────────────────────────────

def compare_methods_significance(costs_method_a: list, costs_method_b: list,
                                  name_a: str = "Method A", name_b: str = "Method B",
                                  alpha: float = 0.05) -> tuple:
    """
    Paired t-test antar dua metode di seluruh fold walk-forward validation.
    Jalankan per fold, bukan hanya satu split — memberi sample cukup untuk uji valid.
    """
    t_stat, p_value = stats.ttest_rel(costs_method_a, costs_method_b)
    is_significant = p_value < alpha

    print(f"\n── Paired t-test: {name_a} vs {name_b} ──")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.4f}")
    print(f"  Perbedaan {'✅ signifikan' if is_significant else '⚠️ tidak signifikan'} "
          f"pada alpha={alpha}")

    if is_significant:
        winner = name_a if np.mean(costs_method_a) < np.mean(costs_method_b) else name_b
        print(f"  {winner} secara statistik lebih baik.")

    return t_stat, p_value, is_significant


def run_all_significance_tests(fold_costs: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    Jalankan paired t-test untuk semua pasangan metode.
    fold_costs = {
        "NeuroStock Joint":   [cost_fold1, cost_fold2, ...],
        "NeuroStock + (s,S)": [cost_fold1, cost_fold2, ...],
        "ARIMA + (s,S)":      [cost_fold1, cost_fold2, ...],
    }
    """
    methods = list(fold_costs.keys())
    rows = []

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            a, b = methods[i], methods[j]
            t_stat, p_value, is_sig = compare_methods_significance(
                fold_costs[a], fold_costs[b], name_a=a, name_b=b, alpha=alpha
            )
            rows.append({
                "Method A": a,
                "Method B": b,
                "t-stat": round(t_stat, 3),
                "p-value": round(p_value, 4),
                "Significant": "✅" if is_sig else "⚠️"
            })

    df = pd.DataFrame(rows)
    print("\n── Significance Test Summary ──")
    print(df.to_string(index=False))
    return df


# ── 6.4.1 Ablation Study ─────────────────────────────────────────────────────

def run_ablation_study(variants: dict, eval_fn) -> pd.DataFrame:
    """
    Ablation study — kontribusi tiap komponen arsitektur.
    Variants:
    1. Full model: Transformer + cross-modal + joint training
    2. No joint:   Transformer + cross-modal, tanpa joint training
    3. No cross:   Transformer tanpa cross-modal attention
    4. LSTM:       LSTM sebagai pengganti transformer
    5. Baseline:   ARIMA + (s,S) policy klasik
    """
    results = {}
    rows = []

    for name, model_fn in variants.items():
        print(f"── Running: {name} ──")
        model = model_fn()
        metrics = eval_fn(model)
        results[name] = metrics
        rows.append({
            "Variant": name,
            "MAPE": round(metrics.get("mape", 0), 2),
            "RMSE": round(metrics.get("rmse", 0), 2),
            "Total Cost": round(metrics.get("total_cost", 0), 2),
            "Service Level": round(metrics.get("service_level", 0), 4),
            "Stockout Rate": round(metrics.get("stockout_rate", 0), 4),
        })
        print(f"  MAPE={metrics.get('mape', 0):.2f}, "
              f"total_cost={metrics.get('total_cost', 0):.2f}, "
              f"service_level={metrics.get('service_level', 0):.4f}")

    df = pd.DataFrame(rows).set_index("Variant")
    print("\n── Ablation Study Results ──")
    print(df.to_string())
    return df


if __name__ == "__main__":
    np.random.seed(42)
    n_folds = 5

    # Simulasi fold costs per metode
    fold_costs = {
        "NeuroStock Joint":   list(np.random.normal(10000, 500, n_folds)),
        "NeuroStock + (s,S)": list(np.random.normal(11500, 600, n_folds)),
        "ARIMA + (s,S)":      list(np.random.normal(13000, 800, n_folds)),
    }

    run_all_significance_tests(fold_costs)

    # Ablation study demo
    def mock_model_fn(mape_val, cost_val, service_val):
        def fn():
            return {"type": "mock"}
        return fn

    def mock_eval(model):
        configs = {
            "mock": {"mape": np.random.uniform(5, 15),
                     "rmse": np.random.uniform(3, 10),
                     "total_cost": np.random.uniform(8000, 15000),
                     "service_level": np.random.uniform(0.85, 1.0),
                     "stockout_rate": np.random.uniform(0.0, 0.15)}
        }
        return configs["mock"]

    variants = {
        "1. Full Model (Transformer+CrossModal+Joint)": lambda: {"type": "mock"},
        "2. No Joint Training":                         lambda: {"type": "mock"},
        "3. No Cross-Modal Attention":                  lambda: {"type": "mock"},
        "4. LSTM Replace Transformer":                  lambda: {"type": "mock"},
        "5. ARIMA + (s,S) Baseline":                   lambda: {"type": "mock"},
    }

    run_ablation_study(variants, mock_eval)