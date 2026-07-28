import numpy as np
import pandas as pd

from src.forecasting.metrics import coverage_probability, mape, rmse

# ── 6.2.1 Forecasting Metrics Comparison ─────────────────────────────────────

def compare_forecasting_models(results: dict) -> pd.DataFrame:
    """
    Bandingkan metrik forecasting antar model.
    results = {
        "NeuroStock": {"y_true": ..., "y_pred": ..., "lower": ..., "upper": ...},
        "ARIMA":      {"y_true": ..., "y_pred": ..., "lower": ..., "upper": ...},
        "Prophet":    {"y_true": ..., "y_pred": ..., "lower": ..., "upper": ...},
    }
    """
    rows = []
    for model_name, data in results.items():
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        row = {
            "Model": model_name,
            "MAPE": round(mape(y_true, y_pred), 4),
            "RMSE": round(rmse(y_true, y_pred), 4),
        }
        if "lower" in data and "upper" in data:
            row["Coverage (%)"] = round(
                coverage_probability(y_true, data["lower"], data["upper"]), 2
            )
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    print("\n── Forecasting Model Comparison ──")
    print(df.to_string())
    return df


# ── 6.2.2 Business Metrics ────────────────────────────────────────────────────

def compute_business_metrics(simulation_log: list) -> dict:
    """
    Hitung metrik bisnis dari log simulasi.
    simulation_log: list of dict per timestep:
        {"inventory": float, "demand": float}
    """
    total_holding_cost = sum(max(s["inventory"], 0) * 0.5 for s in simulation_log)
    total_stockout_cost = sum(max(-s["inventory"], 0) * 2.0 for s in simulation_log)
    n_stockout_days = sum(1 for s in simulation_log if s["inventory"] < 0)
    service_level = 1 - (n_stockout_days / len(simulation_log))

    return {
        "total_holding_cost": round(total_holding_cost, 2),
        "total_stockout_cost": round(total_stockout_cost, 2),
        "stockout_rate": round(n_stockout_days / len(simulation_log), 4),
        "service_level": round(service_level, 4),
        "total_cost": round(total_holding_cost + total_stockout_cost, 2),
    }


def compare_inventory_systems(system_logs: dict) -> pd.DataFrame:
    """
    Tabel perbandingan total_cost antar sistem.
    system_logs = {
        "NeuroStock Joint":       [...simulation_log...],
        "NeuroStock + (s,S)":     [...simulation_log...],
        "ARIMA + (s,S)":          [...simulation_log...],
    }
    Bukti kuantitatif utama untuk README dan portfolio.
    """
    rows = []
    for system_name, log in system_logs.items():
        metrics = compute_business_metrics(log)
        metrics["System"] = system_name
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("System")
    print("\n── Inventory System Comparison ──")
    print(df.to_string())

    # Highlight sistem terbaik
    best = df["total_cost"].idxmin()
    print(f"\n✅ Best system (lowest total cost): {best}")
    return df


if __name__ == "__main__":
    # Demo dengan data simulasi
    np.random.seed(42)
    n = 365

    def make_log(inv_mean, inv_std):
        return [
            {"inventory": np.random.normal(inv_mean, inv_std),
             "demand": np.random.uniform(20, 80)}
            for _ in range(n)
        ]

    system_logs = {
        "NeuroStock Joint":   make_log(80, 20),
        "NeuroStock + (s,S)": make_log(60, 30),
        "ARIMA + (s,S)":      make_log(40, 40),
    }

    df = compare_inventory_systems(system_logs)

    # Demo forecasting comparison
    y_true = np.random.uniform(20, 80, size=100)
    forecast_results = {
        "NeuroStock": {
            "y_true": y_true,
            "y_pred": y_true + np.random.normal(0, 5, 100),
            "lower":  y_true - 10,
            "upper":  y_true + 10,
        },
        "ARIMA": {
            "y_true": y_true,
            "y_pred": y_true + np.random.normal(0, 10, 100),
            "lower":  y_true - 15,
            "upper":  y_true + 15,
        },
        "Prophet": {
            "y_true": y_true,
            "y_pred": y_true + np.random.normal(0, 8, 100),
            "lower":  y_true - 12,
            "upper":  y_true + 12,
        },
    }
    compare_forecasting_models(forecast_results)