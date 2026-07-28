import datetime
import json
import os

import pandas as pd

# ── 10.4.1 Experiment Logging ─────────────────────────────────────────────────

def log_experiment(name: str, config: dict, metrics: dict, notes: str = "") -> str:
    """
    Log satu eksperimen ke file JSON.
    Setiap eksperimen tersimpan terpisah agar bisa dibandingkan dan direproduksi.
    """
    os.makedirs("results/experiments", exist_ok=True)

    record = {
        "name": name,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }

    timestamp_clean = record["timestamp"].replace(":", "-").replace(".", "-")
    log_path = f"results/experiments/{name}_{timestamp_clean}.json"

    with open(log_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"✅ Experiment logged: {log_path}")
    _update_summary(record)
    return log_path


def _update_summary(record: dict):
    """Update tabel ringkasan di results/experiment_summary.md."""
    summary_path = "results/experiment_summary.md"

    # Baca existing summary atau buat baru
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            existing = f.read()
    else:
        existing = "# Experiment Summary\n\n"
        existing += "| Name | Timestamp | Key Metrics | Notes |\n"
        existing += "|---|---|---|---|\n"

    # Format metrics ringkas
    metrics_str = " | ".join([f"{k}={v}" for k, v in list(record["metrics"].items())[:3]])
    new_row = (f"| {record['name']} | {record['timestamp'][:19]} | "
               f"{metrics_str} | {record['notes']} |\n")

    with open(summary_path, "w") as f:
        f.write(existing + new_row)


def load_all_experiments(exp_dir: str = "results/experiments") -> pd.DataFrame:
    """Load semua eksperimen dari folder dan kembalikan sebagai DataFrame."""
    records = []
    if not os.path.exists(exp_dir):
        return pd.DataFrame()

    for fname in os.listdir(exp_dir):
        if fname.endswith(".json"):
            with open(os.path.join(exp_dir, fname)) as f:
                rec = json.load(f)
                flat = {
                    "name": rec["name"],
                    "timestamp": rec["timestamp"],
                    "notes": rec.get("notes", ""),
                }
                flat.update({f"config_{k}": v for k, v in rec.get("config", {}).items()})
                flat.update({f"metric_{k}": v for k, v in rec.get("metrics", {}).items()})
                records.append(flat)

    return pd.DataFrame(records).sort_values("timestamp", ascending=False)


if __name__ == "__main__":
    # Demo logging beberapa eksperimen
    experiments = [
        {
            "name": "forecaster_baseline",
            "config": {"d_model": 64, "n_heads": 4, "n_layers": 3, "lr": 1e-4},
            "metrics": {"mape": 18.4, "rmse": 9.7, "val_loss": 1.45},
            "notes": "ARIMA baseline"
        },
        {
            "name": "forecaster_transformer",
            "config": {"d_model": 64, "n_heads": 4, "n_layers": 3, "lr": 1e-4},
            "metrics": {"mape": 12.1, "rmse": 6.2, "val_loss": 1.12},
            "notes": "Transformer tanpa cross-modal"
        },
        {
            "name": "forecaster_crossmodal",
            "config": {"d_model": 64, "n_heads": 4, "n_layers": 3, "lr": 1e-4},
            "metrics": {"mape": 11.8, "rmse": 5.9, "val_loss": 1.08},
            "notes": "Full model dengan cross-modal attention"
        },
        {
            "name": "rl_mappo_joint",
            "config": {"lr": 3e-4, "clip_eps": 0.2, "entropy_coef": 0.01, "n_stores": 5},
            "metrics": {"avg_reward": -23241.0, "service_level": 0.943, "total_cost": 10045.0},
            "notes": "MAPPO joint training"
        },
    ]

    print("── Logging Experiments ──")
    for exp in experiments:
        log_experiment(**exp)

    print("\n── Experiment Summary ──")
    df = load_all_experiments()
    if not df.empty:
        print(df[["name", "timestamp", "metric_mape", "metric_rmse"]].to_string(index=False))

    print("\n✅ Summary tersimpan di results/experiment_summary.md")