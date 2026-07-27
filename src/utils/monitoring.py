import numpy as np
import csv
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp


# ── 12.3.1 Feature Drift Detection ───────────────────────────────────────────

def detect_feature_drift(reference_data: np.ndarray, current_data: np.ndarray,
                          feature_name: str, alpha: float = 0.05) -> dict:
    """
    Kolmogorov-Smirnov test untuk mendeteksi pergeseran distribusi.
    - reference_data: sample dari data training
    - current_data: sample dari data production terbaru
    Jadwalkan pengecekan ini mingguan — drift signifikan = sinyal retrain.
    """
    statistic, p_value = ks_2samp(reference_data, current_data)
    drift_detected = p_value < alpha
    return {
        "feature": feature_name,
        "ks_statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drift_detected": drift_detected,
    }


def monitor_all_features(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                          feature_cols: list) -> list:
    results = []
    for col in feature_cols:
        result = detect_feature_drift(
            reference_df[col].values, current_df[col].values, col
        )
        results.append(result)
        if result["drift_detected"]:
            print(f"⚠️  Drift terdeteksi pada '{col}' (p={result['p_value']:.4f})")
        else:
            print(f"✅ No drift: '{col}' (p={result['p_value']:.4f})")
    return results


# ── 12.3.2 Prediction Logging ─────────────────────────────────────────────────

def log_prediction_outcome(store_id: str, product_id: str, date: str,
                            predicted_mu: float, predicted_sigma: float,
                            actual_demand: float = None,
                            log_path: str = "logs/prediction_log.csv"):
    """
    Log prediksi vs aktual untuk monitoring performa berkelanjutan.
    Setelah 30 hari, hitung rolling MAPE production untuk deteksi degradasi model.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    file_exists = Path(log_path).exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["store_id", "product_id", "date", "predicted_mu",
                             "predicted_sigma", "actual_demand", "abs_error"])
        abs_error = abs(predicted_mu - actual_demand) if actual_demand is not None else None
        writer.writerow([store_id, product_id, date, predicted_mu,
                        predicted_sigma, actual_demand, abs_error])


def compute_rolling_mape(log_path: str = "logs/prediction_log.csv",
                          window: int = 30) -> float:
    """Hitung rolling MAPE dari log prediksi — deteksi degradasi model."""
    df = pd.read_csv(log_path).dropna(subset=["actual_demand", "abs_error"])
    if len(df) < window:
        print(f"⚠️ Data log hanya {len(df)} baris, butuh minimal {window}")
        return None
    recent = df.tail(window)
    mape = (recent["abs_error"] / (recent["actual_demand"] + 1e-6)).mean() * 100
    print(f"Rolling MAPE ({window} hari terakhir): {mape:.2f}%")
    return mape


if __name__ == "__main__":
    import pandas as pd
    np.random.seed(42)

    print("── Demo Feature Drift Detection ──")
    reference = pd.DataFrame({
        "units_sold": np.random.normal(50, 10, 200),
        "price": np.random.normal(100, 5, 200),
    })
    # Simulasi drift: distribusi bergeser
    current_no_drift = pd.DataFrame({
        "units_sold": np.random.normal(50, 10, 50),
        "price": np.random.normal(100, 5, 50),
    })
    current_with_drift = pd.DataFrame({
        "units_sold": np.random.normal(70, 15, 50),  # mean bergeser
        "price": np.random.normal(120, 8, 50),        # mean bergeser
    })

    print("\nNo drift scenario:")
    monitor_all_features(reference, current_no_drift, ["units_sold", "price"])

    print("\nWith drift scenario:")
    monitor_all_features(reference, current_with_drift, ["units_sold", "price"])

    print("\n── Demo Prediction Logging ──")
    for i in range(5):
        log_prediction_outcome(
            store_id="store_0", product_id="P0001",
            date=f"2026-07-{i+1:02d}",
            predicted_mu=62.4 + i,
            predicted_sigma=2.46,
            actual_demand=64.0 + np.random.normal(0, 3)
        )
    print("✅ Prediction log tersimpan di logs/prediction_log.csv")