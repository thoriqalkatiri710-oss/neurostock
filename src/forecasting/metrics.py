import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def coverage_probability(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Persentase observasi aktual yang jatuh di dalam prediction interval.
    Metrik kalibrasi untuk probabilistic forecasting:
    - Jika klaim 80% interval → angka ini harus mendekati 80%
    - Jauh meleset → model uncertainty tidak terkalibrasi dengan baik
    """
    within = (y_true >= lower) & (y_true <= upper)
    return np.mean(within) * 100


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 lower: np.ndarray = None, upper: np.ndarray = None) -> dict:
    """Hitung semua metrik sekaligus dan print ringkasan."""
    results = {
        "MAPE": mape(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }
    if lower is not None and upper is not None:
        results["Coverage (%)"] = coverage_probability(y_true, lower, upper)

    print("\n── Evaluation Metrics ──")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results