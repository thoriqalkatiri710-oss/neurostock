import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# ── 6.1.1 Baseline Forecasting ───────────────────────────────────────────────

def fit_arima_baseline(train_series, order=(2, 1, 2)):
    """ARIMA sebagai baseline statistik klasik."""
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    return fitted


def forecast_arima(fitted_model, horizon: int):
    """Return mean forecast dan confidence interval 80%."""
    forecast = fitted_model.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.2)  # interval 80%
    return mean, conf_int


def fit_prophet_baseline(df_train: pd.DataFrame):
    """
    Prophet sebagai baseline kedua — lebih robust terhadap multiple seasonality.
    df_train harus punya kolom 'ds' (tanggal) dan 'y' (target).
    """
    model = Prophet(
        interval_width=0.8,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    model.fit(df_train)
    return model


def forecast_prophet(model, horizon: int) -> pd.DataFrame:
    """Return forecast dengan prediction interval."""
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)


# ── 6.1.2 Baseline Inventory Optimization ────────────────────────────────────

def s_S_policy(inventory: float, forecast_demand: float, forecast_std: float,
               s_multiplier: float = 1.5, S_multiplier: float = 3.0) -> float:
    """
    Kebijakan (s, S) klasik:
    - Order sampai level S ketika inventori turun di bawah s
    - s = reorder point (demand + safety stock)
    - S = order-up-to level
    Baseline penting untuk membuktikan apakah joint RL lebih baik
    dari pipeline forecast-lalu-optimasi yang lebih sederhana.
    """
    s = forecast_demand + s_multiplier * forecast_std   # reorder point
    S = forecast_demand + S_multiplier * forecast_std   # order-up-to level
    if inventory < s:
        return max(0.0, S - inventory)
    return 0.0


def simulate_s_S_baseline(demand_series: np.ndarray, forecast_mu: np.ndarray,
                           forecast_std: np.ndarray, initial_inventory: float = 100.0,
                           lead_time: int = 3) -> dict:
    """
    Simulasi full episode menggunakan (s,S) policy.
    Return metrik untuk dibandingkan dengan RL agent.
    """
    inventory = initial_inventory
    pending_orders = []
    total_holding_cost = 0.0
    total_stockout = 0.0
    total_service = 0.0
    n_steps = len(demand_series)

    for t in range(n_steps):
        # Proses order yang tiba
        arriving = sum(qty for qty, arrival in pending_orders if arrival <= t)
        pending_orders = [(qty, arr) for qty, arr in pending_orders if arr > t]
        inventory += arriving

        # Demand
        demand = demand_series[t]
        inventory_after = inventory - demand

        # Metrik
        total_holding_cost += max(inventory_after, 0) * 0.5
        total_stockout += max(-inventory_after, 0) * 2.0
        total_service += min(demand, inventory) / (demand + 1e-6)

        inventory = max(inventory_after, 0)

        # (s,S) order decision
        mu = forecast_mu[t] if t < len(forecast_mu) else forecast_mu[-1]
        std = forecast_std[t] if t < len(forecast_std) else forecast_std[-1]
        order_qty = s_S_policy(inventory, mu, std)
        if order_qty > 0:
            pending_orders.append((order_qty, t + lead_time))

    return {
        "total_holding_cost": total_holding_cost,
        "total_stockout": total_stockout,
        "avg_service_rate": total_service / n_steps,
        "stockout_rate": total_stockout / (n_steps * 50),  # normalized
    }


if __name__ == "__main__":
    # Verifikasi (s,S) policy
    print("── (s,S) Policy Test ──")
    scenarios = [
        ("Low inventory",  20.0, 50.0, 10.0),
        ("High inventory", 200.0, 50.0, 10.0),
        ("Borderline",     75.0, 50.0, 10.0),
    ]
    for name, inv, mu, std in scenarios:
        order = s_S_policy(inv, mu, std)
        print(f"  {name}: inventory={inv}, order={order:.1f}")

    # Simulasi baseline
    np.random.seed(42)
    demand = np.random.uniform(20, 80, size=365)
    mu = np.random.uniform(30, 60, size=365)
    std = np.full(365, 10.0)

    results = simulate_s_S_baseline(demand, mu, std)
    print("\n── s_S Baseline Simulation ──")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")