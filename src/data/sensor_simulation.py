import numpy as np
import pandas as pd


# ── 2.3.1 Simulasi Sensor ─────────────────────────────────────────────────────

def simulate_temperature(
    dates: pd.DatetimeIndex,
    base_temp: float = 25.0,
    amplitude: float = 8.0,
    noise_std: float = 1.5,
    seed: int = 42
) -> np.ndarray:
    """Suhu harian dengan pola musiman sinusoidal + noise Gaussian."""
    rng = np.random.default_rng(seed)
    day_of_year = dates.dayofyear.values
    seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    noise = rng.normal(0, noise_std, size=len(dates))
    return base_temp + seasonal + noise


def simulate_foot_traffic(
    dates: pd.DatetimeIndex,
    temperature: np.ndarray,
    base_traffic: float = 500,
    seed: int = 42
) -> np.ndarray:
    """Foot traffic berkorelasi dengan hari dalam minggu dan suhu."""
    rng = np.random.default_rng(seed)
    day_of_week_effect = np.where(dates.dayofweek >= 5, 1.3, 1.0)  # weekend lebih ramai
    temp_effect = 1 - 0.01 * np.abs(temperature - 24)               # nyaman di ~24°C
    noise = rng.normal(1.0, 0.1, size=len(dates))
    return base_traffic * day_of_week_effect * temp_effect * noise


def simulate_supply_chain_delay(
    dates: pd.DatetimeIndex,
    seed: int = 42
) -> np.ndarray:
    """Delay pengiriman dalam hari — lebih tinggi di akhir bulan."""
    rng = np.random.default_rng(seed)
    end_of_month_effect = np.where(dates.day >= 25, 1.5, 1.0)
    base_delay = rng.exponential(scale=2.0, size=len(dates))
    return np.clip(base_delay * end_of_month_effect, 0, 14).astype(float)


def validate_correlation(sensor: np.ndarray, demand: np.ndarray, label: str = "sensor") -> float:
    """Validasi korelasi sensor-demand harus masuk akal (0.1 - 0.6)."""
    corr = np.corrcoef(sensor, demand)[0, 1]
    assert -1 <= corr <= 1
    status = "✅" if 0.1 <= abs(corr) <= 0.6 else "⚠️ di luar rentang wajar"
    print(f"Korelasi {label}-demand: {corr:.3f} {status}")
    return corr


def generate_sensor_data(clean_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(clean_csv, parse_dates=["date"])
    rng = np.random.default_rng(42)

    n = len(df)
    units_sold = df["units_sold"].values

    # Normalisasi units_sold ke skala 0-1 sebagai sinyal dasar
    demand_signal = (units_sold - units_sold.min()) / (units_sold.max() - units_sold.min() + 1e-8)

    # Temperature: korelasi negatif dengan demand (cuaca panas → orang keluar → beli lebih)
    temperature_base = 25 + 8 * np.sin(2 * np.pi * (df["date"].dt.dayofyear.values - 80) / 365)
    temperature_corr = 3.0 * demand_signal          # komponen korelasi
    temperature_noise = rng.normal(0, 1.5, size=n)
    df["temperature_c"] = temperature_base + temperature_corr + temperature_noise

    # Foot traffic: korelasi positif dengan demand
    weekend_effect = np.where(df["date"].dt.dayofweek >= 5, 1.3, 1.0)
    foot_base = 500 * weekend_effect
    foot_corr = 150 * demand_signal                 # komponen korelasi
    foot_noise = rng.normal(0, 50, size=n)
    df["foot_traffic"] = np.clip(foot_base + foot_corr + foot_noise, 0, None)

    # Supply delay: korelasi negatif dengan demand (delay → stok kurang → demand tak terpenuhi)
    delay_base = rng.exponential(scale=2.0, size=n)
    delay_corr = -1.5 * demand_signal               # komponen korelasi negatif
    df["supply_delay_days"] = np.clip(delay_base + delay_corr + 1.5, 0, 14)

    # Validasi korelasi
    print("\n── Validasi Korelasi Sensor-Demand ──")
    validate_correlation(df["temperature_c"].values, units_sold, "temperature")
    validate_correlation(df["foot_traffic"].values, units_sold, "foot_traffic")
    validate_correlation(df["supply_delay_days"].values, units_sold, "supply_delay")

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = generate_sensor_data(
        clean_csv="data/processed/inventory_clean.csv",
        output_csv="data/simulated/inventory_with_sensors.csv"
    )
    print(df[["date", "temperature_c", "foot_traffic", "supply_delay_days"]].head(10))