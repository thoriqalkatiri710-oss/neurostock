"""
BAGIAN 15.1 — Sensor Data Abstraction Layer
Memisahkan sumber data (simulasi vs asli) dari logika pemrosesan.
Migrasi ke sensor IoT asli tidak memerlukan perombakan besar.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


# ── Abstract Base ─────────────────────────────────────────────────────────────

class SensorDataSource(ABC):
    @abstractmethod
    def get_readings(self, store_id: str, date_range: tuple) -> dict:
        """
        Return dict berisi sensor readings untuk store_id dalam date_range.
        Format: {"temperature_c": [...], "foot_traffic": [...], "supply_delay_days": [...]}
        """
        ...


# ── Simulated Source (current implementation) ─────────────────────────────────

class SimulatedSensorSource(SensorDataSource):
    """Implementasi simulasi dari Bagian 2.3 — dipakai saat ini."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_readings(self, store_id: str, date_range: tuple) -> dict:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        dates = pd.date_range(start, end, freq="D")
        n = len(dates)

        day_of_year = dates.dayofyear.values
        temperature = (25 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                       + self.rng.normal(0, 1.5, n))
        weekend = np.where(dates.dayofweek >= 5, 1.3, 1.0)
        foot_traffic = (500 * weekend * (1 - 0.01 * np.abs(temperature - 24))
                        * self.rng.normal(1.0, 0.1, n))
        supply_delay = np.clip(self.rng.exponential(2.0, n), 0, 14)

        return {
            "dates": dates.tolist(),
            "temperature_c": temperature.tolist(),
            "foot_traffic": foot_traffic.tolist(),
            "supply_delay_days": supply_delay.tolist(),
        }


# ── IoT Source (future integration) ──────────────────────────────────────────

class IoTSensorSource(SensorDataSource):
    """
    Placeholder untuk integrasi sensor IoT asli di masa depan.
    Implementasi nyata akan query dari time-series database (mis. InfluxDB)
    yang menerima data streaming dari sensor IoT fisik via MQTT broker.
    """

    def __init__(self, mqtt_broker_url: str):
        self.broker_url = mqtt_broker_url

    def get_readings(self, store_id: str, date_range: tuple) -> dict:
        raise NotImplementedError(
            "Integrasi sensor IoT asli — pengembangan lanjutan. "
            f"Broker URL: {self.broker_url}"
        )


# ── Factory Function ──────────────────────────────────────────────────────────

def get_sensor_source(source_type: str = "simulated", **kwargs) -> SensorDataSource:
    """
    Factory untuk memilih sumber data sensor.
    source_type: "simulated" | "iot"
    """
    if source_type == "simulated":
        return SimulatedSensorSource(**kwargs)
    elif source_type == "iot":
        if "mqtt_broker_url" not in kwargs:
            raise ValueError("IoT source membutuhkan mqtt_broker_url")
        return IoTSensorSource(**kwargs)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


if __name__ == "__main__":
    print("── 15.1 Sensor Interface Demo ──\n")

    # Test simulated source
    source = get_sensor_source("simulated", seed=42)
    readings = source.get_readings("store_0", ("2026-01-01", "2026-01-07"))

    print(f"Source type    : SimulatedSensorSource")
    print(f"Store          : store_0")
    print(f"Date range     : 2026-01-01 to 2026-01-07")
    print(f"N readings     : {len(readings['dates'])}")
    print(f"Temperature    : {[round(t,1) for t in readings['temperature_c']]}")
    print(f"Foot traffic   : {[round(f,0) for f in readings['foot_traffic']]}")
    print(f"Supply delay   : {[round(d,1) for d in readings['supply_delay_days']]}")

    # Test IoT source (akan raise NotImplementedError)
    print("\n── IoT Source (placeholder) ──")
    try:
        iot_source = get_sensor_source("iot", mqtt_broker_url="mqtt://broker.example.com")
        iot_source.get_readings("store_0", ("2026-01-01", "2026-01-07"))
    except NotImplementedError as e:
        print(f"✅ NotImplementedError raised (expected): {e}")