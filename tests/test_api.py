from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("✅ health check OK")


def test_forecast_endpoint_valid_input():
    payload = {
        "store_id": "store_0",
        "product_id": "P0001",
        "lookback_data": [{"date": "2026-01-01", "units_sold": 45}] * 90,
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    assert "forecast_mu" in response.json()
    assert "forecast_sigma" in response.json()
    print("✅ forecast endpoint OK")


def test_recommend_endpoint():
    payload = {
        "store_id": "store_0",
        "product_id": "P0001",
        "lookback_data": [{"date": "2026-01-01", "units_sold": 45}] * 90,
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_order_qty" in data
    assert "confidence_interval_80" in data
    print("✅ recommend endpoint OK")


if __name__ == "__main__":
    test_health_check()
    test_forecast_endpoint_valid_input()
    test_recommend_endpoint()
    print("\n✅ All API tests passed")