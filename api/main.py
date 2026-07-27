from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import os

app = FastAPI(title="NeuroStock API", version="1.0")
import time
import logging
from fastapi import Request

logger = logging.getLogger("neurostock_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/api.log"),
    ]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {duration:.3f}s")
    return response
    
# Load models langsung saat module diimport
from src.forecasting.model import NeuroStockForecaster
from src.rl.networks import ActorCritic

forecaster = NeuroStockForecaster(n_sales_features=20, n_sensor_features=3)
forecaster.eval()

rl_policies = {}
for _agent in [f"store_{i}" for i in range(5)] + ["distribution_center"]:
    rl_policies[_agent] = ActorCritic(obs_dim=33, action_dim=2)
    rl_policies[_agent].eval()


class ForecastRequest(BaseModel):
    store_id: str
    product_id: str
    lookback_data: list


class InventoryRecommendation(BaseModel):
    store_id: str
    forecast_mu: list
    forecast_sigma: list
    recommended_order_qty: float
    recommended_realloc: float
    confidence_interval_80: dict


def prepare_input_from_request(request: ForecastRequest):
    sales_seq = torch.randn(1, 90, 20)
    sensor_seq = torch.randn(1, 90, 3)
    return sales_seq, sensor_seq


def build_rl_observation(store_id: str, mu: torch.Tensor, sigma: torch.Tensor) -> np.ndarray:
    obs = np.concatenate([
        [100.0 / 1000.0],
        mu.squeeze().numpy(),
        sigma.squeeze().numpy(),
        [0.0],
        [15.0 / 30.0],
        [0.15, 0.04],
    ]).astype(np.float32)
    return obs


@app.post("/forecast", response_model=dict)
def get_forecast(request: ForecastRequest):
    try:
        sales_seq, sensor_seq = prepare_input_from_request(request)
        with torch.no_grad():
            mu, log_var, _ = forecaster(sales_seq, sensor_seq)
            sigma = torch.exp(0.5 * log_var)
        return {
            "forecast_mu": mu.squeeze().tolist(),
            "forecast_sigma": sigma.squeeze().tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/recommend", response_model=InventoryRecommendation)
def get_recommendation(request: ForecastRequest):
    sales_seq, sensor_seq = prepare_input_from_request(request)
    with torch.no_grad():
        mu, log_var, _ = forecaster(sales_seq, sensor_seq)
        sigma = torch.exp(0.5 * log_var)
    obs = build_rl_observation(request.store_id, mu, sigma)
    policy = rl_policies[request.store_id]
    action, _, _ = policy.get_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    order_qty = (action[0][0].item() + 1) / 2 * 1000
    realloc = action[0][1].item() * 200
    return InventoryRecommendation(
        store_id=request.store_id,
        forecast_mu=mu.squeeze().tolist(),
        forecast_sigma=sigma.squeeze().tolist(),
        recommended_order_qty=round(order_qty, 1),
        recommended_realloc=round(realloc, 1),
        confidence_interval_80={
            "lower": (mu - 1.28 * sigma).squeeze().tolist(),
            "upper": (mu + 1.28 * sigma).squeeze().tolist(),
        },
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}