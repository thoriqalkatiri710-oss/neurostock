import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="NeuroStock Dashboard", layout="wide")
st.title("NeuroStock — Demand Forecasting & Inventory Optimization")

API_URL = "http://localhost:8000"


def load_lookback(store_id: str, product_id: str) -> list:
    """Placeholder — di production ambil dari database atau CSV."""
    return [{"date": f"2026-01-{i+1:02d}", "units_sold": int(np.random.randint(20, 80))}
            for i in range(90)]


with st.sidebar:
    store_id = st.selectbox("Pilih toko", [f"store_{i}" for i in range(5)])
    product_id = st.selectbox("Pilih produk", [f"product_{i}" for i in range(10)])
    show_baseline = st.checkbox("Tampilkan baseline (s,S) policy", value=True)

if st.button("Generate rekomendasi"):
    with st.spinner("Menghitung forecast dan rekomendasi..."):
        try:
            payload = {
                "store_id": store_id,
                "product_id": product_id,
                "lookback_data": load_lookback(store_id, product_id)
            }
            result = requests.post(f"{API_URL}/recommend", json=payload).json()

            # ── Metrics ──
            col1, col2, col3 = st.columns(3)
            col1.metric("Rekomendasi order", f"{result['recommended_order_qty']:.0f} unit")
            col2.metric("Rekomendasi realokasi", f"{result['recommended_realloc']:.0f} unit")
            col3.metric("Forecast horizon", f"{len(result['forecast_mu'])} hari")

            # ── Forecast Chart ──
            horizon = list(range(1, len(result["forecast_mu"]) + 1))
            lower = result["confidence_interval_80"]["lower"]
            upper = result["confidence_interval_80"]["upper"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=horizon, y=result["forecast_mu"],
                name="Forecast", line=dict(color="#1D9E75", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=horizon + horizon[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(29,158,117,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Interval 80%"
            ))

            if show_baseline:
                baseline_forecast = [np.mean(result["forecast_mu"])] * len(horizon)
                fig.add_trace(go.Scatter(
                    x=horizon, y=baseline_forecast,
                    name="Baseline (s,S)", line=dict(color="#E74C3C", dash="dash")
                ))

            fig.update_layout(
                title=f"Forecast Demand — {store_id} | {product_id}",
                xaxis_title="Hari ke depan",
                yaxis_title="Unit",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Tabel Detail ──
            df = pd.DataFrame({
                "Hari": horizon,
                "Forecast (mu)": [round(v, 2) for v in result["forecast_mu"]],
                "Sigma": [round(v, 2) for v in result["forecast_sigma"]],
                "Lower 80%": [round(v, 2) for v in lower],
                "Upper 80%": [round(v, 2) for v in upper],
            })
            st.dataframe(df, use_container_width=True)

            # ── What-if Simulation ──
            with st.expander("Lihat detail simulasi what-if"):
                demand_multiplier = st.slider("Simulasikan kenaikan demand (%)", -50, 100, 0)
                adjusted_order = result["recommended_order_qty"] * (1 + demand_multiplier / 100)
                st.write(f"Dengan kenaikan demand {demand_multiplier}%, "
                         f"estimasi rekomendasi order menjadi **{adjusted_order:.0f} unit**")

        except Exception as e:
            st.error(f"Error: {e}. Pastikan API berjalan di {API_URL}")
            st.info("Jalankan: `uvicorn api.main:app --reload --port 8000`")