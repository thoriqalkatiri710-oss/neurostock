"""
BAGIAN 11 — CONTOH NUMERIK END-TO-END
Menelusuri satu siklus penuh sistem dengan angka konkret.
store_0, hari t.
"""

import numpy as np
import torch
import math


def section_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 11.1 SETUP SKENARIO
# ─────────────────────────────────────────────────────────────
section_header("11.1 Setup Skenario")

inventory_t = 120.0
sales_history_7d = [45, 52, 48, 61, 55, 70, 58]
temperature_today = 31.0
day_of_week = "Jumat"
lead_time = 3
in_transit_qty = 40.0
in_transit_arrival = 2  # hari lagi

print(f"Inventori saat ini (I_t)       : {inventory_t} unit")
print(f"Histori penjualan 7 hari       : {sales_history_7d}")
print(f"Rata-rata 7 hari               : {np.mean(sales_history_7d):.1f} unit")
print(f"Suhu hari ini                  : {temperature_today}°C (normal: 25°C)")
print(f"Hari                           : {day_of_week} (mendekati weekend)")
print(f"Lead time                      : {lead_time} hari")
print(f"Order in-transit               : {in_transit_qty} unit (tiba {in_transit_arrival} hari lagi)")


# ─────────────────────────────────────────────────────────────
# 11.2 TAHAP FORECASTING
# ─────────────────────────────────────────────────────────────
section_header("11.2 Tahap Forecasting")

# 11.2.1 Forward pass hasil
mu = 62.4
log_var = 1.8

var = math.exp(log_var)
sigma = math.sqrt(var)

z_80 = 1.28  # z-score untuk interval 80%
lower_80 = mu - z_80 * sigma
upper_80 = mu + z_80 * sigma

print(f"\n── 11.2.1 Output Transformer ──")
print(f"μ (forecast mean)              : {mu} unit")
print(f"log σ²                         : {log_var}")
print(f"σ² = exp({log_var})            : {var:.4f}")
print(f"σ  = √{var:.4f}               : {sigma:.4f}")
print(f"\nInterval prediksi 80%:")
print(f"Lower = {mu} - 1.28×{sigma:.2f} = {lower_80:.2f} unit")
print(f"Upper = {mu} + 1.28×{sigma:.2f} = {upper_80:.2f} unit")
print(f"\nInterpretasi: demand besok diperkirakan {mu} unit")
print(f"80% keyakinan: antara {lower_80:.2f} - {upper_80:.2f} unit")

# 11.2.2 Cross-modal contribution
print(f"\n── 11.2.2 Kontribusi Cross-Modal Attention ──")
mu_tanpa_sensor = np.mean(sales_history_7d) + 1.5  # tren naik ringan
mu_dengan_sensor = mu

koreksi = mu_dengan_sensor - mu_tanpa_sensor
koreksi_pct = (koreksi / mu_tanpa_sensor) * 100

print(f"μ tanpa sensor (histori saja)  : {mu_tanpa_sensor:.1f} unit")
print(f"μ dengan sensor (cross-modal)  : {mu_dengan_sensor:.1f} unit")
print(f"Koreksi dari sinyal suhu+hari  : +{koreksi:.1f} unit ({koreksi_pct:.1f}%)")
print(f"\nSuhu {temperature_today}°C (di atas normal) + {day_of_week}")
print(f"→ attention weight tinggi pada sinyal sensor")
print(f"→ forecast dikoreksi naik {koreksi:.1f} unit")


# ─────────────────────────────────────────────────────────────
# 11.3 TAHAP RL DECISION
# ─────────────────────────────────────────────────────────────
section_header("11.3 Tahap RL Decision")

# Bangun observasi
obs = np.array([
    inventory_t / 1000.0,         # inventory normalized
    mu / 1000.0,                   # forecast mu (1 hari, simplified)
    sigma / 1000.0,                # forecast sigma
    1.0,                           # promo_flag (Jumat = promo)
    2.0 / 30.0,                    # days_to_holiday normalized
    0.15,                          # dc_signal mean
    0.04,                          # dc_signal std
], dtype=np.float32)

print(f"\n── Observasi Agent store_0 ──")
labels = ["inventory/max", "mu/max", "sigma/max", "promo_flag",
          "days_to_holiday", "dc_mean", "dc_std"]
for label, val in zip(labels, obs):
    print(f"  {label:<20}: {val:.4f}")

# Decode action (simulasi policy output)
raw_action = np.array([0.32, -0.15])  # contoh output policy
order_qty = (raw_action[0] + 1) / 2 * 1000
realloc = raw_action[1] * (1000 * 0.2)

print(f"\n── Output Policy (raw action) ──")
print(f"raw_action                     : {raw_action}")
print(f"order_qty = ({raw_action[0]}+1)/2 × 1000 = {order_qty:.1f} unit")
print(f"realloc   = {raw_action[1]} × 200         = {realloc:.1f} unit")


# ─────────────────────────────────────────────────────────────
# 11.4 INVENTORY UPDATE
# ─────────────────────────────────────────────────────────────
section_header("11.4 Inventory Dynamics")

demand_actual = 64.0  # misalkan demand aktual hari ini
arriving_today = 0.0  # order in-transit tiba dalam 2 hari, belum tiba

inventory_after = inventory_t - demand_actual + arriving_today + realloc
inventory_after_order = inventory_after  # order baru tiba setelah lead_time

print(f"\nDemand aktual hari ini         : {demand_actual} unit")
print(f"Order tiba hari ini            : {arriving_today} unit")
print(f"Realokasi dari DC              : {realloc:.1f} unit")
print(f"\nInventory after = {inventory_t} - {demand_actual} + {arriving_today} + {realloc:.1f}")
print(f"                = {inventory_after:.1f} unit")
print(f"Order baru ({order_qty:.0f} unit) tiba dalam {lead_time} hari")


# ─────────────────────────────────────────────────────────────
# 11.5 REWARD COMPUTATION
# ─────────────────────────────────────────────────────────────
section_header("11.5 Reward Computation")

holding_cost = 0.5
stockout_penalty = 2.0
transport_cost = 0.1

r_holding = -holding_cost * max(inventory_after, 0)
r_stockout = -stockout_penalty * max(-inventory_after, 0)
r_service = min(demand_actual, max(inventory_after + demand_actual, 0)) / (demand_actual + 1e-6)
r_transport = -transport_cost * abs(realloc)

w = {"holding": 0.3, "stockout": 0.4, "service": 0.2, "transport": 0.1}
total_reward = (
    w["holding"] * r_holding +
    w["stockout"] * r_stockout +
    w["service"] * r_service * 10 +
    w["transport"] * r_transport
)

print(f"\nR_holding  = -{holding_cost} × max({inventory_after:.1f}, 0) = {r_holding:.2f}")
print(f"R_stockout = -{stockout_penalty} × max(-{inventory_after:.1f}, 0) = {r_stockout:.2f}")
print(f"R_service  = min({demand_actual}, max({inventory_after:.1f}+{demand_actual}, 0)) / {demand_actual} = {r_service:.4f}")
print(f"R_transport= -{transport_cost} × |{realloc:.1f}| = {r_transport:.2f}")
print(f"\nTotal reward = {w['holding']}×{r_holding:.2f} + {w['stockout']}×{r_stockout:.2f} "
      f"+ {w['service']}×{r_service:.4f}×10 + {w['transport']}×{r_transport:.2f}")
print(f"            = {total_reward:.4f}")


# ─────────────────────────────────────────────────────────────
# 11.6 RINGKASAN
# ─────────────────────────────────────────────────────────────
section_header("11.6 Ringkasan Siklus Penuh")

print(f"""
Siklus hari t untuk store_0:

1. INPUT    : Inventori={inventory_t}, histori=[{sales_history_7d}]
              Suhu={temperature_today}°C, {day_of_week}

2. FORECAST : μ={mu} unit (σ={sigma:.2f})
              Koreksi cross-modal: +{koreksi:.1f} unit dari sinyal suhu
              Interval 80%: [{lower_80:.2f}, {upper_80:.2f}]

3. DECISION : Order {order_qty:.0f} unit (tiba {lead_time} hari lagi)
              Realokasi: {realloc:.1f} unit

4. OUTCOME  : Demand aktual={demand_actual}, Inventori after={inventory_after:.1f}
              Reward={total_reward:.4f}

5. LEARNING : Experience (obs, action, reward, next_obs) masuk ke buffer PPO
              → update policy di akhir episode
""")