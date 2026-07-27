# NeuroStock 

> Joint Transformer Forecasting + Multi-Agent Reinforcement Learning untuk Optimasi Inventori Ritel

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Masalah

Toko ritel kehilangan margin tiap bulan karena dua masalah berlawanan: **overstock** (modal tertahan di gudang) dan **stockout** (kehilangan penjualan). Pendekatan konvensional — forecast dulu, lalu optimasi terpisah — tidak optimal karena keputusan inventori tidak memberi feedback ke model forecast.

---

## Pendekatan

---

## Hasil Utama

| Metode | MAPE (%) | Total Cost (relatif) | Service Level |
|---|---|---|---|
| ARIMA + (s,S) policy | 18.4 | 100% (baseline) | 87.2% |
| NeuroStock forecaster + (s,S) | 12.1 | 89% | 91.5% |
| NeuroStock joint (full) | 11.8 | 81% | 94.3% |

Perbedaan NeuroStock Joint vs ARIMA+(s,S): **signifikan secara statistik** (paired t-test, p=0.006).

---

## Quickstart

```bash
# 1. Setup environment
git clone https://github.com/USERNAME/neurostock.git
cd neurostock
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Download dataset
kaggle datasets download -d anirudhchauhan/retail-store-inventory-forecasting-dataset -p data/raw/

# 3. Jalankan pipeline data
python src/data/preprocessing.py
python src/data/sensor_simulation.py
python src/data/feature_engineering.py

# 4. Jalankan API
uvicorn api.main:app --reload --port 8000

# 5. Jalankan dashboard
streamlit run dashboard/app.py
```

---

## Struktur Project

---

## Keterbatasan & Rencana Lanjutan

**Keterbatasan (disengaja jujur):**
- Skala simulasi: 5 toko, 10 produk — bukan ratusan toko seperti enterprise
- Data sensor adalah **simulasi**, bukan IoT asli
- RL training dengan data simulasi demand, bukan data transaksi real-time
- Joint training end-to-end masih eksperimental (alternating training jadi metode utama)

**Rencana lanjutan:**
- Integrasi data sensor IoT asli (suhu gudang, RFID scanner)
- Skala ke lebih banyak toko dengan parameter sharing antar agent
- Retraining otomatis dengan data baru (online learning)
- Deployment ke Kubernetes untuk skalabilitas

---

## Rujukan Akademik

- Schulman et al. (2017) — Proximal Policy Optimization
- Vaswani et al. (2017) — Attention Is All You Need
- Lowe et al. (2017) — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
- Silver et al. (2016) — Mastering the Game of Go with Deep Neural Networks

---

## Cara Berkontribusi

Pull request dan issue sangat disambut. Lihat `docs/` untuk dokumentasi teknis lengkap.

