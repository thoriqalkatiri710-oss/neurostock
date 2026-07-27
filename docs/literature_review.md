# Tinjauan Literatur — NeuroStock

## 16.1 Evolusi Demand Forecasting

### Generasi 1 — Statistik Klasik (1970-an–2000-an)
**Metode:** ARIMA, Exponential Smoothing, Holt-Winters

| | |
|---|---|
| **Kelebihan** | Interpretable, butuh sedikit data, solusi closed-form |
| **Keterbatasan** | Asumsi linearitas, sulit menangkap pola kompleks multi-variat |
| **Kapan masih relevan** | Data dengan pola musiman sederhana, dataset kecil |

### Generasi 2 — Machine Learning Klasik (2000-an–2015)
**Metode:** Random Forest, XGBoost, LightGBM

| | |
|---|---|
| **Kelebihan** | Menangkap non-linearitas, feature engineering fleksibel |
| **Keterbatasan** | Tidak secara native menangani struktur sequential/temporal |

### Generasi 3 — Deep Learning Sequential (2015–2020)
**Metode:** RNN, LSTM, GRU

| | |
|---|---|
| **Kelebihan** | Menangkap dependency temporal secara native |
| **Keterbatasan** | Sulit paralelisasi, vanishing gradient pada sequence panjang |

### Generasi 4 — Transformer-based (2020–sekarang)
**Metode:** Temporal Fusion Transformer, Informer, PatchTST

| | |
|---|---|
| **Kelebihan** | Paralelisasi penuh, attention menangkap dependency jarak jauh |
| **Posisi NeuroStock** | Generasi ini + cross-modal fusion (sensor data) |

> **Catatan penting:** Jangan mengklaim NeuroStock "lebih baik dari segalanya" secara
> mutlak. ARIMA tetap kompetitif untuk data dengan pola musiman sederhana dan dataset
> kecil. Nuansa ini penting ditunjukkan ke evaluator.

---

## 16.2 Evolusi Inventory Optimization

### Generasi 1 — Operations Research Klasik
**Metode:** EOQ, kebijakan (s,S), Newsvendor model

| | |
|---|---|
| **Kelebihan** | Dasar matematis kuat, solusi closed-form di kasus sederhana |
| **Keterbatasan** | Asumsi distribusi demand yang sering terlalu sederhana |

### Generasi 2 — Stochastic Programming & Robust Optimization
**Metode:** Scenario-based programming, robust optimization

| | |
|---|---|
| **Kelebihan** | Menangani uncertainty lebih baik dari EOQ klasik |
| **Keterbatasan** | Computationally expensive untuk problem skala besar |

### Generasi 3 — Reinforcement Learning
**Metode:** DQN, PPO, MADDPG untuk inventory management

| | |
|---|---|
| **Kelebihan** | Belajar kebijakan langsung dari interaksi, menangani sistem kompleks multi-toko |
| **Keterbatasan** | Butuh banyak data/simulasi, less interpretable dibanding (s,S) klasik |

---

## 16.3 Posisi NeuroStock & Kontribusi Konseptual

### Kelemahan Struktural Pipeline Konvensional
Sebagian besar sistem yang ada memperlakukan forecasting dan inventory optimization
sebagai **dua modul terpisah** yang dihubungkan secara loose:

**Masalah:** Optimizer tidak bisa memberi sinyal balik ke forecaster tentang kesalahan
forecasting mana yang sebenarnya paling merugikan secara bisnis.

### Keunggulan Pendekatan Joint (NeuroStock)
**Implikasi konkret:** Forecaster bisa "belajar" bahwa:
- Kesalahan forecast pada produk **margin tinggi** lebih costly
- Kesalahan forecast pada produk **lead time panjang** lebih merugikan
- Dibanding produk yang mudah di-reorder cepat

Sinyal ini tidak bisa tertangkap jika forecaster dilatih independen dari keputusan inventori.

---

## 16.4 Paper Terkait untuk Eksplorasi Lanjutan

| Paper/Area | Relevansi untuk NeuroStock |
|---|---|
| Temporal Fusion Transformer (Lim et al.) | Arsitektur transformer multi-horizon dengan interpretability built-in |
| Deep Inventory Management dengan RL | Perbandingan RL vs kebijakan klasik di single/multi-echelon |
| MADDPG, QMIX | Varian multi-agent RL lain sebagai perbandingan tambahan |
| Conformal Prediction untuk time series | Kalibrasi uncertainty lebih rigorous dari asumsi Gaussian murni |

> **Catatan:** Area di atas terdaftar sebagai **Future Work**, bukan bagian yang sudah
> diimplementasikan, kecuali benar-benar dieksplorasi.

---

## Perbandingan Pendekatan di NeuroStock

| Komponen | Baseline | NeuroStock |
|---|---|---|
| Forecasting | ARIMA | Transformer + Cross-Modal Attention |
| Uncertainty | Confidence interval ARIMA | Gaussian NLL (μ, σ) per timestep |
| Inventory policy | (s,S) klasik | Multi-Agent PPO |
| Koordinasi antar toko | Tidak ada | Distribution Center Agent (CTDE) |
| Training | Terpisah | Joint (alternating) |

---

## Rujukan Akademik

1. Vaswani et al. (2017) — *Attention Is All You Need*
2. Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
3. Lowe et al. (2017) — *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*
4. Lim et al. (2021) — *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*
5. Silver et al. (2016) — *Mastering the Game of Go with Deep Neural Networks and Tree Search*