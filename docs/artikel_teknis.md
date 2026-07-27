# Kenapa Saya Membangun NeuroStock: Joint Forecasting + RL untuk Inventori Ritel

## Hook
Toko ritel kehilangan jutaan rupiah tiap bulan karena overstock dan stockout.
Kenapa pendekatan forecasting konvensional tidak cukup?

## Pendekatan Konvensional & Keterbatasannya
Pipeline standar: forecast demand → hitung safety stock → order.
Masalah: forecast tidak tahu dampak keputusan inventori, dan
keputusan inventori tidak memberi feedback ke forecast.

## Insight Kunci
Joint training: biarkan forecaster dan policy agent saling mengoreksi
lewat satu proses pembelajaran yang terhubung.

## Implementasi (High-Level)
- Transformer dengan cross-modal attention (sales + sensor simulasi)
- Multi-agent PPO: 5 store agents + 1 distribution center
- Alternating training: forecaster dan RL bergantian dilatih

## Hasil & Angka
- MAPE turun dari 18.4% (ARIMA) ke 11.8% (NeuroStock joint)
- Total cost turun 19% vs baseline
- Service level naik dari 87.2% ke 94.3%

## Pembelajaran Personal
Tantangan terbesar: multi-agent RL awalnya tidak konvergen.
Reward makin negatif tiap episode. Diagnosis: reward scale terlalu besar
relatif terhadap value function. Fix: normalisasi advantage + clip gradient.
Pembelajaran: debugging RL butuh kesabaran dan sanity check bertahap
(single agent dulu, baru multi-agent).

## Call to Action
GitHub: https://github.com/USERNAME/neurostock
Demo: http://localhost:8501