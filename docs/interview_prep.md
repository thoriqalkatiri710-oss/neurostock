# Persiapan Interview — NeuroStock

## 1. "Ceritakan tentang project ini" (2-3 menit)
Masalah → Pendekatan → Hasil → Dampak

Masalah: overstock dan stockout di ritel, pipeline forecast-optimasi
terpisah tidak optimal.
Pendekatan: joint training transformer forecaster + multi-agent PPO,
sehingga keduanya saling mengoreksi.
Hasil: MAPE 11.8% vs 18.4% ARIMA, total cost turun 19%, service level 94.3%.
Dampak: sistem bisa memberikan rekomendasi order + realokasi antar toko
secara otomatis.

## 2. "Kenapa joint, bukan forecast-lalu-optimasi biasa?"
Pipeline terpisah: forecast dianggap fixed truth, lalu policy dioptimasi
di atasnya. Masalah: forecast error langsung menjadi decision error tanpa
koreksi. Joint training: policy agent memberi sinyal reward yang
mengoreksi forecast secara implisit. Ablation study membuktikan joint
menurunkan total cost 8% vs pipeline terpisah.

## 3. "Bagian paling menantang?"
Multi-agent RL tidak konvergen di awal. Reward makin negatif tiap episode.
Diagnosis bertahap: cek reward scale, cek advantage normalisasi, cek
gradient norm. Fix: clip_grad_norm=0.5, normalisasi advantage, entropy
bonus untuk eksplorasi lebih lama.

## 4. "Bagaimana kamu tahu model bekerja dengan benar?"
- Sanity check overfit batch kecil (loss turun dari 1.45 ke -1.19)
- Walk-forward validation 5 fold (tidak ada overlap train/test)
- Paired t-test signifikan (p=0.006) vs baseline
- Coverage probability prediction interval diperiksa

## 5. "Apa keterbatasan pendekatan ini?"
- Skala 5 toko, bukan enterprise (ratusan toko)
- Data sensor simulasi, bukan IoT asli
- Joint training end-to-end masih eksperimental
- Alternating training jadi metode utama yang dilaporkan

## 6. "Bagaimana scale ke produksi?"
- Data pipeline real-time (Kafka/Flink untuk stream transaksi)
- Parameter sharing antar agent untuk skalabilitas
- Retraining terjadwal (weekly) dengan data baru
- Monitoring drift dengan statistical process control
- A/B testing sebelum full rollout