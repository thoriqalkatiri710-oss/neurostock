# Design Decisions — NeuroStock

Dokumen ini merekam keputusan desain non-trivial beserta alasannya.
Berguna saat menjawab pertanyaan interview berbulan-bulan setelah project selesai.

---

## 1. Kenapa Alternating Training, bukan End-to-End?

**Keputusan:** Gunakan alternating training (Bagian 5.2) sebagai metode utama.

**Alasan:** End-to-end differentiable training secara teori lebih elegan, tapi sangat
tidak stabil di awal karena dua loss (forecasting NLL dan RL policy gradient) punya
skala dan dinamika yang sangat berbeda. Alternating training memberi keduanya waktu
untuk "matang" sendiri sebelum saling beradaptasi.

**Trade-off:** Alternating lebih lambat konvergen, tapi jauh lebih stabil dan
lebih mudah di-debug karena setiap fase bisa dipantau secara terpisah.

---

## 2. Kenapa Gaussian NLL, bukan MSE?

**Keputusan:** Gunakan Gaussian NLL loss (Bagian 3.5) untuk forecasting.

**Alasan:** MSE hanya mengoptimasi akurasi titik (point forecast). Gaussian NLL
memaksa model untuk juga mengkalibrasi uncertainty (σ) — model yang terlalu
confident pada prediksi yang salah akan mendapat penalti lebih besar.

**Implikasi:** Output σ langsung dipakai oleh RL agent sebagai informasi
ketidakpastian, mempengaruhi keputusan margin keamanan inventori.

---

## 3. Kenapa Bobot Reward w = {holding: 0.3, stockout: 0.4, service: 0.2, transport: 0.1}?

**Keputusan:** Bobot stockout lebih tinggi dari holding (Bagian 4.4).

**Alasan:** Dalam konteks ritel, biaya kehilangan penjualan (stockout) umumnya
lebih mahal dari biaya penyimpanan (holding), karena melibatkan kehilangan
kepercayaan pelanggan di luar biaya langsung. Bobot ini adalah starting point —
harus di-tune sesuai data biaya aktual bisnis.

**Catatan:** Ini hyperparameter bisnis, bukan hyperparameter teknis. Perubahan
bobot harus melibatkan domain expert, bukan hanya data scientist.

---

## 4. Kenapa Sinusoidal Positional Encoding, bukan Learned?

**Keputusan:** Gunakan sinusoidal PE (Bagian 3.2).

**Alasan:** Sinusoidal PE bisa generalisasi ke panjang sequence yang belum pernah
dilihat saat training. Learned PE terbatas pada panjang sequence yang ada di
training data — masalah jika di production sequence lebih panjang/pendek.

---

## 5. Kenapa CTDE (Centralized Training, Decentralized Execution)?

**Keputusan:** Gunakan paradigma CTDE untuk multi-agent RL (Bagian 1.6.2).

**Alasan:** Fully centralized (satu policy untuk semua agent) tidak scalable —
joint action space tumbuh eksponensial. Fully decentralized (tanpa koordinasi)
mengabaikan dependensi antar toko. CTDE adalah middle ground: training
memanfaatkan informasi global, eksekusi tetap lokal per toko.

---

## 6. Kenapa IQR, bukan Z-score untuk Outlier Detection?

**Keputusan:** Gunakan IQR per kelompok (store, product) di Bagian 2.2.2.

**Alasan:** Z-score mengasumsikan distribusi normal dan sensitif terhadap outlier
itu sendiri (outlier ekstrem menggeser mean dan std). IQR lebih robust karena
berbasis median dan quartile yang tidak terpengaruh outlier. Per-group karena
skala demand antar produk bisa berbeda jauh.

---

## 7. Kenapa Walk-Forward Validation, bukan K-Fold Biasa?

**Keputusan:** Gunakan walk-forward splits (Bagian 2.5).

**Alasan:** K-fold standard memungkinkan data masa depan masuk ke training set —
data leakage temporal yang membuat model terlihat lebih baik dari kenyataan.
Walk-forward menjamin train selalu sebelum test secara temporal, sesuai deployment
nyata di mana prediksi selalu tentang masa depan.