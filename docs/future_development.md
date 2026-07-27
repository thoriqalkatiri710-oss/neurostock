# Rencana Pengembangan Lanjutan NeuroStock

## Ringkasan

| Fase | Estimasi Effort | Prioritas |
|---|---|---|
| Integrasi sensor IoT asli | 4-6 minggu | Tinggi |
| Computer vision shelf monitoring | 6-8 minggu | Sedang |
| Federated learning lintas toko | 8-10 minggu | Rendah |
| Eksplorasi offline RL | 3-4 minggu | Tinggi |

---

## 15.1 Integrasi Sensor IoT Asli

**Status saat ini:** Data sensor simulasi (Bagian 2.3)  
**Target:** Query dari InfluxDB yang menerima streaming data sensor fisik via MQTT

**Jalur migrasi:**
- `SensorDataSource` ABC sudah ada di `src/data/sensor_interface.py`
- Implementasi `IoTSensorSource.get_readings()` dengan koneksi MQTT
- Tidak memerlukan perombakan pipeline — hanya ganti `source_type="iot"`

---

## 15.2 Computer Vision Shelf Monitoring

**Konsep:** Kamera rak toko → encoder CNN/ViT → modalitas ketiga di cross-modal attention

**Arsitektur:**

**Catatan:** Di luar scope 3 bulan, tapi `CrossModalFusion` sudah dirancang extensible.

---

## 15.3 Federated Learning Lintas Toko

**Motivasi:** Data penjualan franchise berbeda tidak bisa dipusatkan (privasi/kontrak bisnis)

**Pendekatan:**
- Setiap toko latih model lokal
- Hanya gradient/parameter update yang dibagikan ke server pusat
- Framework: `Flower` (flwr) atau `PySyft`

---

## 15.4 Offline RL untuk Production Safety

**Masalah PPO online di produksi:** Eksplorasi langsung = risiko kerugian finansial nyata

**Solusi:** Conservative Q-Learning (CQL) atau Implicit Q-Learning (IQL)
- Dilatih dari data historis keputusan inventori manusia
- Tidak perlu eksplorasi langsung di lingkungan nyata
- Lebih aman untuk deployment pertama

---

## Gap Simulasi vs Production

| Aspek | Saat Ini (Simulasi) | Production Nyata |
|---|---|---|
| Data sensor | Simulasi korelasi | IoT fisik (MQTT/InfluxDB) |
| Demand model | Random + seasonal | Data transaksi real-time |
| RL exploration | Bebas (simulasi) | Offline RL atau constrained |
| Skala | 5 toko, 10 produk | Ratusan toko |
| Retraining | Manual | Otomatis (scheduled) |