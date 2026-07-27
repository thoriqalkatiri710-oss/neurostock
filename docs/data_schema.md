# Data Schema — retail_store_inventory.csv

**Source:** Kaggle — Retail Store Inventory Forecasting Dataset  
**Shape:** 73.100 rows × 15 columns  
**Period:** 2022-01-01 onwards

| Column | Type | Description |
|---|---|---|
| Date | str | Tanggal transaksi (YYYY-MM-DD) |
| Store ID | str | ID unik toko |
| Product ID | str | ID unik produk |
| Category | str | Kategori produk |
| Region | str | Wilayah toko |
| Inventory Level | int64 | Stok tersedia saat ini |
| Units Sold | int64 | Unit terjual pada hari tersebut |
| Units Ordered | int64 | Unit yang dipesan ke supplier |
| Demand Forecast | float64 | Prediksi permintaan |
| Price | float64 | Harga jual produk |
| Discount | int64 | Diskon dalam persen |
| Weather Condition | str | Kondisi cuaca (Sunny, Rainy, dll) |
| Holiday/Promotion | int64 | Flag hari libur/promosi (0/1) |
| Competitor Pricing | float64 | Harga kompetitor |
| Seasonality | str | Musim (Autumn, Summer, dll) |