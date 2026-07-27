# Experiment Summary

| Name | Timestamp | Key Metrics | Notes |
|---|---|---|---|
| forecaster_baseline | 2026-07-09T19:35:55 | mape=18.4 | rmse=9.7 | val_loss=1.45 | ARIMA baseline |
| forecaster_transformer | 2026-07-09T19:35:55 | mape=12.1 | rmse=6.2 | val_loss=1.12 | Transformer tanpa cross-modal |
| forecaster_crossmodal | 2026-07-09T19:35:55 | mape=11.8 | rmse=5.9 | val_loss=1.08 | Full model dengan cross-modal attention |
| rl_mappo_joint | 2026-07-09T19:35:55 | avg_reward=-23241.0 | service_level=0.943 | total_cost=10045.0 | MAPPO joint training |
