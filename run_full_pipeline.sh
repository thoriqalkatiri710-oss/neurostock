#!/bin/bash
set -e

echo "=== Step 1: Data preprocessing ==="
python src/data/preprocessing.py

echo "=== Step 2: Sensor simulation ==="
python src/data/sensor_simulation.py

echo "=== Step 3: Feature engineering ==="
python src/data/feature_engineering.py

echo "=== Step 4: Sanity check model ==="
python -m src.forecasting.sanity_check

echo "=== Step 5: RL sanity check ==="
python -m src.rl.rl_sanity_check

echo "=== Step 6: Evaluation ==="
python -m src.optimization.evaluation
python -m src.optimization.statistical_tests

echo "=== Pipeline selesai. Lihat results/ untuk output lengkap. ==="