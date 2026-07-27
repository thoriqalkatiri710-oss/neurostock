import pytest


@pytest.mark.slow
def test_full_training_cycle_small_scale():
    """
    Test training penuh skala sangat kecil.
    Hanya dijalankan manual atau nightly — bukan di setiap push CI.
    Jalankan dengan: pytest tests/test_integration_slow.py -v -m slow
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.forecasting.model import NeuroStockForecaster
    from src.forecasting.losses import GaussianNLLLoss
    from src.forecasting.train import train_forecaster

    model = NeuroStockForecaster(
        n_sales_features=5, n_sensor_features=2,
        d_model=8, n_heads=2, n_layers=1, horizon=3
    )
    loss_fn = GaussianNLLLoss()

    dataset = TensorDataset(
        torch.randn(20, 15, 5),
        torch.randn(20, 15, 2),
        torch.randn(20, 3)
    )
    loader = DataLoader(dataset, batch_size=4)

    model, best_loss = train_forecaster(
        model, loader, loader, loss_fn,
        n_epochs=3, patience=2,
        checkpoint_path="checkpoints/slow_test.pt"
    )
    assert best_loss < 10.0
    print(f"✅ Slow test passed: best_loss={best_loss:.4f}")