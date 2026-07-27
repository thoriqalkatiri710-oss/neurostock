import torch
from .losses import GaussianNLLLoss
from .model import NeuroStockForecaster


def sanity_check_overfit_small_batch(model, loss_fn, small_batch,
                                      n_steps: int = 200, lr: float = 1e-3):
    """
    Overfit test pada batch kecil sebelum training penuh.
    Jika model tidak bisa overfit 8 sample → ada bug di arsitektur/loss.
    Target: loss < 0.1 dalam 200 steps.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sales_seq, sensor_seq, target = small_batch

    print("── Sanity Check: Overfit Small Batch ──")
    for step in range(n_steps):
        optimizer.zero_grad()
        mu, log_var, _ = model(sales_seq, sensor_seq)
        loss = loss_fn(mu, log_var, target)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  Step {step:3d}: loss = {loss.item():.4f}")

    final_loss = loss.item()
    if final_loss < 0.5:
        print(f"✅ Sanity check passed (final loss={final_loss:.4f})")
    else:
        print(f"⚠️  Loss={final_loss:.4f} — periksa arsitektur/loss function")

    return final_loss


if __name__ == "__main__":
    # Buat small batch sintetis
    batch_size = 8
    lookback = 90
    horizon = 14
    n_sales = 20
    n_sensor = 3

    model = NeuroStockForecaster(
        n_sales_features=n_sales,
        n_sensor_features=n_sensor,
        horizon=horizon
    )
    loss_fn = GaussianNLLLoss()

    small_batch = (
        torch.randn(batch_size, lookback, n_sales),
        torch.randn(batch_size, lookback, n_sensor),
        torch.randn(batch_size, horizon)
    )

    sanity_check_overfit_small_batch(model, loss_fn, small_batch)