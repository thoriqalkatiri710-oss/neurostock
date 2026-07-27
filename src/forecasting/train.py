import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

logger = logging.getLogger(__name__)


def train_forecaster(model, train_loader: DataLoader, val_loader: DataLoader,
                     loss_fn, n_epochs: int = 100, lr: float = 1e-4,
                     patience: int = 10,
                     checkpoint_path: str = "checkpoints/forecaster_best.pt"):
    """
    Training loop dengan:
    - clip_grad_norm_: mencegah exploding gradient
    - ReduceLROnPlateau: turunkan LR otomatis saat val loss stagnan
    - Early stopping: mencegah overfitting tanpa tebak jumlah epoch
    """
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        train_losses = []
        for batch in train_loader:
            sales_seq, sensor_seq, target = batch
            optimizer.zero_grad()
            mu, log_var, _ = model(sales_seq, sensor_seq)
            loss = loss_fn(mu, log_var, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                sales_seq, sensor_seq, target = batch
                mu, log_var, _ = model(sales_seq, sensor_seq)
                val_loss = loss_fn(mu, log_var, target)
                val_losses.append(val_loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)
        scheduler.step(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # ── Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ Model tersimpan (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping di epoch {epoch+1}")
                break

    return model, best_val_loss