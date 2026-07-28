import optuna
import torch

from src.forecasting.losses import GaussianNLLLoss
from src.forecasting.model import NeuroStockForecaster
from src.forecasting.train import train_forecaster

# ── 10.1.1 Objective Function ─────────────────────────────────────────────────

def objective(trial: optuna.Trial, train_loader, val_loader,
              n_sales_features, n_sensor_features, horizon: int = 14):
    d_model = trial.suggest_categorical("d_model", [32, 64, 128])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    n_layers = trial.suggest_int("n_layers", 2, 5)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    if d_model % n_heads != 0:
        raise optuna.TrialPruned()

    model = NeuroStockForecaster(
        n_sales_features=n_sales_features,
        n_sensor_features=n_sensor_features,
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        dropout=dropout, horizon=horizon
    )
    loss_fn = GaussianNLLLoss()
    _, best_val_loss = train_forecaster(
        model, train_loader, val_loader, loss_fn,
        n_epochs=30, lr=lr, patience=5,
        checkpoint_path=f"checkpoints/trial_{trial.number}.pt"
    )
    return best_val_loss


# ── 10.1.2 Run Search ─────────────────────────────────────────────────────────

def run_hyperparameter_search(train_loader, val_loader, n_sales_features,
                               n_sensor_features, n_trials: int = 50,
                               horizon: int = 14):
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        lambda trial: objective(trial, train_loader, val_loader,
                                n_sales_features, n_sensor_features, horizon),
        n_trials=n_trials
    )
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    return study


# ── 10.1.2 Analisis Hasil ─────────────────────────────────────────────────────

def analyze_study(study: optuna.Study):
    try:
        import optuna.visualization as vis
        fig1 = vis.plot_param_importances(study)
        fig1.write_html("results/hyperparam_importance.html")
        fig2 = vis.plot_optimization_history(study)
        fig2.write_html("results/optimization_history.html")
        fig3 = vis.plot_parallel_coordinate(study)
        fig3.write_html("results/parallel_coordinate.html")
        print("✅ Visualisasi tersimpan di results/")
    except Exception as e:
        print(f"⚠️ Visualisasi gagal: {e}")


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    import src.forecasting.tune as tune_module

    print("── Demo Optuna Search (3 trials) ──")

    n_sales, n_sensor, horizon = 10, 3, 7
    n_train, n_val = 40, 10

    train_dataset = TensorDataset(
        torch.randn(n_train, 30, n_sales),
        torch.randn(n_train, 30, n_sensor),
        torch.randn(n_train, horizon)
    )
    val_dataset = TensorDataset(
        torch.randn(n_val, 30, n_sales),
        torch.randn(n_val, 30, n_sensor),
        torch.randn(n_val, horizon)
    )
    train_loader = DataLoader(train_dataset, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_size=8)

    original_train = tune_module.train_forecaster

    def fast_train(model, train_loader, val_loader, loss_fn, **kwargs):
        loss_fn_obj = GaussianNLLLoss()
        val_loss = 0.0
        for batch in val_loader:
            s, sen, t = batch
            with torch.no_grad():
                mu, lv, _ = model(s, sen)
                val_loss = loss_fn_obj(mu, lv, t).item()
        return model, val_loss

    tune_module.train_forecaster = fast_train

    study = run_hyperparameter_search(
        train_loader, val_loader,
        n_sales_features=n_sales,
        n_sensor_features=n_sensor,
        n_trials=3,
        horizon=horizon
    )

    tune_module.train_forecaster = original_train
    print(f"\nBest params: {study.best_params}")
    print(f"Best val loss: {study.best_value:.4f}")