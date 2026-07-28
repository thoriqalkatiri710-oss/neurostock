import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── 2.4.1 Lag Features ────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_col: str, lags: list,
                     group_cols: list) -> pd.DataFrame:
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


# ── 2.4.2 Rolling Statistics ──────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list,
                         group_cols: list) -> pd.DataFrame:
    """shift(1) sebelum rolling — mencegah data leakage dari hari yang diprediksi."""
    for w in windows:
        grp = df.groupby(group_cols)[target_col]
        df[f"{target_col}_roll_mean_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w).mean())
        df[f"{target_col}_roll_std_{w}"] = grp.transform(lambda x: x.shift(1).rolling(w).std())
    return df


# ── 2.4.3 Calendar Features ───────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Representasi siklis sin/cos mencegah model menganggap Des-Jan sebagai jauh."""
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ── 2.4.4 Normalisasi per Group ───────────────────────────────────────────────

def fit_scalers_per_group(df: pd.DataFrame, target_col: str,
                          group_cols: list) -> dict:
    """Fit scaler per (store, product) — hanya pada data training."""
    scalers = {}
    for key, group in df.groupby(group_cols):
        scaler = StandardScaler()
        scaler.fit(group[[target_col]])
        scalers[key] = scaler
    return scalers


def apply_scalers(df: pd.DataFrame, target_col: str,
                  group_cols: list, scalers: dict) -> pd.DataFrame:
    """Transform saja — jangan fit ulang pada val/test (mencegah data leakage)."""
    scaled_col = f"{target_col}_scaled"
    df[scaled_col] = np.nan
    for key, group in df.groupby(group_cols):
        if key in scalers:
            idx = group.index
            df.loc[idx, scaled_col] = scalers[key].transform(group[[target_col]]).flatten()
    return df


def save_scalers(scalers: dict, path: str = "checkpoints/scalers.pkl"):
    with open(path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved: {path}")


def load_scalers(path: str = "checkpoints/scalers.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 2.5 Walk-Forward Split ────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, date_col: str,
                        n_splits: int, horizon: int) -> list:
    """
    Walk-forward cross-validation untuk time series.
    Setiap fold: train tumbuh, test = horizon hari ke depan.
    """
    dates = sorted(df[date_col].unique())
    total_len = len(dates)
    fold_size = (total_len - horizon) // n_splits

    splits = []
    for i in range(1, n_splits + 1):
        train_end_idx = fold_size * i
        test_end_idx = min(train_end_idx + horizon, total_len)
        train_dates = dates[:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        splits.append((train_dates, test_dates))
    return splits


def save_splits(splits: list, path: str = "checkpoints/walk_forward_splits.pkl"):
    """Simpan splits ke disk — konsistensi antar eksperimen."""
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Splits saved: {path}")


def load_splits(path: str = "checkpoints/walk_forward_splits.pkl") -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_splits(splits: list):
    """Pastikan tidak ada overlap antara train dan test di setiap fold."""
    print("\n── Walk-Forward Split Summary ──")
    for i, (train_dates, test_dates) in enumerate(splits):
        train_set = set(str(d) for d in train_dates)
        test_set = set(str(d) for d in test_dates)
        overlap = train_set & test_set
        status = "✅" if len(overlap) == 0 else f"❌ overlap: {len(overlap)}"
        print(f"Fold {i+1}: train={len(train_dates)} days | "
              f"test={len(test_dates)} days | {status}")


# ── Pipeline Utama ────────────────────────────────────────────────────────────

def run_feature_pipeline(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv, parse_dates=["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    print(f"Loaded: {df.shape}")

    df = add_lag_features(df, "units_sold", lags=[1, 7, 14, 28],
                          group_cols=["store_id", "product_id"])
    print("✅ Lag features added")

    df = add_rolling_features(df, "units_sold", windows=[7, 14, 30],
                              group_cols=["store_id", "product_id"])
    print("✅ Rolling features added")

    df = add_calendar_features(df)
    print("✅ Calendar features added")

    before = len(df)
    df = df.dropna(subset=["units_sold_lag_28"]).reset_index(drop=True)
    print(f"✅ Dropped {before - len(df)} rows with NaN from lag/rolling")

    dates = df["date"].sort_values().unique()
    n = len(dates)
    train_end = dates[int(n * 0.70)]
    val_end = dates[int(n * 0.85)]

    train_df = df[df["date"] <= train_end]
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test_df = df[df["date"] > val_end]

    print("\n── Split ──")
    print(f"Train: {len(train_df)} rows | hingga {train_end.date()}")
    print(f"Val:   {len(val_df)} rows | hingga {val_end.date()}")
    print(f"Test:  {len(test_df)} rows | setelah {val_end.date()}")

    scalers = fit_scalers_per_group(train_df, "units_sold", ["store_id", "product_id"])
    save_scalers(scalers)

    df = apply_scalers(df, "units_sold", ["store_id", "product_id"], scalers)
    print("✅ Normalisasi selesai")

    # Walk-forward splits
    splits = walk_forward_splits(df, date_col="date", n_splits=5, horizon=14)
    save_splits(splits)
    validate_splits(splits)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(f"Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    df = run_feature_pipeline(
        input_csv="data/simulated/inventory_with_sensors.csv",
        output_csv="data/processed/inventory_features.csv"
    )