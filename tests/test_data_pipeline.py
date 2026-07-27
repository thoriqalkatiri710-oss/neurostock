import pandas as pd
import numpy as np
import pytest
from src.data.preprocessing import impute_by_group, detect_outliers_iqr, fill_missing_dates
from src.data.feature_engineering import (
    add_lag_features, add_rolling_features, add_calendar_features
)

@pytest.fixture
def sample_df():
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store_id": ["store_0"] * 100,
        "product_id": ["product_0"] * 100,
        "units_sold": np.random.poisson(50, 100).astype(float),
    })


def test_impute_by_group_fills_all_na(sample_df):
    df = sample_df.copy()
    df.loc[5:10, "units_sold"] = np.nan
    result = impute_by_group(df, "units_sold", ["store_id", "product_id"], strategy="median")
    assert result["units_sold"].isna().sum() == 0


def test_detect_outliers_flags_extreme_values(sample_df):
    df = sample_df.copy()
    df.loc[50, "units_sold"] = 10000  # outlier ekstrem
    result = detect_outliers_iqr(df, "units_sold", ["store_id", "product_id"])
    assert result.loc[50, "is_outlier"] == True
    assert result["is_outlier"].sum() < len(df) * 0.1  # tidak boleh terlalu banyak flag


def test_fill_missing_dates_no_gaps(sample_df):
    df = sample_df.copy()
    df_with_gap = df.drop(df.index[10:15]).reset_index(drop=True)  # buat gap
    result = fill_missing_dates(df_with_gap)
    date_diffs = result["date"].diff().dropna()
    assert (date_diffs == pd.Timedelta(days=1)).all()


def test_lag_features_no_leakage(sample_df):
    df = add_lag_features(sample_df, "units_sold", lags=[1, 7],
                          group_cols=["store_id", "product_id"])
    # baris pertama harus NaN karena tidak ada histori
    assert pd.isna(df.loc[0, "units_sold_lag_1"])
    # nilai lag_1 di baris ke-i harus sama dengan units_sold di baris ke-(i-1)
    assert df.loc[5, "units_sold_lag_1"] == sample_df.loc[4, "units_sold"]


def test_rolling_features_shifted_correctly(sample_df):
    df = add_rolling_features(sample_df, "units_sold", windows=[7],
                              group_cols=["store_id", "product_id"])
    # rolling mean di baris t TIDAK BOLEH memuat nilai units_sold di baris t sendiri
    manual_mean = sample_df["units_sold"].iloc[3:10].mean()
    assert abs(df.loc[10, "units_sold_roll_mean_7"] - manual_mean) < 1e-6


def test_calendar_features_cyclical_range(sample_df):
    df = add_calendar_features(sample_df)
    assert df["dow_sin"].between(-1, 1).all()
    assert df["dow_cos"].between(-1, 1).all()