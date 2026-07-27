import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_").replace("/", "_") for c in df.columns]
    return df


def audit_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    audit = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return audit[audit["missing_count"] > 0].sort_values("missing_pct", ascending=False)


def impute_by_group(df: pd.DataFrame, col: str, group_cols: list, strategy: str = "median") -> pd.DataFrame:
    if strategy == "median":
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))
    elif strategy == "ffill_bfill":
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.ffill().bfill())
    return df


def detect_outliers_iqr(df: pd.DataFrame, col: str, group_cols: list, factor: float = 1.5) -> pd.DataFrame:
    def flag_group(g):
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        return (g[col] < lower) | (g[col] > upper)

    flags = []
    for _, group in df.groupby(group_cols):
        flags.append(flag_group(group))
    df["is_outlier"] = pd.concat(flags).sort_index()
    return df


def cap_outliers(df: pd.DataFrame, col: str, group_cols: list, factor: float = 1.5) -> pd.DataFrame:
    def cap_group(g):
        q1, q3 = g[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        g[col] = g[col].clip(lower=lower, upper=upper)
        return g
    return df.groupby(group_cols, group_keys=False).apply(cap_group)


def standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    return df


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing dates per (store_id, product_id) using merge approach."""
    all_groups = df[["store_id", "product_id"]].drop_duplicates()
    date_min = df["date"].min()
    date_max = df["date"].max()
    full_range = pd.date_range(date_min, date_max, freq="D")

    full_index = all_groups.merge(pd.DataFrame({"date": full_range}), how="cross")
    df = full_index.merge(df, on=["store_id", "product_id", "date"], how="left")
    return df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)


def run_pipeline(raw_path: str) -> pd.DataFrame:
    df = load_data(raw_path)
    print(f"Loaded: {df.shape}")

    print("\n── Missing Value Audit ──")
    print(audit_missing(df))

    df = standardize_dates(df)

    df = fill_missing_dates(df)
    print(f"\nAfter date reindex: {df.shape}")

    numeric_cols = ["inventory_level", "units_sold", "units_ordered",
                    "demand_forecast", "price", "discount", "competitor_pricing"]
    for col in numeric_cols:
        df = impute_by_group(df, col, ["store_id", "product_id"], strategy="median")

    for col in ["weather_condition", "seasonality", "category", "region"]:
        df[col] = df[col].fillna("unknown")

    df = detect_outliers_iqr(df, "units_sold", ["store_id", "product_id"])
    print(f"\nOutlier rows flagged: {df['is_outlier'].sum()}")

    df.loc[df["units_sold"] < 0, "units_sold"] = 0
    df.loc[df["inventory_level"] < 0, "inventory_level"] = 0

    df.to_csv("data/processed/inventory_clean.csv", index=False)
    print("\nSaved: data/processed/inventory_clean.csv")

    return df


if __name__ == "__main__":
    df = run_pipeline("data/raw/retail_store_inventory.csv")
    print(df.head())