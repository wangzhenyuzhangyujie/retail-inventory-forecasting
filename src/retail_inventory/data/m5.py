import os

import numpy as np
import pandas as pd

from retail_inventory.data.download import download_m5


def ensure_m5(raw_dir, url_base):
    required = ["calendar.csv", "sales_train_validation.csv", "sell_prices.csv"]
    if not all(os.path.exists(os.path.join(raw_dir, f)) for f in required):
        download_m5(raw_dir, url_base)


def load_m5_panel(raw_dir, url_base=None, n_series=1000, min_last_days=420, seed=2026):
    """Load M5 into long panel format. Uses a deterministic bottom-level subset."""
    if url_base is not None:
        ensure_m5(raw_dir, url_base)
    sales_path = os.path.join(raw_dir, "sales_train_validation.csv")
    calendar_path = os.path.join(raw_dir, "calendar.csv")
    price_path = os.path.join(raw_dir, "sell_prices.csv")
    if not (os.path.exists(sales_path) and os.path.exists(calendar_path) and os.path.exists(price_path)):
        raise FileNotFoundError("M5 raw files not found in %s" % raw_dir)

    sales = pd.read_csv(sales_path)
    rng = np.random.RandomState(seed)
    n_series = min(n_series, len(sales))
    idx = np.sort(rng.choice(np.arange(len(sales)), size=n_series, replace=False))
    sales = sales.iloc[idx].copy()

    d_cols = [c for c in sales.columns if c.startswith("d_")]
    if min_last_days:
        d_cols = d_cols[-int(min_last_days) :]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long_df = sales[id_cols + d_cols].melt(id_vars=id_cols, var_name="d", value_name="demand")
    long_df["series_id"] = long_df["id"].str.replace("_validation", "", regex=False)

    calendar = pd.read_csv(calendar_path)
    calendar = calendar[["d", "date", "wm_yr_wk", "weekday", "month", "year", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]]
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar["event"] = calendar["event_name_1"].notnull().astype(int)
    long_df = long_df.merge(calendar, on="d", how="left")

    prices = pd.read_csv(price_path)
    prices = prices.merge(
        long_df[["store_id", "item_id", "wm_yr_wk"]].drop_duplicates(),
        on=["store_id", "item_id", "wm_yr_wk"],
        how="inner",
    )
    long_df = long_df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long_df["price"] = long_df["sell_price"].fillna(long_df.groupby("series_id")["sell_price"].transform("median"))
    long_df["price"] = long_df["price"].fillna(long_df["price"].median()).fillna(1.0)
    long_df["promo"] = 0
    snap = []
    for _, row in long_df[["state_id", "snap_CA", "snap_TX", "snap_WI"]].iterrows():
        col = "snap_%s" % row["state_id"]
        snap.append(row[col] if col in row else 0)
    long_df["snap"] = snap
    long_df["promo"] = np.maximum(long_df["event"].values, long_df["snap"].fillna(0).values).astype(int)
    long_df = long_df.sort_values(["series_id", "date"]).reset_index(drop=True)
    long_df["t"] = long_df.groupby("series_id").cumcount()
    long_df["category_id"] = long_df["cat_id"]
    return long_df[
        [
            "series_id",
            "date",
            "t",
            "demand",
            "price",
            "promo",
            "event",
            "item_id",
            "store_id",
            "category_id",
            "state_id",
            "dept_id",
        ]
    ]
