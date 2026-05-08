import os

import pandas as pd


def load_favorita_panel(raw_dir, n_series=1000):
    """Load a local Favorita/Store Sales Kaggle extract if present.

    Expected files are `train.csv` and optionally `stores.csv`, `oil.csv`, `holidays_events.csv`.
    Kaggle credentials are usually required to download this dataset, so this loader is local-only.
    """
    train_path = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError("Favorita train.csv not found in %s" % raw_dir)
    train = pd.read_csv(train_path, parse_dates=["date"])
    train["series_id"] = train["store_nbr"].astype(str) + "_" + train["family"].astype(str)
    keep = sorted(train["series_id"].unique())[:n_series]
    train = train[train["series_id"].isin(keep)].copy()
    train = train.rename(columns={"sales": "demand"})
    train["price"] = 1.0
    train["promo"] = (train.get("onpromotion", 0).fillna(0) > 0).astype(int)
    train["event"] = 0
    train["item_id"] = train["family"].astype(str)
    train["store_id"] = train["store_nbr"].astype(str)
    train["category_id"] = train["family"].astype(str)
    train["state_id"] = "EC"
    train = train.sort_values(["series_id", "date"]).reset_index(drop=True)
    train["t"] = train.groupby("series_id").cumcount()
    return train[["series_id", "date", "t", "demand", "price", "promo", "event", "item_id", "store_id", "category_id", "state_id"]]
