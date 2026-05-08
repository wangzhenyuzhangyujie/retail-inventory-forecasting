import os

import pandas as pd


def load_store_item_panel(raw_dir):
    """Load the Store Item Demand Forecasting Challenge data.

    The original Kaggle dataset has daily sales for 10 stores and 50 items
    from 2013-01-01 to 2017-12-31. It only contains date/store/item/sales,
    so optional AI Cases covariates such as price and promotions are set to
    neutral values.
    """
    train_path = os.path.join(raw_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Store Item train.csv not found in %s. Expected the Kaggle Store Item Demand Forecasting Challenge file."
            % raw_dir
        )
    df = pd.read_csv(train_path, parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Store Item train.csv is missing columns: %s" % sorted(missing))

    panel = df.rename(columns={"sales": "demand"}).copy()
    panel["store_id"] = "store_" + panel["store"].astype(str)
    panel["item_id"] = "item_" + panel["item"].astype(str)
    panel["series_id"] = panel["item_id"] + "_" + panel["store_id"]
    panel["category_id"] = "store_item"
    panel["state_id"] = "unknown"
    panel["price"] = 1.0
    panel["promo"] = 0
    panel["event"] = 0
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)
    panel["t"] = panel.groupby("series_id").cumcount()
    return panel[
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
        ]
    ]
