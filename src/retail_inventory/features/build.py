import numpy as np
import pandas as pd


STATIC_COLS = ["item_id", "store_id", "category_id", "state_id"]


def add_intermediate_features(panel, rolling_windows):
    panel = panel.sort_values(["series_id", "t"]).copy()
    panel["dow"] = pd.to_datetime(panel["date"]).dt.dayofweek
    panel["month"] = pd.to_datetime(panel["date"]).dt.month
    panel["is_weekend"] = panel["dow"].isin([5, 6]).astype(int)
    for w in rolling_windows:
        shifted = panel.groupby("series_id")["demand"].shift(1)
        panel["roll_mean_%d" % w] = shifted.groupby(panel["series_id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        panel["roll_std_%d" % w] = shifted.groupby(panel["series_id"]).transform(lambda s: s.rolling(w, min_periods=1).std())
        zero = (panel["demand"] <= 0).astype(float).groupby(panel["series_id"]).shift(1)
        panel["zero_ratio_%d" % w] = zero.groupby(panel["series_id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
    panel["roll_std_28"] = panel.get("roll_std_28", 0.0)
    panel["roll_mean_28"] = panel.get("roll_mean_28", 0.0)
    panel["cv2_28"] = (panel["roll_std_28"].fillna(0.0) / (panel["roll_mean_28"].fillna(0.0) + 1e-6)) ** 2
    return panel


def build_supervised_frame(panel, horizon, protection_period, lags, rolling_windows, review_period=7):
    """Build origin-level supervised rows.

    An origin row at time t uses demand up to and including t, then predicts
    t+1...t+horizon and the protection-period cumulative target.
    """
    panel = panel.sort_values(["series_id", "t"]).copy()
    panel = add_intermediate_features(panel, rolling_windows)
    rows = []
    max_lag = max(max(lags), max(rolling_windows))
    for sid, g in panel.groupby("series_id", sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        demand = g["demand"].values.astype(float)
        for idx in range(max_lag, len(g) - horizon):
            if (idx - max_lag) % review_period != 0:
                continue
            row = {
                "series_id": sid,
                "origin_t": int(g.loc[idx, "t"]),
                "origin_date": g.loc[idx, "date"],
                "price": float(g.loc[idx, "price"]),
                "promo": int(g.loc[idx, "promo"]),
                "event": int(g.loc[idx, "event"]),
                "dow": int(g.loc[idx, "dow"]),
                "month": int(g.loc[idx, "month"]),
                "is_weekend": int(g.loc[idx, "is_weekend"]),
                "cv2_28": float(g.loc[idx, "cv2_28"]) if not pd.isnull(g.loc[idx, "cv2_28"]) else 0.0,
            }
            for c in STATIC_COLS:
                if c in g:
                    row[c] = g.loc[idx, c]
            for lag in lags:
                row["lag_%d" % lag] = float(demand[idx - lag + 1])
            for w in rolling_windows:
                row["roll_mean_%d" % w] = float(g.loc[idx, "roll_mean_%d" % w])
                row["roll_std_%d" % w] = float(g.loc[idx, "roll_std_%d" % w]) if not pd.isnull(g.loc[idx, "roll_std_%d" % w]) else 0.0
                row["zero_ratio_%d" % w] = float(g.loc[idx, "zero_ratio_%d" % w]) if not pd.isnull(g.loc[idx, "zero_ratio_%d" % w]) else 0.0
            future = demand[idx + 1 : idx + horizon + 1]
            for h in range(1, horizon + 1):
                row["y_h%d" % h] = float(future[h - 1])
            row["target_pp"] = float(future[:protection_period].sum())
            row["target_horizon"] = float(future.sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No supervised rows produced; check horizon/lags/history length.")
    return df


def encode_features(train_df, other_dfs, target_cols):
    all_df = pd.concat([train_df] + list(other_dfs), axis=0, ignore_index=True)
    drop_cols = set(["series_id", "origin_date"] + list(target_cols))
    feature_cols = [c for c in all_df.columns if c not in drop_cols and not c.startswith("y_h")]
    cat_cols = [c for c in feature_cols if all_df[c].dtype == object]
    encoded = pd.get_dummies(all_df[feature_cols], columns=cat_cols, dummy_na=True)
    encoded = encoded.fillna(0.0)
    n_train = len(train_df)
    out = [encoded.iloc[:n_train].reset_index(drop=True)]
    start = n_train
    for df in other_dfs:
        out.append(encoded.iloc[start : start + len(df)].reset_index(drop=True))
        start += len(df)
    return out, list(encoded.columns)


def split_by_time(frame, test_days=84, validation_days=56):
    max_t = frame["origin_t"].max()
    test_start = max_t - test_days + 1
    val_start = test_start - validation_days
    train = frame[frame["origin_t"] < val_start].reset_index(drop=True)
    val = frame[(frame["origin_t"] >= val_start) & (frame["origin_t"] < test_start)].reset_index(drop=True)
    test = frame[frame["origin_t"] >= test_start].reset_index(drop=True)
    if len(val) == 0:
        val = train.sample(frac=0.2, random_state=2026).reset_index(drop=True)
        train = train.drop(val.index, errors="ignore").reset_index(drop=True)
    return train, val, test
