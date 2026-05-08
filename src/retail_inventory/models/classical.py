import numpy as np
import pandas as pd
from scipy.stats import norm


def croston_forecast(history, alpha=0.1):
    demand = np.asarray(history, dtype=float)
    positives = np.where(demand > 0)[0]
    if len(positives) == 0:
        return 0.0
    z = demand[positives[0]]
    p = max(1.0, positives[0] + 1.0)
    last = positives[0]
    for idx in positives[1:]:
        z = alpha * demand[idx] + (1.0 - alpha) * z
        p = alpha * (idx - last) + (1.0 - alpha) * p
        last = idx
    return max(0.0, z / max(p, 1e-6))


def _point_forecast(history, method, horizon):
    history = np.asarray(history, dtype=float)
    if len(history) == 0:
        return np.zeros(horizon)
    if method == "seasonal_naive":
        if len(history) >= 7:
            base = history[-7:]
            return np.resize(base, horizon)
        return np.repeat(history[-1], horizon)
    if method == "moving_average":
        return np.repeat(np.mean(history[-28:]), horizon)
    if method == "exp_smoothing":
        alpha = 0.25
        level = history[0]
        for y in history[1:]:
            level = alpha * y + (1 - alpha) * level
        return np.repeat(level, horizon)
    if method == "croston":
        return np.repeat(croston_forecast(history), horizon)
    raise ValueError("Unknown classical method: %s" % method)


def make_classical_predictions(panel, frame, methods, horizon, protection_period, alphas):
    rows = []
    grouped = {sid: g.sort_values("t").reset_index(drop=True) for sid, g in panel.groupby("series_id", sort=False)}
    for _, row in frame.iterrows():
        sid = row["series_id"]
        g = grouped[sid]
        origin_t = int(row["origin_t"])
        history = g[g["t"] <= origin_t]["demand"].values.astype(float)
        hist_pp = []
        if len(history) > protection_period:
            for i in range(0, len(history) - protection_period):
                hist_pp.append(history[i : i + protection_period].sum())
        hist_pp = np.asarray(hist_pp, dtype=float)
        hist_std = float(np.std(hist_pp)) if len(hist_pp) > 2 else max(1.0, np.std(history[-28:]))
        for method in methods:
            daily = _point_forecast(history, method, horizon)
            pp_mean = float(np.sum(daily[:protection_period]))
            pred = {
                "series_id": sid,
                "origin_t": origin_t,
                "model": method,
                "pp_mean": max(0.0, pp_mean),
            }
            for a in alphas:
                pred["q_%s" % a] = max(0.0, pp_mean + hist_std * norm.ppf(float(a)))
            rows.append(pred)
    return pd.DataFrame(rows)
