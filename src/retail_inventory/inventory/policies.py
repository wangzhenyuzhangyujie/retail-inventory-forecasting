import numpy as np
import pandas as pd


def critical_fractile(stockout_cost, holding_cost):
    return float(stockout_cost) / (float(stockout_cost) + float(holding_cost))


def build_policy_table(predictions, alphas, stockout_cost, holding_cost, model_name, tau_floor=0.90):
    tau = max(float(tau_floor), critical_fractile(stockout_cost, holding_cost))
    q_cols = ["q_%s" % a for a in alphas]
    q = predictions[q_cols].values.astype(float)
    levels = [np.interp(tau, alphas, row) for row in q]
    out = predictions[["series_id", "origin_t"]].copy()
    out["model"] = model_name
    out["order_up_to"] = np.maximum(0.0, levels)
    out["tau"] = tau
    return out


def build_point_safety_policy(predictions, stockout_cost, holding_cost, model_name="point_safety"):
    z = max(0.0, critical_fractile(stockout_cost, holding_cost))
    out = predictions[["series_id", "origin_t"]].copy()
    if "q_0.9" in predictions:
        out["order_up_to"] = predictions["q_0.9"].values
    else:
        out["order_up_to"] = predictions["pp_mean"].values * (1.0 + 0.15 + 0.15 * z)
    out["model"] = model_name
    out["tau"] = z
    return out
