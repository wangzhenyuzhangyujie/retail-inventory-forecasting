import numpy as np
import pandas as pd


def forecast_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))
    smape = np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-6))
    wape = np.sum(np.abs(err)) / (np.sum(np.abs(y_true)) + 1e-6)
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape, "WAPE": wape}


def pinball_loss(y_true, q_pred, alpha):
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y_true - q_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff))


def quantile_metrics(y_true, q_df, alphas):
    out = {}
    for a in alphas:
        col = "q_%s" % a
        out["pinball_%s" % a] = pinball_loss(y_true, q_df[col].values, float(a))
        out["coverage_%s" % a] = np.mean(y_true <= q_df[col].values)
    out["pinball_mean"] = np.mean([out["pinball_%s" % a] for a in alphas])
    return out


def summarize_inventory(summary_df):
    metric_cols = [
        "total_cost",
        "holding_cost",
        "stockout_cost",
        "ordering_cost",
        "fill_rate",
        "cycle_service_level",
        "stockout_rate",
        "average_on_hand",
    ]
    grouped = summary_df.groupby("model")[metric_cols].mean().reset_index()
    base_cost = grouped["total_cost"].max()
    grouped["cost_reduction_vs_worst"] = 1.0 - grouped["total_cost"] / max(base_cost, 1e-6)
    return grouped.sort_values("total_cost").reset_index(drop=True)


def paired_bootstrap_cost(summary_df, baseline_model, candidate_model, n_boot=300, seed=2026):
    base = summary_df[summary_df["model"] == baseline_model][["series_id", "total_cost"]].rename(columns={"total_cost": "base"})
    cand = summary_df[summary_df["model"] == candidate_model][["series_id", "total_cost"]].rename(columns={"total_cost": "cand"})
    merged = base.merge(cand, on="series_id", how="inner")
    if len(merged) == 0:
        return {"mean_reduction": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rng = np.random.RandomState(seed)
    reductions = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(merged), size=len(merged))
        sample = merged.iloc[idx]
        reductions.append(1.0 - sample["cand"].mean() / max(sample["base"].mean(), 1e-6))
    return {
        "mean_reduction": float(1.0 - merged["cand"].mean() / max(merged["base"].mean(), 1e-6)),
        "ci_low": float(np.quantile(reductions, 0.025)),
        "ci_high": float(np.quantile(reductions, 0.975)),
    }
