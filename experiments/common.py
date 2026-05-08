import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from retail_inventory.evaluation.metrics import forecast_metrics, paired_bootstrap_cost, quantile_metrics, summarize_inventory
from retail_inventory.evaluation.plots import plot_accuracy_cost_scatter, plot_cost_breakdown
from retail_inventory.features.build import build_supervised_frame, encode_features, split_by_time
from retail_inventory.inventory.policies import build_point_safety_policy, build_policy_table, critical_fractile
from retail_inventory.inventory.simulator import initial_inventory_from_history, simulate_inventory
from retail_inventory.models.calibration import CostAwareCalibrator
from retail_inventory.models.classical import make_classical_predictions
from retail_inventory.models.tabular import PointGBDT, QuantileGBDT
from retail_inventory.utils.config import ensure_dir


def _q_cols(alphas):
    return ["q_%s" % a for a in alphas]


def _forecast_row(model, y_true, y_pred, q_df=None, alphas=None):
    out = {"model": model}
    out.update(forecast_metrics(y_true, y_pred))
    if q_df is not None and alphas is not None:
        out.update(quantile_metrics(y_true, q_df, alphas))
    return out


def run_panel_experiment(panel, cfg, output_prefix, include_classical=True):
    seed = cfg.get("seed", 2026)
    horizon = int(cfg["data"]["horizon"])
    lead_time = int(cfg["data"]["lead_time"])
    review_period = int(cfg["data"]["review_period"])
    protection_period = lead_time + review_period
    alphas = [float(a) for a in cfg["features"]["quantile_alphas"]]
    holding_cost = float(cfg["inventory"]["holding_cost"])
    stockout_cost = float(cfg["inventory"]["stockout_cost"])
    tau_floor = float(cfg["inventory"]["service_level_floor"])
    tau_base = max(tau_floor, critical_fractile(stockout_cost, holding_cost))

    frame = build_supervised_frame(
        panel,
        horizon=horizon,
        protection_period=protection_period,
        lags=cfg["features"]["lags"],
        rolling_windows=cfg["features"]["rolling_windows"],
        review_period=review_period,
    )
    train, val, test = split_by_time(
        frame,
        test_days=int(cfg["experiments"].get("m5", {}).get("test_days", 84)),
        validation_days=int(cfg["experiments"].get("m5", {}).get("validation_days", 56)),
    )
    (X_train, X_val, X_test), feature_cols = encode_features(train, [val, test], target_cols=["target_pp", "target_horizon"])
    y_train = train["target_pp"].values.astype(float)
    y_val = val["target_pp"].values.astype(float)
    y_test = test["target_pp"].values.astype(float)

    forecast_rows = []
    all_policy_tables = []
    prediction_tables = []

    if include_classical:
        classical_methods = ["seasonal_naive", "moving_average", "exp_smoothing", "croston"]
        classical_test = make_classical_predictions(panel, test, classical_methods, horizon, protection_period, alphas)
        for method in classical_methods:
            pred_m = classical_test[classical_test["model"] == method].reset_index(drop=True)
            forecast_rows.append(_forecast_row(method, y_test, pred_m["pp_mean"].values, pred_m[_q_cols(alphas)], alphas))
            all_policy_tables.append(
                build_policy_table(pred_m, alphas, stockout_cost, holding_cost, method, tau_floor=tau_floor)
            )
        prediction_tables.append(classical_test)

    point = PointGBDT(params=cfg["models"].get("lightgbm", cfg["models"].get("gbdt", {})), seed=seed).fit(X_train, y_train)
    point_val = point.predict(X_val)
    point_test = point.predict(X_test)
    residual_std = float(np.std(y_val - point_val)) if len(y_val) else 1.0
    point_preds = test[["series_id", "origin_t"]].copy()
    point_preds["model"] = "point_gbdt_safety"
    point_preds["pp_mean"] = point_test
    for a in alphas:
        point_preds["q_%s" % a] = np.maximum(0.0, point_test + residual_std * _normal_ppf(a))
    forecast_rows.append(_forecast_row("point_gbdt_safety", y_test, point_test, point_preds[_q_cols(alphas)], alphas))
    all_policy_tables.append(build_point_safety_policy(point_preds, stockout_cost, holding_cost, "point_gbdt_safety"))
    prediction_tables.append(point_preds)

    quant = QuantileGBDT(alphas=alphas, params=cfg["models"].get("lightgbm", cfg["models"].get("gbdt", {})), seed=seed).fit(X_train, y_train)
    q_val = quant.predict(X_val)
    q_test = quant.predict(X_test)
    q_test_meta = pd.concat([test[["series_id", "origin_t"]].reset_index(drop=True), q_test], axis=1)
    q_test_meta["model"] = "pp_quantile_fixed"
    q_test_meta["pp_mean"] = q_test["q_0.5"].values if "q_0.5" in q_test else q_test.iloc[:, 0].values
    forecast_rows.append(
        _forecast_row("pp_quantile_fixed", y_test, q_test_meta["pp_mean"].values, q_test_meta[_q_cols(alphas)], alphas)
    )
    all_policy_tables.append(build_policy_table(q_test_meta, alphas, stockout_cost, holding_cost, "pp_quantile_fixed", tau_floor=tau_floor))
    prediction_tables.append(q_test_meta)

    calib_feature_candidates = [
        c
        for c in feature_cols
        if ("zero_ratio" in c or "cv2" in c or "roll_" in c or "lag_" in c or c in ["price", "promo", "event", "dow", "month"])
    ]
    calib_cols = [feature_cols.index(c) for c in calib_feature_candidates[: min(64, len(calib_feature_candidates))]]
    if len(calib_cols) == 0:
        calib_cols = list(range(min(32, X_train.shape[1])))
    calibrator = CostAwareCalibrator(
        alphas=alphas,
        tau_base=tau_base,
        tau_floor=tau_floor,
        tau_eps=float(cfg["inventory"]["tau_epsilon"]),
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        epochs=int(cfg["models"]["calibrator"]["epochs"]),
        batch_size=int(cfg["models"]["calibrator"]["batch_size"]),
        lr=float(cfg["models"]["calibrator"]["lr"]),
        weight_decay=float(cfg["models"]["calibrator"]["weight_decay"]),
        tau_regularization=float(cfg["models"]["calibrator"]["tau_regularization"]),
        seed=seed,
    )
    calibrator.fit(X_val.values[:, calib_cols], q_val[_q_cols(alphas)].values, y_val)
    s_test, tau_test = calibrator.predict_order_up_to(X_test.values[:, calib_cols], q_test[_q_cols(alphas)].values)
    prop_policy = test[["series_id", "origin_t"]].copy()
    prop_policy["model"] = "pp_quantile_calibrated"
    prop_policy["order_up_to"] = s_test
    prop_policy["tau"] = tau_test
    all_policy_tables.append(prop_policy)

    prop_pred = q_test_meta.copy()
    prop_pred["model"] = "pp_quantile_calibrated"
    prop_pred["pp_mean"] = s_test
    forecast_rows.append(_forecast_row("pp_quantile_calibrated", y_test, q_test_meta["pp_mean"].values, q_test_meta[_q_cols(alphas)], alphas))
    prediction_tables.append(prop_pred)

    start_t = int(test["origin_t"].min())
    end_t = int(test["origin_t"].max() + horizon)
    initial = initial_inventory_from_history(panel[panel["t"] < start_t], protection_period, cfg["inventory"]["initial_quantile"])
    inv_summaries = []
    inv_traces = []
    for pol in all_policy_tables:
        trace, summary = simulate_inventory(
            panel,
            pol,
            lead_time=lead_time,
            review_period=review_period,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            fixed_order_cost=float(cfg["inventory"]["fixed_order_cost"]),
            initial_levels=initial,
            start_t=start_t,
            end_t=end_t,
            warmup_days=int(cfg["data"]["warmup_days"]),
        )
        inv_traces.append(trace)
        inv_summaries.append(summary)

    forecast_table = pd.DataFrame(forecast_rows)
    inventory_series = pd.concat(inv_summaries, axis=0, ignore_index=True)
    inventory_table = summarize_inventory(inventory_series)
    fixed_name = "pp_quantile_fixed"
    proposed_name = "pp_quantile_calibrated"
    boot = paired_bootstrap_cost(
        inventory_series,
        baseline_model=fixed_name,
        candidate_model=proposed_name,
        n_boot=int(cfg["experiments"].get("bootstrap_samples", 300)),
        seed=seed,
    )
    boot_row = {"baseline": fixed_name, "candidate": proposed_name}
    boot_row.update(boot)
    boot_table = pd.DataFrame([boot_row])

    tables_dir = ensure_dir(os.path.join(ROOT, "outputs", "tables"))
    figs_dir = ensure_dir(os.path.join(ROOT, "outputs", "figures"))
    forecast_table.to_csv(os.path.join(tables_dir, "%s_forecast_metrics.csv" % output_prefix), index=False)
    inventory_table.to_csv(os.path.join(tables_dir, "%s_inventory_metrics.csv" % output_prefix), index=False)
    inventory_series.to_csv(os.path.join(tables_dir, "%s_inventory_by_series.csv" % output_prefix), index=False)
    boot_table.to_csv(os.path.join(tables_dir, "%s_bootstrap.csv" % output_prefix), index=False)
    pd.concat(prediction_tables, axis=0, ignore_index=True).to_csv(
        os.path.join(tables_dir, "%s_predictions.csv" % output_prefix), index=False
    )
    plot_cost_breakdown(inventory_table, os.path.join(figs_dir, "%s_cost_breakdown.png" % output_prefix))
    plot_accuracy_cost_scatter(forecast_table, inventory_table, os.path.join(figs_dir, "%s_accuracy_cost_scatter.png" % output_prefix))

    return {
        "frame": frame,
        "train": train,
        "val": val,
        "test": test,
        "forecast_table": forecast_table,
        "inventory_table": inventory_table,
        "inventory_series": inventory_series,
        "bootstrap": boot_table,
    }


def _normal_ppf(alpha):
    from scipy.stats import norm

    return norm.ppf(float(alpha))
