import argparse
import os
import sys

import pandas as pd


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


CHECKS = [
    (
        "outputs/tables/layer1_synthetic_inventory_metrics.csv",
        ("model",),
        {
            ("pp_quantile_fixed",): {"total_cost": 5479.505121, "fill_rate": 0.848454},
            ("croston",): {"total_cost": 5550.834649, "fill_rate": 0.823202},
            ("moving_average",): {"total_cost": 5562.223268, "fill_rate": 0.830995},
            ("pp_quantile_calibrated",): {"total_cost": 6089.449885, "fill_rate": 0.870033},
            ("exp_smoothing",): {"total_cost": 6194.279130, "fill_rate": 0.812307},
            ("seasonal_naive",): {"total_cost": 7053.087937, "fill_rate": 0.842364},
            ("point_gbdt_safety",): {"total_cost": 7264.217143, "fill_rate": 0.891951},
        },
    ),
    (
        "outputs/tables/m5_n1000_inventory_metrics.csv",
        ("model",),
        {
            ("pp_quantile_calibrated",): {"total_cost": 959.105403, "fill_rate": 0.828434},
            ("pp_quantile_fixed",): {"total_cost": 1009.832117, "fill_rate": 0.849809},
        },
    ),
    (
        "outputs/tables/store_item_full_inventory_metrics.csv",
        ("model",),
        {
            ("pp_quantile_calibrated",): {"total_cost": 15529.367733, "fill_rate": 0.838500},
            ("pp_quantile_fixed",): {"total_cost": 15726.158562, "fill_rate": 0.846680},
        },
    ),
    (
        "outputs/tables/ablation_m5_summary.csv",
        ("ablation", "model"),
        {
            ("full", "pp_quantile_calibrated"): {"total_cost": 959.105403, "fill_rate": 0.828434},
            ("full", "pp_quantile_fixed"): {"total_cost": 1009.832117, "fill_rate": 0.849809},
            ("no_tau_adaptation", "pp_quantile_calibrated"): {"total_cost": 1009.832085, "fill_rate": 0.849809},
            ("no_tau_adaptation", "pp_quantile_fixed"): {"total_cost": 1009.832117, "fill_rate": 0.849809},
            ("strong_tau_regularization", "pp_quantile_calibrated"): {"total_cost": 959.004892, "fill_rate": 0.828371},
            ("strong_tau_regularization", "pp_quantile_fixed"): {"total_cost": 1009.832117, "fill_rate": 0.849809},
            ("short_lags_only", "pp_quantile_calibrated"): {"total_cost": 1024.857345, "fill_rate": 0.847000},
            ("short_lags_only", "pp_quantile_fixed"): {"total_cost": 1090.557850, "fill_rate": 0.868311},
        },
    ),
    (
        "outputs/tables/sensitivity_m5_summary.csv",
        ("lead_time", "stockout_cost_setting", "model"),
        {
            ("3", "3.0", "pp_quantile_calibrated"): {"total_cost": 701.002867, "fill_rate": 0.770888},
            ("3", "3.0", "pp_quantile_fixed"): {"total_cost": 701.002867, "fill_rate": 0.770888},
            ("3", "5.0", "pp_quantile_calibrated"): {"total_cost": 847.854061, "fill_rate": 0.799137},
            ("3", "5.0", "pp_quantile_fixed"): {"total_cost": 793.200654, "fill_rate": 0.791125},
            ("3", "9.0", "pp_quantile_calibrated"): {"total_cost": 881.270490, "fill_rate": 0.770888},
            ("3", "9.0", "pp_quantile_fixed"): {"total_cost": 974.459425, "fill_rate": 0.825106},
            ("7", "3.0", "pp_quantile_calibrated"): {"total_cost": 911.431599, "fill_rate": 0.828371},
            ("7", "3.0", "pp_quantile_fixed"): {"total_cost": 911.431599, "fill_rate": 0.828371},
            ("7", "5.0", "pp_quantile_calibrated"): {"total_cost": 959.105403, "fill_rate": 0.828434},
            ("7", "5.0", "pp_quantile_fixed"): {"total_cost": 1009.832117, "fill_rate": 0.849809},
            ("7", "9.0", "pp_quantile_calibrated"): {"total_cost": 1080.837146, "fill_rate": 0.830313},
            ("7", "9.0", "pp_quantile_fixed"): {"total_cost": 1211.402110, "fill_rate": 0.884633},
            ("14", "3.0", "pp_quantile_calibrated"): {"total_cost": 1272.975689, "fill_rate": 0.896218},
            ("14", "3.0", "pp_quantile_fixed"): {"total_cost": 1272.975689, "fill_rate": 0.896218},
            ("14", "5.0", "pp_quantile_calibrated"): {"total_cost": 1317.924037, "fill_rate": 0.898297},
            ("14", "5.0", "pp_quantile_fixed"): {"total_cost": 1381.640681, "fill_rate": 0.913386},
            ("14", "9.0", "pp_quantile_calibrated"): {"total_cost": 1368.467956, "fill_rate": 0.896218},
            ("14", "9.0", "pp_quantile_fixed"): {"total_cost": 1626.447695, "fill_rate": 0.939289},
        },
    ),
]


def normalize_key(values, key_cols):
    out = []
    for col, value in zip(key_cols, values):
        if col == "lead_time":
            out.append(str(int(value)))
        elif col == "stockout_cost_setting":
            out.append("%.1f" % float(value))
        else:
            out.append(str(value))
    return tuple(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    failures = []
    for rel_path, key_cols, expected_rows in CHECKS:
        path = os.path.join(ROOT, rel_path)
        if not os.path.exists(path):
            failures.append("%s is missing" % rel_path)
            continue
        df = pd.read_csv(path)
        actual = {normalize_key(row[list(key_cols)].tolist(), key_cols): row for _, row in df.iterrows()}
        for raw_key, expected_metrics in expected_rows.items():
            key = normalize_key(raw_key, key_cols)
            if key not in actual:
                failures.append("%s missing row %s" % (rel_path, key))
                continue
            row = actual[key]
            for metric, expected in expected_metrics.items():
                current = float(row[metric])
                if abs(current - expected) > args.atol:
                    failures.append(
                        "%s %s %s expected %.6f current %.6f"
                        % (rel_path, key, metric, expected, current)
                    )

    if failures:
        print("Report output check failed:")
        for item in failures:
            print("  " + item)
        return 1
    print("Report output check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
