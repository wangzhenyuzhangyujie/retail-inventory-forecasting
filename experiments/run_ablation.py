import argparse
import copy
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import run_panel_experiment
from retail_inventory.data.m5 import load_m5_panel
from retail_inventory.data.synthetic import generate_synthetic_panel
from retail_inventory.utils.config import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--dataset", choices=["synthetic", "m5"], default="synthetic")
    parser.add_argument("--n-series", type=int, default=300)
    args = parser.parse_args()
    base_cfg = load_config(args.config)
    set_seed(base_cfg.get("seed", 2026))
    if args.dataset == "m5":
        panel = load_m5_panel(
            raw_dir=os.path.join(ROOT, "data", "raw", "m5"),
            url_base=base_cfg["data"]["m5_url_base"],
            n_series=args.n_series,
            min_last_days=int(base_cfg["data"]["m5_min_last_days"]),
            seed=base_cfg.get("seed", 2026),
        )
    else:
        panel = generate_synthetic_panel(n_series=args.n_series, n_days=420, seed=base_cfg.get("seed", 2026))

    variants = []
    settings = [
        ("full", {}),
        ("no_tau_adaptation", {"inventory.tau_epsilon": 0.0}),
        ("strong_tau_regularization", {"models.calibrator.tau_regularization": 0.2}),
        ("short_lags_only", {"features.lags": [1, 7, 14], "features.rolling_windows": [7, 14]}),
    ]
    for name, overrides in settings:
        cfg = copy.deepcopy(base_cfg)
        for key, value in overrides.items():
            target = cfg
            parts = key.split(".")
            for p in parts[:-1]:
                target = target[p]
            target[parts[-1]] = value
        print("Running ablation", name)
        result = run_panel_experiment(panel, cfg, output_prefix="ablation_%s_%s" % (args.dataset, name), include_classical=False)
        inv = result["inventory_table"].copy()
        inv["ablation"] = name
        variants.append(inv)

    import pandas as pd

    out = os.path.join(ROOT, "outputs", "tables", "ablation_%s_summary.csv" % args.dataset)
    pd.concat(variants, axis=0, ignore_index=True).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
