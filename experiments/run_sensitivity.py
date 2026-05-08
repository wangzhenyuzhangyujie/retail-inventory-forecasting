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
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    if args.dataset == "m5":
        panel = load_m5_panel(
            raw_dir=os.path.join(ROOT, "data", "raw", "m5"),
            url_base=cfg["data"]["m5_url_base"],
            n_series=args.n_series,
            min_last_days=int(cfg["data"]["m5_min_last_days"]),
            seed=cfg.get("seed", 2026),
        )
    else:
        panel = generate_synthetic_panel(n_series=args.n_series, n_days=420, seed=cfg.get("seed", 2026))

    scenarios = []
    for lead_time in [3, 7, 14]:
        for stockout_cost in [3.0, 5.0, 9.0]:
            c = copy.deepcopy(cfg)
            c["data"]["lead_time"] = lead_time
            c["inventory"]["stockout_cost"] = stockout_cost
            tag = "%s_L%d_cu%s" % (args.dataset, lead_time, str(stockout_cost).replace(".", "p"))
            print("Running scenario", tag)
            result = run_panel_experiment(panel, c, output_prefix="sensitivity_%s" % tag, include_classical=False)
            inv = result["inventory_table"].copy()
            inv["lead_time"] = lead_time
            inv["stockout_cost_setting"] = stockout_cost
            scenarios.append(inv)

    out = os.path.join(ROOT, "outputs", "tables", "sensitivity_%s_summary.csv" % args.dataset)
    import pandas as pd

    pd.concat(scenarios, axis=0, ignore_index=True).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
