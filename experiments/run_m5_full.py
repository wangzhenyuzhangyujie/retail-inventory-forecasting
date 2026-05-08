import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import run_panel_experiment
from retail_inventory.data.m5 import load_m5_panel
from retail_inventory.utils.config import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--n-series", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    n_series = args.n_series or int(cfg["experiments"]["m5"]["n_series"])
    panel = load_m5_panel(
        raw_dir=os.path.join(ROOT, "data", "raw", "m5"),
        url_base=cfg["data"]["m5_url_base"],
        n_series=n_series,
        min_last_days=int(cfg["data"]["m5_min_last_days"]),
        seed=cfg.get("seed", 2026),
    )
    result = run_panel_experiment(panel, cfg, output_prefix="m5_n%d" % n_series, include_classical=True)
    print("Forecast metrics")
    print(result["forecast_table"].sort_values("RMSE").to_string(index=False))
    print("\nInventory metrics")
    print(result["inventory_table"].to_string(index=False))
    print("\nBootstrap")
    print(result["bootstrap"].to_string(index=False))


if __name__ == "__main__":
    main()
