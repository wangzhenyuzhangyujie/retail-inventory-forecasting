import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import run_panel_experiment
from retail_inventory.data.favorita import load_favorita_panel
from retail_inventory.utils.config import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw", "favorita"))
    parser.add_argument("--n-series", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    panel = load_favorita_panel(args.raw_dir, n_series=args.n_series)
    result = run_panel_experiment(panel, cfg, output_prefix="favorita_n%d" % args.n_series, include_classical=True)
    print(result["forecast_table"].sort_values("RMSE").to_string(index=False))
    print(result["inventory_table"].to_string(index=False))


if __name__ == "__main__":
    main()
