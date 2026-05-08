import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import run_panel_experiment
from retail_inventory.data.store_item import load_store_item_panel
from retail_inventory.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    parser.add_argument("--raw-dir", default=os.path.join(ROOT, "data", "raw", "store_item"))
    parser.add_argument("--output-prefix", default="store_item_full")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_store_item_panel(args.raw_dir)
    print("Loaded Store Item panel:", panel["series_id"].nunique(), "series,", len(panel), "rows")
    result = run_panel_experiment(panel, cfg, output_prefix=args.output_prefix, include_classical=True)
    print(result["inventory_table"].sort_values("total_cost"))
    print(result["bootstrap"])


if __name__ == "__main__":
    main()
