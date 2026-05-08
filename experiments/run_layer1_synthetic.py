import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common import run_panel_experiment
from retail_inventory.data.synthetic import generate_synthetic_panel
from retail_inventory.utils.config import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 2026))
    s_cfg = cfg["experiments"]["synthetic"]
    panel = generate_synthetic_panel(
        n_series=int(s_cfg["n_series"]),
        n_days=int(s_cfg["n_days"]),
        seed=cfg.get("seed", 2026),
    )
    result = run_panel_experiment(panel, cfg, output_prefix="layer1_synthetic", include_classical=True)
    print("Forecast metrics")
    print(result["forecast_table"].sort_values("RMSE").to_string(index=False))
    print("\nInventory metrics")
    print(result["inventory_table"].to_string(index=False))
    print("\nBootstrap")
    print(result["bootstrap"].to_string(index=False))


if __name__ == "__main__":
    main()
