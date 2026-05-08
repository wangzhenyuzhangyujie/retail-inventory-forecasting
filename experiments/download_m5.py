import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from retail_inventory.data.download import download_m5
from retail_inventory.utils.config import load_config


def main():
    cfg = load_config(os.path.join(ROOT, "configs", "default.yaml"))
    raw_dir = os.path.join(ROOT, "data", "raw", "m5")
    paths = download_m5(raw_dir, cfg["data"]["m5_url_base"])
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
