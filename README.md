# Retail Inventory Forecasting for Replenishment Optimization

This repository contains the code, experiment outputs, and report materials for a
graduate course project on retail demand forecasting and inventory
replenishment.

The project studies one question:

```text
Does a lower sales forecasting error necessarily imply lower inventory cost?
```

The main proposed method is:

```text
Protection-period quantile forecasting
+ cost-aware critical-quantile calibration
+ rolling periodic-review inventory simulation
```

The experiments follow three layers:

1. Classical algorithms on synthetic retail demand regimes.
2. Extended GBDT/quantile baselines on real competition datasets.
3. The proposed forecast-to-decision calibration method with ablation and
   sensitivity analysis.

## Repository Structure

```text
retail_inventory_forecasting/
  configs/                 # experiment configuration
  data/                    # data instructions; raw data is not tracked by git
  experiments/             # runnable experiment entry points
  outputs/                 # generated tables and figures used by the report
  reports/                 # report draft, audit notes, and experiment notes
  src/retail_inventory/    # data, feature, model, inventory, and evaluation code
  requirements.txt
```

## Environment

The experiments were run with Python 3.7. A newer Python 3.x environment may
also work, but the submitted results should be reproduced with the dependency
set below.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick dependency check:

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, statsmodels, yaml
import matplotlib, seaborn, torch, lightgbm, xgboost
print("imports_ok")
PY
```

## Data

Raw competition data is intentionally not tracked in git.

Expected local layout:

```text
data/raw/m5/
  calendar.csv
  sales_train_validation.csv
  sell_prices.csv

data/raw/store_item/
  train.csv
  test.csv
  sample_submission.csv
```

M5 can be downloaded with the included script:

```bash
python experiments/download_m5.py
```

For this course submission, the M5 raw files are also published as a GitHub
Release asset:

```bash
gh release download v1.0-course-submission \
  --repo wangzhenyuzhangyujie/retail-inventory-forecasting \
  --pattern m5_raw_data.zip
unzip -o m5_raw_data.zip -d data/raw/m5
```

Store Item Demand Forecasting Challenge data should be downloaded from Kaggle
and placed under `data/raw/store_item/`:

```text
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
```

See [data/README.md](data/README.md) for details and licensing notes.

## Smoke Tests

These checks validate the public entry points without running the full
experiment suite:

```bash
python experiments/run_layer1_synthetic.py --help
python experiments/run_m5_full.py --help
python experiments/run_store_item_full.py --help
```

Run a quick M5 sample after preparing the M5 raw files:

```bash
python experiments/run_m5_full.py --config configs/default.yaml --n-series 10
```

## Full Reproduction Commands

Run the experiments used by the report:

```bash
python experiments/run_layer1_synthetic.py --config configs/default.yaml
python experiments/download_m5.py
python experiments/run_m5_full.py --config configs/default.yaml --n-series 1000
python experiments/run_store_item_full.py --config configs/default.yaml --output-prefix store_item_full
python experiments/run_ablation.py --dataset m5 --n-series 1000
python experiments/run_sensitivity.py --dataset m5 --n-series 1000
```

`experiments/run_all.py` runs the synthetic, M5 main, M5 ablation, and M5
sensitivity experiments. Store Item is separate because it requires manual
Kaggle data preparation.

Outputs are written to:

```text
outputs/tables/
outputs/figures/
```

## Main Report Results

The submitted report uses the generated files already included under
`outputs/`.

| Dataset | Baseline | Proposed method | Relative total-cost reduction |
|---|---:|---:|---:|
| M5 subset, 1000 bottom-level SKU-store series | 1009.83 | 959.11 | 5.02% |
| Store Item, 500 store-item series | 15726.16 | 15529.37 | 1.25% |

The M5 ablation shows that removing adaptive critical-quantile calibration
makes the proposed method collapse back to the fixed-quantile baseline.

## Evaluation Setting

The project uses local rolling-origin replenishment simulation. It does not
claim official Kaggle leaderboard participation or hidden-test ranking.

At each review date, the simulator assumes demand observed up to that date is
available. Test labels are used only for final metrics and inventory simulation.
See [reports/code_leakage_audit.md](reports/code_leakage_audit.md) for the code
leakage audit and remaining rigor notes.

## License

Code in this repository is released under the MIT License. Dataset rights and
licenses remain with the original data providers.
