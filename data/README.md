# Data Preparation

Raw data is not tracked in git because the main M5 CSV files exceed the normal
GitHub repository file-size limit and Store Item is distributed through Kaggle.

## Expected Layout

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

## M5

The project uses the M5 Forecasting Walmart retail data. The raw files can be
downloaded through the included helper:

```bash
python experiments/download_m5.py
```

For the course submission, the same three raw M5 files are also attached to the
GitHub Release `v1.0-course-submission` as `m5_raw_data.zip`:

```bash
gh release download v1.0-course-submission \
  --repo wangzhenyuzhangyujie/retail-inventory-forecasting \
  --pattern m5_raw_data.zip
unzip -o m5_raw_data.zip -d data/raw/m5
```

The included M5 experiment uses a deterministic sample of 1000 bottom-level
SKU-store series rather than the full official M5 leaderboard setup.

## Store Item Demand Forecasting Challenge

Store Item data should be downloaded from Kaggle:

```text
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
```

After downloading, place `train.csv`, `test.csv`, and `sample_submission.csv`
under `data/raw/store_item/`.

The report uses the full `train.csv` panel for local rolling-origin inventory
evaluation. It does not use the Kaggle hidden-test leaderboard.

## Licensing Notes

Code in this repository is MIT licensed. Dataset rights and licenses remain
with the original providers. M5 is also available through Zenodo with CC BY 4.0
metadata. Store Item raw files are not redistributed in this repository because
Kaggle competition data redistribution terms should be checked before mirroring.
