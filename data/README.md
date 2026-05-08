# 数据准备说明

原始数据不直接提交到 git 仓库。原因是 M5 的主要 CSV 文件超过 GitHub 普通仓库的单文件大小限制，而 Store Item 数据由 Kaggle 竞赛页面分发。

## 目录结构

请将数据放置为下面的结构：

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

## M5 数据

本项目使用 M5 Forecasting 的 Walmart 零售日销量数据。可以通过项目自带脚本下载：

```bash
python experiments/download_m5.py
```

为了便于课程检查，本提交也把同样的三个 M5 原始文件作为 GitHub Release 附件提供，文件名为 `m5_raw_data.zip`：

```bash
gh release download v1.0-course-submission \
  --repo wangzhenyuzhangyujie/retail-inventory-forecasting \
  --pattern m5_raw_data.zip
unzip -o m5_raw_data.zip -d data/raw/m5
```

Release 链接：

```text
https://github.com/wangzhenyuzhangyujie/retail-inventory-forecasting/releases/tag/v1.0-course-submission
```

报告中的 M5 实验使用确定性随机种子抽取 1000 条 bottom-level SKU-store 序列，而不是完整 M5 官方 leaderboard 设置。

## Store Item Demand Forecasting Challenge

Store Item 数据请从 Kaggle 官方页面下载：

```text
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
```

下载后，将 `train.csv`、`test.csv`、`sample_submission.csv` 放到：

```text
data/raw/store_item/
```

报告使用的是 `train.csv` 中的全量 500 条 store-item 序列，并在本地 rolling-origin 补货仿真中划分训练、验证和测试区间。本文没有使用 Kaggle hidden-test leaderboard 成绩。

## 许可说明

本仓库代码采用 MIT License。数据版权和许可归原始数据提供方所有。M5 数据也可通过 Zenodo 获取，其页面元数据标注为 CC BY 4.0。Store Item 原始文件没有在本仓库或 Release 中二次分发，因为 Kaggle 竞赛数据的再分发条款需要单独确认。
