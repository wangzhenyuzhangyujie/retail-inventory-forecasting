# 零售需求预测与库存补货优化

这是研究生《高级机器学习理论》课程报告配套的代码仓库。仓库里放了实验代码、配置文件、已经生成的结果表和图片，也保留了数据准备说明与实验审计文档，方便检查报告中的结果来源。

这个项目想回答的问题很直接：

```text
更低的销量预测误差，是否一定带来更低的库存成本？
```

围绕这个问题，实验采用的主线是：

```text
保护期需求分位数预测
+ 成本感知的关键分位数校准
+ 滚动周期补货库存仿真
```

为了对应课程报告的三个层次，实验被拆成了三步：

1. 先用经典基础算法跑通合成零售需求仿真。
2. 再在真实竞赛数据集上建立 GBDT / 分位数预测基线。
3. 最后加入面向预测到决策转换的成本感知校准方法，并做消融和敏感性分析。

## 仓库结构

```text
retail_inventory_forecasting/
  configs/                 # 实验配置
  data/                    # 数据准备说明；原始大数据不进入 git
  experiments/             # 可直接运行的实验入口
  outputs/                 # 报告使用的结果表和图片
  reports/                 # 报告草稿、代码审计和实验说明
  src/retail_inventory/    # 数据、特征、模型、库存仿真和评价代码
  requirements.txt         # Python 依赖
```

## 环境配置

报告中的结果是在 Python 3.7 环境下得到的。更高版本的 Python 3.x 通常也能运行，不过如果只是为了复现课程报告，建议优先使用下面这套依赖配置。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装后可以先做一次依赖导入检查：

```bash
python - <<'PY'
import numpy, pandas, scipy, sklearn, statsmodels, yaml
import matplotlib, seaborn, torch, lightgbm, xgboost
print("imports_ok")
PY
```

## 数据准备

原始竞赛数据没有直接放进 git 仓库。M5 的主要 CSV 文件超过 GitHub 普通仓库的单文件大小限制，Store Item 数据则由 Kaggle 分发，二者都更适合按下面的方式单独准备。

代码默认读取的数据目录如下：

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

M5 数据可以用仓库里的脚本下载：

```bash
python experiments/download_m5.py
```

这个脚本内部会调用系统 `curl`。如果运行环境里没有 `curl`，可以先安装 `curl`，也可以直接使用下面的 GitHub Release 数据包。

课程提交版本同时在 GitHub Release 中放了 M5 原始数据压缩包：

```bash
gh release download v1.0-course-submission \
  --repo wangzhenyuzhangyujie/retail-inventory-forecasting \
  --pattern m5_raw_data.zip
unzip -o m5_raw_data.zip -d data/raw/m5
```

Store Item Demand Forecasting Challenge 数据需要从 Kaggle 官方页面下载，然后放入 `data/raw/store_item/`：

```text
https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
```

数据来源、文件放置方式和许可说明可以继续看 [data/README.md](data/README.md)。

## 快速检查

下面这些命令只检查实验入口是否可调用，不会启动完整实验：

```bash
python experiments/run_layer1_synthetic.py --help
python experiments/run_m5_full.py --help
python experiments/run_store_item_full.py --help
```

M5 原始数据准备好后，可以先跑一个小样本 smoke test：

```bash
python experiments/run_m5_full.py --config configs/default.yaml --n-series 10
```

## 完整复现实验

报告中的主要实验可以按下面的命令复现：

```bash
python experiments/run_layer1_synthetic.py --config configs/default.yaml
python experiments/download_m5.py
python experiments/run_m5_full.py --config configs/default.yaml --n-series 1000
python experiments/run_store_item_full.py --config configs/default.yaml --output-prefix store_item_full
python experiments/run_ablation.py --dataset m5 --n-series 1000
python experiments/run_sensitivity.py --dataset m5 --n-series 1000
```

`experiments/run_all.py` 会依次运行 synthetic、M5 主实验、M5 消融实验和 M5 敏感性分析。Store Item 需要先手动准备 Kaggle 数据，所以这里单独列出运行命令。

实验输出会写到：

```text
outputs/tables/
outputs/figures/
```

## 报告主要结果

课程报告里引用的结果文件已经放在 `outputs/` 目录下，可以直接查看；如果需要核验，也可以按上面的命令重新运行。

| 数据集 | 固定分位数基线成本 | 提出方法成本 | 总库存成本相对下降 |
|---|---:|---:|---:|
| M5 子集，1000 条 bottom-level SKU-store 序列 | 1009.83 | 959.11 | 5.02% |
| Store Item，全量 500 条 store-item 序列 | 15726.16 | 15529.37 | 1.25% |

M5 消融实验显示，去掉自适应 critical-quantile 校准后，方法基本退化为固定分位数基线。这说明主要收益来自成本感知的补货分位数校准，而不是结果表命名或后处理差异。

## 评价设置说明

本项目使用的是本地 rolling-origin 补货仿真，不声称参加 Kaggle 官方 hidden-test leaderboard，也不声称取得官方排名。

在每个补货检查点，实验假设此前已经发生并被观测到的销量可以作为历史信息。测试标签只用于最终预测指标和库存仿真评价，不参与模型训练。关于代码泄漏检查和仍然存在的严谨性限制，见 [reports/code_leakage_audit.md](reports/code_leakage_audit.md)。

## 公开链接

项目代码：

```text
https://github.com/wangzhenyuzhangyujie/retail-inventory-forecasting
```

M5 数据 Release：

```text
https://github.com/wangzhenyuzhangyujie/retail-inventory-forecasting/releases/tag/v1.0-course-submission
```

## 许可

本仓库代码采用 MIT License。数据版权和许可归原始数据提供方所有。
