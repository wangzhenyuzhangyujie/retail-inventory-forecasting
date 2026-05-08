# AI Cases Dataset Comparison

Source task: AI Cases "Demand Forecasting & Replenishment Optimization".

The AI Cases page lists two usable datasets for this task:

- Walmart M5 Dataset
- Store Item Demand Forecasting Challenge & Dataset

This project now evaluates both using the same rolling replenishment pipeline.

## Dataset Setup

### M5

- Source: Walmart M5 dataset listed by AI Cases.
- Local setup: deterministic subset of 1000 bottom-level SKU-store series.
- History used: last 420 days.
- Evaluation: local rolling-origin inventory simulation.

### Store Item

- Source: Store Item Demand Forecasting Challenge & Dataset listed by AI Cases.
- Local raw file source: public GitHub mirror of the Kaggle challenge CSV files because Kaggle API credentials are not configured locally.
- Data shape: 500 store-item series, 913000 daily rows.
- Period: 2013-01-01 to 2017-12-31.
- Evaluation: local rolling-origin inventory simulation.

## Layer Definition

- Layer 1: four classical baselines: seasonal naive, moving average, exponential smoothing, Croston.
- Layer 2: protection-period quantile GBDT with fixed critical fractile.
- Layer 3: protection-period quantile GBDT with cost-aware adaptive critical-fractile calibration.

## Main Inventory Results

### M5 Subset

| Layer / method | Total cost |
|---|---:|
| Layer 3: calibrated PP quantile | 959.11 |
| Layer 2: fixed PP quantile | 1009.83 |
| Best Layer 1: moving average | 1137.93 |
| Layer 1 average | 1225.55 |

Relative cost reduction:

- Layer 2 vs Layer 1 average: 17.60%.
- Layer 3 vs Layer 2: 5.02%.
- Layer 3 vs Layer 1 average: 21.74%.
- Layer 3 vs best Layer 1: 15.72%.

### Store Item Full

| Layer / method | Total cost |
|---|---:|
| Layer 3: calibrated PP quantile | 15529.37 |
| Layer 2: fixed PP quantile | 15726.16 |
| Best Layer 1: exponential smoothing | 23681.29 |
| Layer 1 average | 24504.82 |

Relative cost reduction:

- Layer 2 vs Layer 1 average: 35.82%.
- Layer 3 vs Layer 2: 1.25%.
- Layer 3 vs Layer 1 average: 36.63%.
- Layer 3 vs best Layer 1: 34.42%.

Bootstrap for Layer 3 vs Layer 2 on Store Item:

- Mean reduction: about 1.25%.
- 95% CI: [1.15%, 1.35%].

## Conclusion

The two AI Cases datasets give the same qualitative ordering under total inventory cost:

```text
Layer 3 calibrated PP quantile
< Layer 2 fixed PP quantile
< Layer 1 classical baselines
```

The effect size differs:

- On M5, adaptive calibration gives a clear additional gain over fixed quantile replenishment.
- On Store Item, most of the gain comes from switching from classical forecasting to protection-period quantile GBDT; adaptive calibration still helps, but only modestly.

This difference is reasonable because Store Item is cleaner, denser, and has fewer covariates than M5. A fixed critical-fractile policy already captures much of the decision signal, leaving less room for context-aware calibration.
