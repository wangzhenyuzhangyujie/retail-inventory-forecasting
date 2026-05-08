# Initial Results Summary

Experiments were run on May 6, 2026.

## M5 Main Result

Configuration:

- 1000 bottom-level M5 SKU-store series
- `L = 7`, `R = 7`
- holding cost `h = 1`
- lost-sales cost `p = 5`
- fixed ordering cost `K = 0`

Main inventory table:

| Model | Total cost | Holding | Stockout | Fill rate | Avg. on hand |
|---|---:|---:|---:|---:|---:|
| PP quantile + calibrated | 959.11 | 840.91 | 118.19 | 0.828 | 10.78 |
| PP quantile fixed | 1009.83 | 899.69 | 110.14 | 0.850 | 11.53 |
| Moving average | 1137.93 | 1051.85 | 86.09 | 0.846 | 13.49 |
| Croston | 1154.64 | 1066.45 | 88.19 | 0.837 | 13.67 |
| Seasonal naive | 1293.00 | 1212.15 | 80.85 | 0.879 | 15.54 |
| Exp. smoothing | 1316.62 | 1238.80 | 77.82 | 0.882 | 15.88 |
| Point GBDT + safety | 1925.55 | 1845.60 | 79.96 | 0.961 | 23.66 |

Relative to fixed protection-period quantile replenishment:

```text
cost reduction = 5.02%
bootstrap 95% CI = [4.50%, 5.49%]
```

Interpretation:

- The point forecast with safety stock achieves the highest fill rate but over-stocks heavily.
- The calibrated method reduces holding cost enough to lower total cost, at the price of slightly lower fill rate.
- The result supports the report's main claim: forecast quality and replenishment utility are not identical.

## M5 Ablation

Key rows:

| Ablation | Model | Total cost |
|---|---|---:|
| full | calibrated | 959.11 |
| full | fixed | 1009.83 |
| no tau adaptation | calibrated | 1009.83 |
| strong tau regularization | calibrated | 959.00 |
| short lags only | calibrated | 1024.86 |
| short lags only | fixed | 1090.56 |

Interpretation:

- Removing tau adaptation collapses the method to fixed quantile replenishment.
- The calibration gain remains with stronger regularization.
- Shorter feature history weakens the whole system, but calibration still improves over fixed quantile under that feature set.

## M5 Sensitivity

The calibrated method improves fixed quantile in most high-cost or longer-protection scenarios:

- `L=7, p=5`: calibrated total cost 959.11 vs fixed 1009.83.
- `L=7, p=9`: calibrated 1080.84 vs fixed 1211.40.
- `L=14, p=5`: calibrated 1317.92 vs fixed 1381.64.
- `L=14, p=9`: calibrated 1368.47 vs fixed 1626.45.

Cases where calibrated equals or underperforms fixed:

- `L=3, p=3`: identical.
- `L=7, p=3`: identical.
- `L=14, p=3`: identical.
- `L=3, p=5`: calibrated is worse than fixed.

Interpretation:

- The calibration layer is most useful when the stockout penalty is high or the protection period is longer.
- For short lead time and moderate costs, fixed critical-fractile replenishment can already be sufficient.

## Synthetic Layer

Synthetic experiments demonstrate the full classical-baseline layer and the accuracy-cost mismatch. In the LightGBM-backed synthetic run, the calibrated method over-adjusted and increased cost relative to fixed quantile. This is useful as a limitation:

> Inventory-aware calibration should be selected and tuned using a validation inventory objective; it is not a universal improvement independent of demand regime.

This limitation should be included in the report discussion.
