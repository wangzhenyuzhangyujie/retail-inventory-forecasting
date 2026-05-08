# Experiment Execution Plan

## Research Question

Does lower sales-forecast error necessarily lead to lower replenishment cost?

The project tests this through a unified pipeline:

```text
demand history and covariates
-> protection-period quantile forecasting
-> periodic-review order-up-to policy
-> rolling lost-sales inventory simulation
-> forecast metrics + inventory KPIs
```

## Main Method

The proposed method is:

```text
Protection-period quantile forecasting
+ cost-aware decision calibration
```

For review period `R` and lead time `L`, the target used by replenishment is:

```text
D_pp(t) = sum_{h=1}^{L+R} demand_{t+h}
```

The model predicts quantiles of `D_pp(t)` at:

```text
alpha = {0.5, 0.7, 0.8, 0.9, 0.95, 0.99}
```

The fixed baseline uses:

```text
tau_base = stockout_cost / (stockout_cost + holding_cost)
S_t = Q_pp(tau_base)
```

The calibrated method learns an effective decision quantile:

```text
tau_t = clip(tau_base + epsilon * tanh(g_phi(z_t)), tau_floor, 0.99)
S_t = Interp({Q_pp(alpha)}, tau_t)
q_t = max(0, S_t - inventory_position_t)
```

The calibration layer is trained with a newsvendor-style inventory surrogate:

```text
L_inventory = h * max(S_t - D_pp(t), 0) + p * max(D_pp(t) - S_t, 0)
```

This should be described as a forecast-to-decision calibration layer, not as a new theory of critical fractiles.

## Implemented Experiments

### Layer 1: Classical Simulation

Command:

```bash
python experiments/run_layer1_synthetic.py --config configs/default.yaml
```

Algorithms:

- Seasonal naive
- Moving average
- Exponential smoothing
- Croston
- GBDT point forecast with safety stock
- Protection-period quantile GBDT
- Protection-period quantile GBDT + calibration

### Layer 2 and 3: M5 Main Experiment

Command:

```bash
python experiments/download_m5.py
python experiments/run_m5_full.py --config configs/default.yaml --n-series 1000
```

Dataset:

- M5 bottom-level SKU-store series subset
- 1000 deterministic sampled series
- Last 420 days used for local rolling experiments

### Ablation

Command:

```bash
python experiments/run_ablation.py --dataset m5 --n-series 1000
```

Variants:

- Full method
- No tau adaptation
- Strong tau regularization
- Short lag/rolling features only

### Sensitivity

Command:

```bash
python experiments/run_sensitivity.py --dataset m5 --n-series 1000
```

Scenarios:

- `L in {3, 7, 14}`
- `stockout_cost in {3, 5, 9}`
- `R = 7`
- `holding_cost = 1`
- `K = 0`

## Main Output Files

Tables:

- `outputs/tables/layer1_synthetic_forecast_metrics.csv`
- `outputs/tables/layer1_synthetic_inventory_metrics.csv`
- `outputs/tables/m5_n1000_forecast_metrics.csv`
- `outputs/tables/m5_n1000_inventory_metrics.csv`
- `outputs/tables/m5_n1000_bootstrap.csv`
- `outputs/tables/ablation_m5_summary.csv`
- `outputs/tables/sensitivity_m5_summary.csv`

Figures:

- `outputs/figures/layer1_synthetic_accuracy_cost_scatter.png`
- `outputs/figures/layer1_synthetic_cost_breakdown.png`
- `outputs/figures/m5_n1000_accuracy_cost_scatter.png`
- `outputs/figures/m5_n1000_cost_breakdown.png`
- sensitivity and ablation cost/accuracy plots under `outputs/figures/`

## Report Positioning

The innovation should be stated as:

> This work directly predicts protection-period demand quantiles and learns a bounded inventory-cost-aware calibration layer to reduce the gap between probability forecasting and rolling replenishment decisions.

Avoid claiming:

- first cost-aware inventory model
- first adaptive critical fractile
- first decision-focused inventory method

The safest novelty claim is:

> a lightweight, interpretable forecast-to-decision calibration framework for multi-SKU, multi-store periodic-review replenishment on retail panel data.
