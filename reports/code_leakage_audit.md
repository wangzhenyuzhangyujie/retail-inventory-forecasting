# Code Leakage Audit

Audit target: current retail inventory forecasting experiments.

Main question: whether the experiment code hard-codes favorable results, trains on the test target, or leaks test labels into model fitting.

## Verdict

No direct result backdoor or direct test-target training was found.

The main supervised models are trained as follows:

- Point GBDT: fitted on `X_train, y_train`.
- Quantile GBDT: fitted on `X_train, y_train`.
- Cost-aware calibrator: fitted on `X_val, q_val, y_val`.
- Test labels `y_test` are used for metrics and inventory simulation evaluation, not for model fitting.

However, there are several rigor issues that should be documented or fixed before final reporting.

## What Is Clean

### No target columns in feature matrix

`encode_features` drops:

- `target_pp`
- `target_horizon`
- all `y_h*` future daily labels

The audit script confirmed:

- M5: feature matrix contains no `y_h*`, `target_pp`, or `target_horizon`.
- Store Item: feature matrix contains no `y_h*`, `target_pp`, or `target_horizon`.

### Chronological split is non-overlapping

For M5 subset:

- Train origin range: 56 to 245
- Validation origin range: 252 to 301
- Test origin range: 308 to 385

For Store Item:

- Train origin range: 56 to 1652
- Validation origin range: 1659 to 1708
- Test origin range: 1715 to 1792

There is no train/validation/test origin overlap.

### No hard-coded result values

The code does hard-code model names for reporting and bootstrap comparison, but no total-cost values, reductions, or winning outputs are hard-coded into the experiment pipeline.

## Rigor Risks

### 1. Same-day timing convention

Feature construction defines `lag_1` as demand at the origin day:

```python
row["lag_%d" % lag] = float(demand[idx - lag + 1])
```

Therefore `lag_1 = demand[idx]`.

The target is future demand:

```python
future = demand[idx + 1 : idx + horizon + 1]
```

This is valid if the decision is made at the end of day `t`, after observing demand at day `t`.

But the inventory simulator currently places review orders before consuming demand on the same day:

```python
if (t - s_t) % review_period == 0:
    place order
demand = float(day["demand"])
sales = min(on_hand, demand)
```

So the implementation mixes two timing conventions:

- features assume end-of-day information is available;
- simulator processes order before same-day demand.

This is not direct future-label leakage for the protection-period target, but it is a temporal convention issue. For a stricter implementation, either:

- define decisions as end-of-day and simulate same-day demand before placing the review order; or
- define decisions as start-of-day and shift lag features so `lag_1 = demand[t-1]`.

### 2. Rolling test-period history is used in later test origins

The experiment is a rolling-origin evaluation. Later test origins use lag/rolling features computed from earlier test-period observed demand.

This is acceptable for sequential replenishment if the report explicitly says:

> At each rolling review date, all demand observed up to that date is available.

It would not be acceptable for a single-shot Kaggle submission where all future test labels are hidden.

### 3. One-hot columns are aligned using train + validation + test

`encode_features` concatenates train, validation, and test before one-hot encoding categorical variables:

```python
all_df = pd.concat([train_df] + list(other_dfs), axis=0, ignore_index=True)
```

Targets are dropped before encoding, so this is not label leakage. But it is a mildly transductive preprocessing choice because the set of test categories is known at encoding time.

For full strictness, fit category levels on train and use `reindex` to align validation/test columns.

### 4. M5 price imputation uses full selected panel median

M5 missing prices are filled with a per-series median computed across the selected panel:

```python
long_df["price"] = long_df["sell_price"].fillna(long_df.groupby("series_id")["sell_price"].transform("median"))
```

This uses future covariate values for imputation. It is not demand-label leakage, but a stricter version should compute price imputation statistics from the training period only.

## Recommended Fixes Before Final Report

High priority:

- Clarify the experiment as rolling-origin replenishment, not Kaggle hidden-test evaluation.
- Fix or explicitly state the order-timing convention.

Medium priority:

- Change one-hot encoding to train-fitted columns only.
- Compute price-imputation medians from the training portion only.

Optional:

- Add a strict no-same-day-lag sensitivity run to show the conclusions are robust.

## Bottom Line

The reported results are not produced by direct hard-coding or direct test-target training. The core ordering of methods is credible under the current rolling-origin setup.

For maximum defensibility, the final report should avoid claiming Kaggle-style hidden-test evaluation and should describe the evaluation as:

> local rolling-origin replenishment simulation using observed demand up to each review date.
