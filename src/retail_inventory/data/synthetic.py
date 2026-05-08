import numpy as np
import pandas as pd


def _seasonal_pattern(t):
    weekly = 1.0 + 0.25 * np.sin(2 * np.pi * t / 7.0)
    monthly = 1.0 + 0.12 * np.sin(2 * np.pi * t / 30.0)
    return weekly * monthly


def generate_synthetic_panel(n_series=120, n_days=420, seed=2026):
    """Generate retail-like demand with smooth, promo, intermittent, and lumpy regimes."""
    rng = np.random.RandomState(seed)
    rows = []
    regimes = ["smooth", "promo", "intermittent", "lumpy"]
    for i in range(n_series):
        regime = regimes[i % len(regimes)]
        base = rng.uniform(4.0, 28.0)
        price = rng.uniform(2.0, 25.0)
        category = "cat_%d" % (i % 5)
        store = "store_%d" % (i % 8)
        item = "item_%03d" % i
        last_positive = -1
        for t in range(n_days):
            promo = int((regime == "promo" and t % 42 in (4, 5, 6, 7)) or rng.rand() < 0.015)
            event = int(t % 91 in (0, 1, 2))
            lam = base * _seasonal_pattern(t)
            if promo:
                lam *= rng.uniform(1.5, 2.6)
            if event:
                lam *= rng.uniform(1.1, 1.5)

            if regime == "smooth":
                demand = max(0, rng.normal(lam, 0.20 * lam + 1.0))
            elif regime == "promo":
                demand = rng.poisson(max(lam, 0.1))
            elif regime == "intermittent":
                occur_prob = min(0.75, 0.12 + lam / 95.0)
                demand = 0 if rng.rand() > occur_prob else rng.gamma(2.0, max(lam / 2.0, 0.1))
            else:
                occur_prob = min(0.85, 0.20 + lam / 80.0)
                if rng.rand() > occur_prob:
                    demand = 0
                else:
                    demand = rng.negative_binomial(3, 3.0 / (3.0 + max(lam, 0.1)))

            demand = int(round(max(0, demand)))
            if demand > 0:
                last_positive = t
            rows.append(
                {
                    "series_id": "%s_%s" % (store, item),
                    "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=t),
                    "t": t,
                    "demand": demand,
                    "price": price * (0.92 if promo else 1.0),
                    "promo": promo,
                    "event": event,
                    "item_id": item,
                    "store_id": store,
                    "category_id": category,
                    "state_id": "synthetic",
                    "regime": regime,
                    "days_since_positive": 999 if last_positive < 0 else t - last_positive,
                }
            )
    return pd.DataFrame(rows)
