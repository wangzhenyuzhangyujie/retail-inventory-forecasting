from collections import defaultdict

import numpy as np
import pandas as pd


def initial_inventory_from_history(panel, protection_period, quantile=0.90):
    levels = {}
    for sid, g in panel.groupby("series_id", sort=False):
        y = g.sort_values("t")["demand"].values.astype(float)
        sums = []
        for i in range(0, max(0, len(y) - protection_period)):
            sums.append(y[i : i + protection_period].sum())
        levels[sid] = float(np.quantile(sums, quantile)) if sums else float(np.mean(y[-protection_period:]))
    return levels


def simulate_inventory(
    panel,
    policy_table,
    lead_time=7,
    review_period=7,
    holding_cost=1.0,
    stockout_cost=5.0,
    fixed_order_cost=0.0,
    initial_levels=None,
    start_t=None,
    end_t=None,
    warmup_days=28,
):
    panel = panel.sort_values(["series_id", "t"]).copy()
    policy = {}
    for _, row in policy_table.iterrows():
        policy[(row["series_id"], int(row["origin_t"]))] = float(row["order_up_to"])
    model = policy_table["model"].iloc[0] if len(policy_table) else "unknown"
    rows = []
    summaries = []
    for sid, g in panel.groupby("series_id", sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        if start_t is None:
            s_t = int(policy_table[policy_table["series_id"] == sid]["origin_t"].min())
        else:
            s_t = int(start_t)
        e_t = int(end_t if end_t is not None else g["t"].max())
        on_hand = float(initial_levels.get(sid, 0.0) if initial_levels else 0.0)
        pipeline = defaultdict(float)
        total_demand = total_sales = 0.0
        stockout_days = service_hits = review_count = 0
        cost_h = cost_s = cost_o = 0.0
        onhand_sum = 0.0
        eval_days = 0
        for _, day in g[(g["t"] >= s_t) & (g["t"] <= e_t)].iterrows():
            t = int(day["t"])
            on_hand += pipeline.pop(t, 0.0)
            if (t - s_t) % review_period == 0:
                s_level = policy.get((sid, t), None)
                if s_level is not None:
                    ip = on_hand + sum(pipeline.values())
                    q = max(0.0, s_level - ip)
                    if q > 1e-9:
                        pipeline[t + lead_time] += q
                        if t >= s_t + warmup_days:
                            cost_o += fixed_order_cost
                    review_count += 1
                else:
                    q = 0.0
            else:
                q = 0.0
            demand = float(day["demand"])
            sales = min(on_hand, demand)
            lost = demand - sales
            on_hand -= sales
            if t >= s_t + warmup_days:
                total_demand += demand
                total_sales += sales
                stockout_days += int(lost > 1e-9)
                service_hits += int(lost <= 1e-9)
                cost_h += holding_cost * on_hand
                cost_s += stockout_cost * lost
                onhand_sum += on_hand
                eval_days += 1
                rows.append(
                    {
                        "model": model,
                        "series_id": sid,
                        "t": t,
                        "demand": demand,
                        "sales": sales,
                        "lost_sales": lost,
                        "on_hand": on_hand,
                        "order_qty": q,
                        "holding_cost": holding_cost * on_hand,
                        "stockout_cost": stockout_cost * lost,
                        "ordering_cost": fixed_order_cost if q > 1e-9 else 0.0,
                    }
                )
        total_cost = cost_h + cost_s + cost_o
        summaries.append(
            {
                "model": model,
                "series_id": sid,
                "total_cost": total_cost,
                "holding_cost": cost_h,
                "stockout_cost": cost_s,
                "ordering_cost": cost_o,
                "fill_rate": total_sales / total_demand if total_demand > 0 else 1.0,
                "cycle_service_level": service_hits / max(eval_days, 1),
                "stockout_rate": stockout_days / max(eval_days, 1),
                "average_on_hand": onhand_sum / max(eval_days, 1),
                "total_demand": total_demand,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries)
