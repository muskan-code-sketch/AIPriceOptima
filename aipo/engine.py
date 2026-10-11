"""Pricing maths: evidence from the data + transparent what-if scenarios.

No Streamlit imports here so everything can be unit-tested.

Scenario model (constant price elasticity)
------------------------------------------
For every historical order:

    new_units  = units * (new_effective_price / old_effective_price) ** (-elasticity)
    net_sales  = effective_price * units
    revenue    = TotalAmount scaled by (new net sales / old net sales)   # keeps fees proportional
    profit     = net_sales - unit_cost * units,  unit_cost = cost_pct * list price

The elasticity is an *assumption* (the dataset itself shows no price-demand link),
so every result is a scenario, not a forecast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- evidence
def measure_price_sensitivity(df: pd.DataFrame) -> dict | None:
    """Log-log regression of units on effective price (slope = -elasticity)."""
    d = df[(df["EffectivePrice"] > 0) & (df["Quantity"] > 0)]
    if len(d) < 30:
        return None
    res = stats.linregress(np.log(d["EffectivePrice"].to_numpy()), np.log(d["Quantity"].to_numpy()))
    half = 1.96 * res.stderr
    return {
        "elasticity": float(-res.slope),
        "ci_low": float(-(res.slope + half)),
        "ci_high": float(-(res.slope - half)),
        "p_value": float(res.pvalue),
        "n": int(len(d)),
        "significant": bool(res.pvalue < 0.05),
    }


def discount_lift(df: pd.DataFrame) -> dict | None:
    """Do discounted orders contain more units than full-price orders?"""
    disc = df.loc[df["Discount"] > 0, "Quantity"]
    full = df.loc[df["Discount"] == 0, "Quantity"]
    if len(disc) < 10 or len(full) < 10:
        return None
    test = stats.ttest_ind(disc, full, equal_var=False)
    diff = disc.mean() - full.mean()
    se = np.sqrt(disc.var(ddof=1) / len(disc) + full.var(ddof=1) / len(full))
    return {
        "units_discounted": float(disc.mean()),
        "units_full_price": float(full.mean()),
        "lift_pct": float(diff / full.mean() * 100),
        "ci_low_pct": float((diff - 1.96 * se) / full.mean() * 100),
        "ci_high_pct": float((diff + 1.96 * se) / full.mean() * 100),
        "p_value": float(test.pvalue),
        "n_discounted": int(len(disc)),
        "n_full_price": int(len(full)),
        "significant": bool(test.pvalue < 0.05),
    }


def units_by_discount_band(df: pd.DataFrame) -> pd.DataFrame:
    """Average units per order for each discount level, with a 95% interval."""
    g = df.assign(Band=(df["Discount"] * 100).round().astype(int)).groupby("Band")["Quantity"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci"] = 1.96 * out["std"].fillna(0) / np.sqrt(out["count"])
    out["Label"] = out["Band"].astype(str) + "%"
    return out


def discount_cap_analysis(df: pd.DataFrame, cap: float, volume_loss: float) -> dict:
    """Revenue effect of capping every discount at `cap`.

    volume_loss: share of units lost on the orders whose discount is reduced.
    break_even_loss: loss share at which the cap stops paying off.
    """
    price, disc, qty = df["UnitPrice"].to_numpy(), df["Discount"].to_numpy(), df["Quantity"].to_numpy()
    hit = disc > cap + 1e-9
    ep_old = price * (1 - disc)
    ep_new = np.where(hit, price * (1 - cap), ep_old)
    qty_new = np.where(hit, qty * (1 - volume_loss), qty)
    net_old, net_new = ep_old * qty, ep_new * qty_new
    rev_old = df["TotalAmount"].to_numpy()
    rev_new = rev_old * net_new / net_old

    gain_if_no_loss = float(((ep_new - ep_old) * qty)[hit].sum())
    old_hit = float(net_old[hit].sum())
    new_hit_no_loss = float((ep_new * qty)[hit].sum())
    break_even = 1 - old_hit / new_hit_no_loss if new_hit_no_loss > 0 else 0.0
    return {
        "orders_affected": int(hit.sum()),
        "share_affected": float(hit.mean()) if len(hit) else 0.0,
        "revenue_now": float(rev_old.sum()),
        "revenue_after": float(rev_new.sum()),
        "revenue_delta": float(rev_new.sum() - rev_old.sum()),
        "gain_if_no_loss": gain_if_no_loss,
        "break_even_loss": float(max(break_even, 0.0)),
    }


# --------------------------------------------------------------------------- scenarios
def _scenario(df: pd.DataFrame, price_change, elasticity: float, cost_pct: float, discount_cap):
    price = df["UnitPrice"].to_numpy()
    disc = df["Discount"].to_numpy()
    qty = df["Quantity"].to_numpy()
    new_disc = disc if discount_cap is None else np.minimum(disc, discount_cap)

    ep0 = price * (1 - disc)
    ep1 = price * (1 + price_change) * (1 - new_disc)
    qty1 = qty * (ep1 / ep0) ** (-elasticity)

    net0, net1 = ep0 * qty, ep1 * qty1
    rev0 = df["TotalAmount"].to_numpy()
    rev1 = rev0 * net1 / net0
    unit_cost = cost_pct * price
    prof0, prof1 = net0 - unit_cost * qty, net1 - unit_cost * qty1
    return rev0, rev1, qty, qty1, prof0, prof1


def simulate(df, price_change, elasticity, cost_pct, discount_cap=None) -> dict:
    """Totals for one price move applied to all rows of `df`."""
    rev0, rev1, q0, q1, p0, p1 = _scenario(df, price_change, elasticity, cost_pct, discount_cap)
    return {
        "revenue_base": float(rev0.sum()), "revenue_new": float(rev1.sum()),
        "units_base": float(q0.sum()), "units_new": float(q1.sum()),
        "profit_base": float(p0.sum()), "profit_new": float(p1.sum()),
    }


def price_curve(df, deltas, elasticity, cost_pct, discount_cap=None) -> pd.DataFrame:
    """Revenue / units / profit change (%) for each candidate price move."""
    rows = []
    for dlt in deltas:
        s = simulate(df, dlt, elasticity, cost_pct, discount_cap)
        rows.append({
            "price_change": dlt,
            "revenue": s["revenue_new"], "units": s["units_new"], "profit": s["profit_new"],
            "revenue_pct": (s["revenue_new"] / s["revenue_base"] - 1) * 100,
            "units_pct": (s["units_new"] / s["units_base"] - 1) * 100,
            "profit_pct": (s["profit_new"] / s["profit_base"] - 1) * 100 if s["profit_base"] else np.nan,
        })
    return pd.DataFrame(rows)


def optimize_by_group(df, group_col, guardrail, elasticity, cost_pct, objective="Profit",
                      discount_cap=None, step=0.01) -> pd.DataFrame:
    """Best price move inside +/-guardrail for every group (e.g. product)."""
    deltas = np.round(np.arange(-guardrail, guardrail + step / 2, step), 4)
    codes, labels = pd.factorize(df[group_col], sort=True)
    k = len(labels)

    best_val = np.full(k, -np.inf)
    best_delta = np.zeros(k)
    best = {m: np.zeros(k) for m in ("rev", "prof", "units")}
    for dlt in deltas:
        _, rev1, _, q1, _, p1 = _scenario(df, dlt, elasticity, cost_pct, discount_cap)
        agg = {"rev": np.bincount(codes, rev1, k), "prof": np.bincount(codes, p1, k),
               "units": np.bincount(codes, q1, k)}
        val = agg["prof"] if objective == "Profit" else agg["rev"]
        better = val > best_val + 1e-9
        best_val = np.where(better, val, best_val)
        best_delta = np.where(better, dlt, best_delta)
        for m in best:
            best[m] = np.where(better, agg[m], best[m])

    rev0, _, q0, _, p0, _ = _scenario(df, 0.0, elasticity, cost_pct, None)
    base = {"rev": np.bincount(codes, rev0, k), "prof": np.bincount(codes, p0, k),
            "units": np.bincount(codes, q0, k)}
    return pd.DataFrame({
        group_col: labels,
        "price_change": best_delta,
        "revenue_base": base["rev"], "revenue_new": best["rev"],
        "profit_base": base["prof"], "profit_new": best["prof"],
        "units_base": base["units"], "units_new": best["units"],
    })
