"""Tab 4 - Recommendations: what should I do, product by product?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from . import engine, ui
from .data import Settings

MIN_GAIN_PCT = 2.0  # a product must improve by at least this much to be worth changing


@st.cache_data(show_spinner="Searching the best price for every product...")
def _recommend(df: pd.DataFrame, s: Settings, objective: str, cap) -> pd.DataFrame:
    r = engine.optimize_by_group(df, "ProductID", s.guardrail, s.elasticity, s.cost_pct, objective, cap)
    info = df.groupby("ProductID").agg(Orders=("OrderID", "count"), AvgPrice=("UnitPrice", "mean"),
                                       AvgDiscount=("Discount", "mean")).reset_index()
    r = r.merge(info, on="ProductID")
    r["rev_pct"] = (r["revenue_new"] / r["revenue_base"] - 1) * 100
    r["profit_pct"] = np.where(r["profit_base"] > 0, (r["profit_new"] / r["profit_base"] - 1) * 100, 0.0)
    r["profit_gain"] = r["profit_new"] - r["profit_base"]

    gain = r["profit_pct"] if objective == "Profit" else r["rev_pct"]
    worth = gain >= MIN_GAIN_PCT
    moves = r["price_change"].abs() >= 0.01
    r["Action"] = np.select(
        [worth & moves & (r["price_change"] > 0), worth & moves & (r["price_change"] < 0), worth & ~moves],
        ["Raise price", "Lower price", "Cap discounts"], default="Hold")
    active = r["Action"] != "Hold"
    r["Recommended price"] = np.where(active, r["AvgPrice"] * (1 + r["price_change"]), r["AvgPrice"])
    r["PriceMove"] = np.where(active, r["price_change"] * 100, 0.0)
    r["DiscountRule"] = np.where(active & (cap is not None), f"max {0 if cap is None else cap * 100:.0f}%", "unchanged")
    # products we leave alone keep today's numbers
    for new, base in (("revenue_new", "revenue_base"), ("profit_new", "profit_base"), ("units_new", "units_base")):
        r[new] = np.where(active, r[new], r[base])
    r["rev_pct"] = np.where(active, r["rev_pct"], 0.0)
    r["profit_pct"] = np.where(active, r["profit_pct"], 0.0)
    r["profit_gain"] = np.where(active, r["profit_gain"], 0.0)
    return r


def render(df: pd.DataFrame, s: Settings) -> None:
    ui.callout(
        f"Built on your sidebar assumptions: customers' price sensitivity <b>{s.elasticity:.1f}</b>, product cost "
        f"<b>{s.cost_pct * 100:.0f}%</b> of list price, price moves limited to <b>&plusmn;{s.guardrail * 100:.0f}%</b>. "
        "Change them in the sidebar and every number here updates. The result is sensitive to them - "
        "see <i>How fragile is this plan?</i> below.",
        "info", "These are scenarios, not forecasts")

    a, b = st.columns(2, gap="large")
    objective = a.radio("Optimise for", ["Profit", "Revenue"], horizontal=True, key="rec_obj")
    policy = b.radio("Discounts", ["Keep as they are", "Cap at..."], index=1, horizontal=True, key="rec_policy",
                     help="Your data shows discounts add no measurable volume, so capping them is the "
                          "most evidence-backed lever (see Discount Insights).")
    cap = None
    if policy == "Cap at...":
        cap = b.slider("Max discount", 0, 30, 5, step=5, format="%d%%", key="rec_cap") / 100

    r = _recommend(df, s, objective, cap)
    n, n_change = len(r), int((r["Action"] != "Hold").sum())
    rev0, rev1 = r["revenue_base"].sum(), r["revenue_new"].sum()
    pr0, pr1 = r["profit_base"].sum(), r["profit_new"].sum()
    counts = r["Action"].value_counts()

    m = st.columns(4)
    m[0].metric("Products to change", f"{n_change} of {n}", border=True)
    m[1].metric("Raise / lower price", f"{counts.get('Raise price', 0)} / {counts.get('Lower price', 0)}", border=True)
    m[2].metric("Expected revenue", ui.cmoney(rev1), f"{(rev1 / rev0 - 1) * 100:+.1f}%", border=True)
    m[3].metric("Expected profit", ui.cmoney(pr1), f"{(pr1 / pr0 - 1) * 100:+.1f}%" if pr0 else None, border=True)

    if n_change == 0:
        ui.callout(
            f"No product gains {MIN_GAIN_PCT:.0f}% or more from a change, so the plan is to <b>hold steady</b>: "
            f"today's prices are already close to the best point for {objective.lower()} under these assumptions. "
            "Try a different price sensitivity or product cost in the sidebar to see how that changes.",
            "good", "Hold steady")
    else:
        key_gain = (pr1 / pr0 if objective == "Profit" else rev1 / rev0) - 1
        ui.callout(
            f"Changing <b>{n_change}</b> of {n} products moves {objective.lower()} by <b>{key_gain * 100:+.1f}%</b> "
            f"(revenue {(rev1 / rev0 - 1) * 100:+.1f}%, profit {(pr1 / pr0 - 1) * 100:+.1f}%). "
            f"Products that would improve by less than {MIN_GAIN_PCT:.0f}% are left at <i>Hold</i> - "
            "too small to be worth the risk.",
            "good" if key_gain >= 0 else "warn", "Bottom line")

    # ---- chart ------------------------------------------------------------------------
    ui.section("Expected profit change by product",
               "The 20 products with the biggest expected gain (products on Hold are not shown)")
    chart = r[r["Action"] != "Hold"].nlargest(20, "profit_gain").sort_values("profit_gain")
    if chart.empty:
        st.info("No product is marked for change, so there is nothing to chart.")
    else:
        fig = px.bar(chart, x="profit_gain", y="ProductID", orientation="h", color="Action",
                     color_discrete_map={"Raise price": ui.TEAL, "Lower price": ui.AMBER,
                                         "Cap discounts": ui.INDIGO, "Hold": "#CBD5E1"},
                     category_orders={"ProductID": list(chart["ProductID"])})
        fig.update_traces(hovertemplate="%{y}<br>profit change $%{x:,.0f}<extra></extra>")
        fig.update_xaxes(title="Profit change ($)", tickprefix="$", tickformat="~s")
        fig.update_yaxes(title=None, type="category", dtick=1)
        ui.show(fig, int(max(240, 24 * len(chart) + 100)), legend=True)

    # ---- table ------------------------------------------------------------------------
    ui.section("Product-by-product plan")
    show = r.assign(AvgDiscount=r["AvgDiscount"] * 100).sort_values("profit_gain", ascending=False)[
        ["ProductID", "Action", "AvgPrice", "Recommended price", "PriceMove", "DiscountRule", "AvgDiscount",
         "rev_pct", "profit_pct", "profit_gain", "Orders"]]
    st.dataframe(
        show, hide_index=True, width="stretch", height=420,
        column_config={
            "ProductID": "Product", "Action": st.column_config.TextColumn("Action", width="small"),
            "AvgPrice": st.column_config.NumberColumn("Avg price now", format="$%.2f"),
            "Recommended price": st.column_config.NumberColumn("Recommended avg price", format="$%.2f"),
            "PriceMove": st.column_config.NumberColumn("Price move", format="%+.0f%%"),
            "DiscountRule": st.column_config.TextColumn("Disc. cap", width="small"),
            "AvgDiscount": st.column_config.NumberColumn("Avg discount now", format="%.1f%%"),
            "rev_pct": st.column_config.NumberColumn("Revenue", format="%+.1f%%"),
            "profit_pct": st.column_config.NumberColumn("Profit", format="%+.1f%%"),
            "profit_gain": st.column_config.NumberColumn("Profit change ($)", format="%+,.0f"),
            "Orders": st.column_config.NumberColumn(format="%,d"),
        })
    st.download_button("Download recommendations (CSV)", show.to_csv(index=False).encode(),
                       "price_recommendations.csv", "text/csv", key="rec_dl")

    # ---- robustness -------------------------------------------------------------------
    st.divider()
    ui.section("How fragile is this plan?",
               "The same recommended prices, if customers react differently from what you assumed")
    by_prod = r.set_index("ProductID")
    active_row = df["ProductID"].map(by_prod["Action"] != "Hold").to_numpy(dtype=bool)
    move_row = df["ProductID"].map(by_prod["PriceMove"] / 100).to_numpy(dtype=float)
    cap_row = None if cap is None else np.where(active_row, cap, 1.0)
    rows = []
    for label, e in (("Less sensitive", max(s.elasticity - 0.5, 0.0)), ("As assumed", s.elasticity),
                     ("More sensitive", s.elasticity + 0.5)):
        sim = engine.simulate(df, move_row, e, s.cost_pct, cap_row)
        rows.append({"Customers are": label, "Sensitivity": e,
                     "Revenue": (sim["revenue_new"] / sim["revenue_base"] - 1) * 100,
                     "Profit": (sim["profit_new"] / sim["profit_base"] - 1) * 100})
    rob = pd.DataFrame(rows)
    st.dataframe(rob, hide_index=True, width="stretch", column_config={
        "Sensitivity": st.column_config.NumberColumn(format="%.1f"),
        "Revenue": st.column_config.NumberColumn("Revenue change", format="%+.1f%%"),
        "Profit": st.column_config.NumberColumn("Profit change", format="%+.1f%%")})
    if n_change == 0:
        st.caption("Nothing to stress-test: the plan changes no prices.")
    elif rob["Profit"].min() < 0:
        ui.callout("If customers turn out to be more price-sensitive than assumed, this plan <b>loses profit</b>. "
                   "Run it as a pilot on a few products and measure the real response before rolling out.",
                   "warn", "Run it as an experiment")
    else:
        ui.callout("The plan stays profitable across the sensitivity range shown, but real-world response is unknown - "
                   "a small pilot is still the safest first step.", "info", "Still validate with a pilot")
