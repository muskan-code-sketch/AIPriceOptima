"""Tab 3 - Price Simulator: what happens if I change the price?"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import engine, ui
from .data import Settings


def render(df: pd.DataFrame, s: Settings) -> None:
    left, right = st.columns([1, 2.2], gap="large")

    with left:
        ui.section("1. Choose a scenario")
        products = ["All products in selection"] + sorted(df["ProductID"].unique())
        scope = st.selectbox("Apply to", products, key="sim_scope")
        g = int(round(s.guardrail * 100))
        move = st.slider("Change list price by", -g, g, min(5, g), step=1, format="%+d%%", key="sim_move") / 100
        policy = st.radio("Discounts", ["Keep as they are", "Cap at..."], horizontal=True, key="sim_policy")
        cap = None
        if policy == "Cap at...":
            cap = st.slider("Max discount", 0, 30, 5, step=5, format="%d%%", key="sim_cap") / 100
        objective = st.radio("Find the best price for", ["Profit", "Revenue"], horizontal=True, key="sim_obj",
                             help="Revenue is only maximised at the edge of your price range unless "
                                  "customers' price sensitivity is exactly 1. Profit has a real sweet spot.")
        st.caption(f"Assumptions (sidebar): price sensitivity **{s.elasticity:.1f}**, product cost "
                   f"**{s.cost_pct * 100:.0f}%** of list price.")

    data = df if scope == products[0] else df[df["ProductID"] == scope]
    sim = engine.simulate(data, move, s.elasticity, s.cost_pct, cap)
    deltas = np.round(np.arange(-s.guardrail, s.guardrail + 0.0049, 0.01), 4)
    curve = engine.price_curve(data, deltas, s.elasticity, s.cost_pct, cap)
    key = "profit_pct" if objective == "Profit" else "revenue_pct"
    best = curve.loc[curve[key].idxmax()]

    with right:
        ui.section("2. See the impact", f"{len(data):,} orders - {scope.lower() if scope != products[0] else 'all products'}")
        rv, un, pf = sim["revenue_new"], sim["units_new"], sim["profit_new"]
        m = st.columns(3)
        m[0].metric("Revenue", ui.cmoney(rv), f"{ui.cmoney(rv - sim['revenue_base'])} ({(rv / sim['revenue_base'] - 1) * 100:+.1f}%)", border=True)
        m[1].metric("Units sold", f"{un:,.0f}", f"{un - sim['units_base']:+,.0f} ({(un / sim['units_base'] - 1) * 100:+.1f}%)", border=True)
        m[2].metric("Profit", ui.cmoney(pf), f"{ui.cmoney(pf - sim['profit_base'])} ({(pf / sim['profit_base'] - 1) * 100:+.1f}%)" if sim["profit_base"] else None, border=True)

        fig = go.Figure()
        x = curve["price_change"] * 100
        fig.add_scatter(x=x, y=curve["revenue_pct"], name="Revenue", line=dict(color=ui.INDIGO, width=3))
        fig.add_scatter(x=x, y=curve["profit_pct"], name="Profit", line=dict(color=ui.TEAL, width=3))
        fig.add_scatter(x=x, y=curve["units_pct"], name="Units", line=dict(color=ui.SLATE, width=2, dash="dot"))
        fig.add_hline(y=0, line_color="#CBD5E1")
        fig.add_vline(x=move * 100, line_dash="dash", line_color=ui.AMBER,
                      annotation_text="your scenario", annotation_position="bottom right")
        fig.add_scatter(x=[best["price_change"] * 100], y=[best[key]], mode="markers",
                        marker=dict(size=13, color=ui.ROSE, symbol="star", line=dict(color="white", width=1)),
                        name=f"Best for {objective.lower()}",
                        hovertemplate=f"Best for {objective.lower()}: %{{x:+.0f}}% price<extra></extra>")
        fig.update_xaxes(title="Change in list price (%)", ticksuffix="%")
        fig.update_yaxes(title="Change vs today (%)", ticksuffix="%")
        ui.show(fig, 380, legend=True)

        # ---- plain-language read-out ---------------------------------------------------
        rev_pct = (rv / sim["revenue_base"] - 1) * 100
        prof_pct = (pf / sim["profit_base"] - 1) * 100 if sim["profit_base"] else 0
        at_edge = abs(abs(best["price_change"]) - s.guardrail) < 1e-9
        verdict = "good" if prof_pct > 0.5 else ("bad" if prof_pct < -0.5 else "info")
        text = (f"A <b>{move * 100:+.0f}%</b> price move changes revenue by <b>{rev_pct:+.1f}%</b> and profit by "
                f"<b>{prof_pct:+.1f}%</b>.")
        if abs(best["price_change"]) < 1e-9:
            text += f" Today's prices are already the best point for {objective.lower()} under these assumptions."
        else:
            text += (f" The best price for {objective.lower()} is <b>{best['price_change'] * 100:+.0f}%</b> "
                     f"({objective.lower()} {best[key]:+.1f}%).")
        if at_edge:
            text += " That sits at the edge of your price range - widen it in the sidebar to see if it keeps improving."
        if objective == "Profit" and best["revenue_pct"] < -2:
            text += (f" Note it gives up <b>{abs(best['revenue_pct']):.1f}%</b> of revenue - "
                     "profit improves because fewer, better-margin units are sold.")
        ui.callout(text, verdict, "Read-out")

    with st.expander("How this simulator works"):
        st.markdown(
            "- Every historical order is re-priced. Units change by `(new price / old price) ^ -sensitivity`.\n"
            "- **Price sensitivity** is an assumption you set in the sidebar - your data does not contain a "
            "price-vs-demand signal (see *Discount Insights*), so results are scenarios, not forecasts.\n"
            "- **Profit** = net sales - product cost x units. Taxes/shipping are treated as pass-through.\n"
            "- Best price is searched in 1% steps within your price range."
        )
