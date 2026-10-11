"""Tab 2 - Discount Insights: what do discounts really do?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import engine, ui


def render(df: pd.DataFrame) -> None:
    gross = df["Gross"].sum()
    cost = df["DiscountAmt"].sum()
    lift = engine.discount_lift(df)

    c = st.columns(4)
    c[0].metric("Discount given away", ui.cmoney(cost), border=True,
                help="Sum of unit price x quantity x discount rate")
    c[1].metric("Share of gross sales", f"{cost / gross * 100:.1f}%" if gross else "-", border=True)
    c[2].metric("Orders with a discount", f"{(df['Discount'] > 0).mean() * 100:.0f}%", border=True)
    if lift:
        c[3].metric("Extra units from discounting", f"{lift['lift_pct']:+.1f}%", border=True,
                    help="Average units per order, discounted vs full-price orders")
    else:
        c[3].metric("Extra units from discounting", "n/a", border=True)

    # ---- verdict ----------------------------------------------------------------------
    if lift is None:
        ui.callout("Not enough discounted and full-price orders in the current selection to test the effect.",
                   "warn", "Not enough data")
    elif lift["significant"] and lift["lift_pct"] > 0:
        ui.callout(
            f"Discounted orders contain <b>{lift['lift_pct']:+.1f}%</b> more units "
            f"(95% range {lift['ci_low_pct']:+.1f}% to {lift['ci_high_pct']:+.1f}%). "
            "That is a real volume effect, so compare it with the discount cost before cutting.",
            "good", "Discounts do lift volume")
    else:
        ui.callout(
            f"Discounted orders hold <b>{lift['units_discounted']:.2f}</b> units on average vs "
            f"<b>{lift['units_full_price']:.2f}</b> for full-price orders - a difference of "
            f"<b>{lift['lift_pct']:+.1f}%</b> (95% range {lift['ci_low_pct']:+.1f}% to {lift['ci_high_pct']:+.1f}%, "
            f"p = {lift['p_value']:.2f}). That is statistically indistinguishable from zero, so the "
            f"<b>{ui.cmoney(cost)}</b> of discounts shows no measurable return in extra volume.",
            "warn", "Discounts are not buying extra volume in this data")

    # ---- evidence + breakdown ---------------------------------------------------------
    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Do bigger discounts sell more units?", "Average units per order at each discount level (bars = 95% range)")
        bands = engine.units_by_discount_band(df)
        fig = go.Figure(go.Bar(
            x=bands["Label"], y=bands["mean"], error_y=dict(type="data", array=bands["ci"], color="#94A3B8"),
            marker_color=ui.INDIGO, customdata=bands["count"],
            hovertemplate="Discount %{x}<br>%{y:.2f} units/order<br>%{customdata:,} orders<extra></extra>"))
        fig.add_hline(y=df["Quantity"].mean(), line_dash="dash", line_color=ui.AMBER,
                      annotation_text="overall average", annotation_position="top left")
        fig.update_yaxes(title="Units per order", rangemode="tozero")
        fig.update_xaxes(title="Discount rate")
        ui.show(fig, 350)
    with right:
        ui.section("Where is the discount money going?", "Total discount given - pick a breakdown")
        by = st.radio("Break down by", ["Product", "City", "Month"], horizontal=True, key="disc_by",
                      label_visibility="collapsed")
        key = {"Product": "ProductID", "City": "City", "Month": "MonthStart"}[by]
        t = df.groupby(key)["DiscountAmt"].sum().reset_index()
        if by == "Month":
            fig = px.area(t.sort_values(key), x=key, y="DiscountAmt", color_discrete_sequence=[ui.ROSE])
            fig.update_xaxes(title=None)
        else:
            t = t.nlargest(12, "DiscountAmt").sort_values("DiscountAmt")
            fig = px.bar(t, x="DiscountAmt", y=key, orientation="h", color_discrete_sequence=[ui.ROSE])
            fig.update_yaxes(title=None)
        if by == "Month":
            fig.update_traces(hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>")
            fig.update_yaxes(title=None, tickprefix="$", tickformat="~s")
        else:
            fig.update_traces(hovertemplate="%{y}: $%{x:,.0f}<extra></extra>")
            fig.update_xaxes(title=None, tickprefix="$", tickformat="~s")
        ui.show(fig, 350)

    # ---- calculator -------------------------------------------------------------------
    st.divider()
    ui.section("Discount cap calculator", "What if no order got more than X% off?")
    a, b = st.columns(2, gap="large")
    cap = a.slider("Maximum discount allowed", 0, 30, 5, step=5, format="%d%%", key="disc_cap") / 100
    loss = b.slider("Units you expect to lose on affected orders", 0, 30, 5, format="%d%%", key="disc_loss",
                    help="Safety margin. The data shows ~0%, but be conservative.") / 100
    res = engine.discount_cap_analysis(df, cap, loss)

    m = st.columns(4)
    m[0].metric("Orders affected", f"{res['orders_affected']:,}", f"{res['share_affected'] * 100:.0f}% of orders",
                delta_color="off", border=True)
    m[1].metric("Revenue recovered if no units lost", ui.cmoney(res["gain_if_no_loss"]), border=True)
    m[2].metric(f"Revenue change with {loss * 100:.0f}% unit loss", ui.cmoney(res["revenue_delta"]),
                f"{res['revenue_delta'] / res['revenue_now'] * 100:+.1f}%", border=True)
    m[3].metric("Break-even unit loss", f"{res['break_even_loss'] * 100:.1f}%", border=True,
                help="If you lose more than this share of units on affected orders, the cap loses money")

    if res["orders_affected"] == 0:
        ui.callout("No order in the current selection has a discount above that cap.", "info")
    else:
        good = res["revenue_delta"] > 0
        be = res["break_even_loss"] * 100
        text = (f"Capping discounts at <b>{cap * 100:.0f}%</b> touches {res['orders_affected']:,} orders. "
                f"It pays off as long as you lose fewer than <b>{be:.1f}%</b> of the units on those orders.")
        if lift:
            if lift["ci_high_pct"] < be:
                text += (f" Even the upper end of the volume effect seen from discounts today "
                         f"(+{max(lift['ci_high_pct'], 0):.1f}%) sits below that break-even point.")
            else:
                text += (f" Caution: the volume effect seen from discounts today could be as high as "
                         f"+{lift['ci_high_pct']:.1f}%, which is above that break-even point.")
        ui.callout(text, "good" if good else "bad",
                   "Worth testing" if good else "Not worth it at this loss level")
        st.caption("Tip: roll this out as an A/B test on a few products first - the data can show correlation, "
                   "not what customers do after a policy change.")

    # ---- ranked table -----------------------------------------------------------------
    with st.expander("Products ranked by discount cost"):
        tbl = df.groupby("ProductID").agg(
            Orders=("OrderID", "count"), Gross=("Gross", "sum"), DiscountGiven=("DiscountAmt", "sum"),
            AvgDiscount=("Discount", "mean")).reset_index()
        tbl["AvgDiscount"] *= 100
        tbl["PctOfGross"] = tbl["DiscountGiven"] / tbl["Gross"] * 100
        st.dataframe(
            tbl.sort_values("DiscountGiven", ascending=False), hide_index=True, width="stretch",
            column_config={
                "ProductID": "Product", "Orders": st.column_config.NumberColumn(format="%,d"),
                "Gross": st.column_config.NumberColumn("Gross sales", format="$%,.0f"),
                "DiscountGiven": st.column_config.NumberColumn("Discount given", format="$%,.0f"),
                "AvgDiscount": st.column_config.NumberColumn("Avg discount", format="%.1f%%"),
                "PctOfGross": st.column_config.ProgressColumn("% of gross", format="%.1f%%", min_value=0,
                                                              max_value=float(max(tbl["PctOfGross"].max(), 1))),
            })
