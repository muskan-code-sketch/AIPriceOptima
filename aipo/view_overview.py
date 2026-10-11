"""Tab 1 - Overview: how is the business doing?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import ui
from .data import MONTHS


def _kpis(d: pd.DataFrame) -> dict:
    n = len(d)
    return {
        "revenue": d["TotalAmount"].sum(),
        "orders": n,
        "units": d["Quantity"].sum(),
        "aov": d["TotalAmount"].mean() if n else 0.0,
        "discount": d["Discount"].mean() if n else 0.0,
        "price": d["UnitPrice"].mean() if n else 0.0,
    }


def render(df: pd.DataFrame, prev: pd.DataFrame | None) -> None:
    cur = _kpis(df)
    old = _kpis(prev) if prev is not None and len(prev) else None

    def delta(key):
        return ui.pct_delta(cur[key], old[key]) if old else None

    cols = st.columns(6)
    cols[0].metric("Revenue", ui.cmoney(cur["revenue"]), delta("revenue"), border=True)
    cols[1].metric("Orders", f"{cur['orders']:,}", delta("orders"), border=True)
    cols[2].metric("Units sold", f"{cur['units']:,.0f}", delta("units"), border=True)
    cols[3].metric("Avg order value", ui.money(cur["aov"]), delta("aov"), border=True)
    cols[4].metric("Avg discount", f"{cur['discount'] * 100:.1f}%",
                   f"{(cur['discount'] - old['discount']) * 100:+.1f} pp" if old else None,
                   delta_color="inverse", border=True)
    cols[5].metric("Avg unit price", ui.money(cur["price"]), delta("price"), border=True)
    if old:
        st.caption("Arrows compare with the previous period of the same length.")

    # ---- auto-generated takeaways -----------------------------------------------------
    month_avg = df.groupby(["Year", "Month"])["TotalAmount"].sum().groupby("Month").mean()
    best_m, worst_m = month_avg.idxmax(), month_avg.idxmin()
    top5_share = df.groupby("ProductID")["TotalAmount"].sum().nlargest(5).sum() / cur["revenue"] * 100
    top_city = df.groupby("City")["TotalAmount"].sum().idxmax()
    disc_cost = df["DiscountAmt"].sum()
    ui.callout(
        "<ul>"
        f"<li><b>{MONTHS[best_m - 1]}</b> is the strongest month (about {ui.cmoney(month_avg.max())} on average, "
        f"{(month_avg.max() / month_avg.mean() - 1) * 100:+.0f}% vs a typical month); "
        f"<b>{MONTHS[worst_m - 1]}</b> is the weakest ({(month_avg.min() / month_avg.mean() - 1) * 100:+.0f}%).</li>"
        f"<li>The top 5 products bring in <b>{top5_share:.0f}%</b> of revenue and <b>{top_city}</b> is the largest city.</li>"
        f"<li>Discounts gave away <b>{ui.cmoney(disc_cost)}</b> "
        f"({disc_cost / df['Gross'].sum() * 100:.1f}% of gross sales) - see the <i>Discount Insights</i> tab.</li>"
        "</ul>",
        title="What stands out in the selected data",
    )

    # ---- trend + seasonality ----------------------------------------------------------
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("Revenue trend", "Monthly revenue with a 3-month moving average")
        m = df.groupby("MonthStart", as_index=False)["TotalAmount"].sum().sort_values("MonthStart")
        m["MA3"] = m["TotalAmount"].rolling(3, min_periods=1).mean()
        fig = go.Figure()
        fig.add_bar(x=m["MonthStart"], y=m["TotalAmount"], name="Monthly revenue",
                    marker_color="#C7D2FE", hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=m["MonthStart"], y=m["MA3"], name="3-month average", mode="lines",
                        line=dict(color=ui.INDIGO, width=3), hovertemplate="$%{y:,.0f}<extra></extra>")
        fig.update_yaxes(tickprefix="$", tickformat="~s")
        ui.show(fig, 340, legend=True)
    with right:
        ui.section("Seasonality", "Revenue by month and year - darker is higher")
        pv = df.pivot_table(index="Year", columns="Month", values="TotalAmount", aggfunc="sum")
        pv = pv.reindex(columns=range(1, 13))
        fig = px.imshow(pv.values, x=MONTHS, y=[str(y) for y in pv.index], aspect="auto",
                        color_continuous_scale=["#EEF2FF", "#818CF8", "#4338CA"])
        fig.update_traces(hovertemplate="%{y} %{x}<br>$%{z:,.0f}<extra></extra>")
        fig.update_coloraxes(showscale=False)
        ui.show(fig, 340)

    # ---- leaders ----------------------------------------------------------------------
    st.divider()
    metric = st.radio("Rank by", ["Revenue", "Orders", "Units"], horizontal=True, key="ov_rank")
    col = {"Revenue": "TotalAmount", "Orders": "OrderID", "Units": "Quantity"}[metric]
    agg = "count" if metric == "Orders" else "sum"
    c1, c2 = st.columns(2, gap="large")
    for box, dim, label in ((c1, "ProductID", "Top products"), (c2, "City", "Top cities")):
        with box:
            ui.section(label)
            t = df.groupby(dim)[col].agg(agg).nlargest(10).sort_values().reset_index()
            fig = px.bar(t, x=col, y=dim, orientation="h", color_discrete_sequence=[ui.INDIGO])
            fig.update_traces(hovertemplate="%{y}: %{x:,.0f}<extra></extra>")
            fig.update_xaxes(title=None, tickprefix="$" if metric == "Revenue" else "", tickformat="~s")
            fig.update_yaxes(title=None)
            ui.show(fig, 360)

    # ---- year over year ---------------------------------------------------------------
    yoy = df.groupby("Year").agg(Revenue=("TotalAmount", "sum"), Orders=("OrderID", "count"),
                                 Units=("Quantity", "sum"), AvgDiscount=("Discount", "mean")).reset_index()
    if len(yoy) > 1:
        yoy["AvgDiscount"] = yoy["AvgDiscount"] * 100
        yoy["Revenue growth"] = yoy["Revenue"].pct_change() * 100
        with st.expander("Year-over-year table"):
            st.dataframe(
                yoy, hide_index=True, width="stretch",
                column_config={
                    "Year": st.column_config.NumberColumn(format="%d"),
                    "Revenue": st.column_config.NumberColumn(format="$%,.0f"),
                    "Orders": st.column_config.NumberColumn(format="%,d"),
                    "Units": st.column_config.NumberColumn(format="%,d"),
                    "AvgDiscount": st.column_config.NumberColumn("Avg discount", format="%.1f%%",
                                                                 help="Average discount rate"),
                    "Revenue growth": st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )
