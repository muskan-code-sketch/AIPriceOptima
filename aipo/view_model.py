"""Tab 6 - Model Health: how much should I trust the models?"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from . import ml, ui
from .data import MONTHS


def render() -> None:
    try:
        rep = ml.heldout_report()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    tbl, test, imp = rep["table"], rep["test"], rep["importance"]
    ens = tbl[(tbl["Split"] == "Test (unseen)") & (tbl["Model"] == "Ensemble")].iloc[0]
    train = tbl[(tbl["Split"] == "Train") & (tbl["Model"] == "Ensemble")].iloc[0]

    c = st.columns(4)
    c[0].metric("Unseen-order accuracy (R²)", f"{ens['R²'] * 100:.2f}%", border=True,
                help=f"Ensemble on {rep['n_test']:,} orders the models never saw (20% hold-out)")
    c[1].metric("Typical error (MAE)", ui.money(ens["MAE"], 2), border=True)
    c[2].metric("Error as % of order (MAPE)", f"{ens['MAPE']:.1f}%", border=True)
    c[3].metric("Train vs unseen gap", f"{(train['R²'] - ens['R²']) * 100:.2f} pts", border=True,
                help="A small gap means the models are not just memorising the training data")

    ui.callout(
        f"Scores are high because an order's revenue is almost <b>price x units x (1 - discount) + fees</b>, "
        "and those are exactly the model inputs. The models are a reliable <b>revenue estimator</b>. "
        "They do <b>not</b> learn how customers react to price - in this dataset units show no relationship "
        "with price or discount - so price-change decisions use the explicit assumptions in the "
        "<i>Price Simulator</i> and <i>Recommendations</i> tabs.",
        "warn", "How to read these numbers")

    left, right = st.columns(2, gap="large")
    with left:
        ui.section("Predicted vs actual", f"{min(len(test), 1200):,} unseen orders - closer to the line is better")
        s = test.sample(min(len(test), 1200), random_state=1)
        hi = float(max(s["TotalAmount"].max(), s["Ensemble"].max()))
        fig = go.Figure()
        fig.add_scatter(x=s["TotalAmount"], y=s["Ensemble"], mode="markers", name="Orders",
                        marker=dict(size=5, color=ui.INDIGO, opacity=0.45),
                        hovertemplate="Actual $%{x:,.0f}<br>Predicted $%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", name="Perfect", line=dict(color=ui.AMBER, dash="dash"))
        fig.update_xaxes(title="Actual revenue ($)")
        fig.update_yaxes(title="Predicted revenue ($)")
        ui.show(fig, 360, legend=True)
    with right:
        ui.section("Where are the errors?", "Average absolute error by order month")
        err = test.assign(Abs=test["Error"].abs()).groupby("Month")["Abs"].mean().reindex(range(1, 13))
        fig = go.Figure(go.Bar(x=MONTHS, y=err.values, marker_color=ui.TEAL,
                               hovertemplate="%{x}: $%{y:.2f}<extra></extra>"))
        fig.add_hline(y=float(test["Error"].abs().mean()), line_dash="dash", line_color=ui.AMBER,
                      annotation_text="overall", annotation_position="top left")
        fig.update_yaxes(title="Mean absolute error ($)", tickprefix="$")
        ui.show(fig, 360)

    ui.section("What the models pay attention to", "Share of importance per feature (each model sums to 100%)")
    fig = go.Figure()
    d = imp.sort_values("Average")
    fig.add_bar(y=d["Feature"], x=d["XGBoost"] * 100, name="XGBoost", orientation="h", marker_color=ui.INDIGO)
    fig.add_bar(y=d["Feature"], x=d["LightGBM"] * 100, name="LightGBM", orientation="h", marker_color=ui.TEAL)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Importance (%)", ticksuffix="%")
    ui.show(fig, 380, legend=True)

    with st.expander("Full metrics table"):
        st.dataframe(
            tbl, hide_index=True, width="stretch",
            column_config={"R²": st.column_config.NumberColumn(format="%.4f"),
                           "MAE": st.column_config.NumberColumn(format="$%.2f"),
                           "RMSE": st.column_config.NumberColumn(format="$%.2f"),
                           "MAPE": st.column_config.NumberColumn(format="%.2f%%")})
        st.caption("Test = the 20% of orders held out with the same random split used in the notebook "
                   "(`random_state=42`). Train = the other 80%.")
