"""Tab 5 - Revenue Predictor: single order and batch."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import ml, ui
from .data import WEEKDAYS

# TotalAmount ~= 1.081 * net sales + 7.65 (fitted on the dataset: taxes/shipping on top of net sales)
FEE_SLOPE, FEE_FIXED = 1.0805, 7.65


def _single(models, df: pd.DataFrame) -> None:
    with st.form("single_order"):
        c1, c2, c3 = st.columns(3)
        price = c1.number_input("List price ($)", 1.0, 2000.0, float(round(df["UnitPrice"].median(), 2)), 1.0)
        discount = c2.select_slider("Discount", [0, 5, 10, 15, 20, 25, 30], value=5, format_func=lambda v: f"{v}%")
        qty = c3.number_input("Units in the order", 1, 20, 3, 1)
        date = st.date_input("Order date", dt.date(2024, 6, 15), min_value=dt.date(2020, 1, 1),
                             max_value=dt.date(2026, 12, 31))
        go_btn = st.form_submit_button("Predict revenue", type="primary", width="stretch")

    if not go_btn:
        st.caption("Fill in the order and press **Predict revenue**.")
        return

    raw = pd.DataFrame({"OrderDate": [pd.Timestamp(date)], "UnitPrice": [price],
                        "Discount": [discount / 100], "Quantity": [qty]})
    p = ml.predict_raw(models, raw).iloc[0]
    net = price * qty * (1 - discount / 100)
    rule = FEE_SLOPE * net + FEE_FIXED

    c = st.columns(4)
    c[0].metric("Predicted revenue", ui.money(p["Ensemble"], 2), border=True, help="Average of XGBoost and LightGBM")
    c[1].metric("XGBoost", ui.money(p["XGBoost"], 2), border=True)
    c[2].metric("LightGBM", ui.money(p["LightGBM"], 2), border=True)
    c[3].metric("Net sales (price x units - discount)", ui.money(net, 2), border=True)

    gap = abs(p["XGBoost"] - p["LightGBM"]) / max(p["Ensemble"], 1e-9) * 100
    conf = ("good", "High agreement") if gap < 3 else (("info", "Moderate agreement") if gap < 8 else ("warn", "Models disagree"))
    ui.callout(f"The two models differ by <b>{gap:.1f}%</b>. The simple rule <i>net sales x 1.08 + $7.65</i> "
               f"gives <b>{ui.money(rule, 2)}</b>, close to the model answer - revenue per order is mostly "
               "arithmetic, which is why the model scores so high.", conf[0], conf[1])

    if date.year not in range(2020, 2025):
        st.warning("The models were trained on 2020-2024 orders; dates outside that range are extrapolations.")

    fig = go.Figure(go.Bar(x=["XGBoost", "LightGBM", "Ensemble", "Rule of thumb"],
                           y=[p["XGBoost"], p["LightGBM"], p["Ensemble"], rule],
                           marker_color=[ui.INDIGO, ui.TEAL, ui.AMBER, "#CBD5E1"],
                           text=[ui.money(v, 2) for v in (p["XGBoost"], p["LightGBM"], p["Ensemble"], rule)],
                           textposition="outside"))
    fig.update_yaxes(title="Revenue ($)", rangemode="tozero")
    ui.show(fig, 300)


def _batch(models) -> None:
    template = pd.DataFrame({"OrderDate": ["2024-03-15", "2024-07-04", "2024-11-29"],
                             "UnitPrice": [120.0, 340.5, 89.99], "Discount": [0.0, 0.1, 0.25],
                             "Quantity": [2, 4, 1]})
    st.download_button("Download CSV template", template.to_csv(index=False).encode(), "batch_template.csv",
                       "text/csv", key="tmpl")
    st.caption("Required columns: " + ", ".join(f"`{c}`" for c in ml.REQUIRED_INPUT_COLUMNS)
               + ". Discount is a fraction (0.10 = 10%). Add `TotalAmount` to also get accuracy metrics.")

    up = st.file_uploader("Upload a CSV", type=["csv"], key="batch_up")
    if up is None:
        return
    try:
        raw = pd.read_csv(up)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read that file: {exc}")
        return

    missing = [c for c in ml.REQUIRED_INPUT_COLUMNS if c not in raw.columns]
    if missing:
        st.error("Missing required column(s): " + ", ".join(missing))
        return
    raw = raw.copy()
    raw["OrderDate"] = pd.to_datetime(raw["OrderDate"], errors="coerce")
    for col in ("UnitPrice", "Discount", "Quantity"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    bad = raw[ml.REQUIRED_INPUT_COLUMNS].isna().any(axis=1)
    if bad.any():
        st.warning(f"{int(bad.sum())} row(s) have missing or invalid values and were skipped.")
    clean = raw[~bad]
    if clean.empty:
        st.error("No valid rows to score.")
        return
    if (clean["Discount"] > 1).any():
        st.warning("Some discounts are above 1 - did you enter percentages (10) instead of fractions (0.10)?")

    preds = ml.predict_raw(models, clean)
    out = clean.join(preds.round(2))
    c = st.columns(3)
    c[0].metric("Orders scored", f"{len(out):,}", border=True)
    c[1].metric("Total predicted revenue", ui.cmoney(out["Ensemble"].sum()), border=True)
    if "TotalAmount" in out.columns and out["TotalAmount"].notna().all():
        m = ml.metrics(out["TotalAmount"], out["Ensemble"])
        c[2].metric("Accuracy vs your TotalAmount", f"R² {m['R²']:.3f}", f"MAE {ui.money(m['MAE'], 2)}",
                    delta_color="off", border=True)
    st.dataframe(out, width="stretch", hide_index=True, height=320)
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv",
                       "text/csv", type="primary", key="batch_dl")


def render(df: pd.DataFrame) -> None:
    try:
        models = ml.load_models()
    except RuntimeError as exc:
        st.error(str(exc))
        st.code("xgboost_revenue_model.json (or .pkl)\nlightgbm_revenue_model.pkl\nfeature_columns.pkl")
        return
    ui.callout("Estimate what an order will bring in from its price, discount, quantity and date. "
               "This is a <b>revenue estimator</b>, not a demand forecast - to ask <i>what if I change the price?</i> "
               "use the <b>Price Simulator</b> tab.", "info", "What this tab does")
    single, batch = st.tabs(["Single order", "Batch (CSV)"])
    with single:
        _single(models, df)
    with batch:
        _batch(models)
