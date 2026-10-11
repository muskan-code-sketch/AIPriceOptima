"""AI Price Optima - Amazon sales analytics, discount insights and pricing scenarios.

Run locally:  streamlit run streamlit_app_FIXED.py
"""
from __future__ import annotations

import datetime as dt
import warnings

import pandas as pd
import streamlit as st

from aipo import engine, ui, view_discounts, view_model, view_overview, view_predict, view_recommend, view_simulator
from aipo.data import Settings, load_orders

warnings.filterwarnings("ignore")
st.set_page_config(page_title="AI Price Optima", page_icon="💡", layout="wide", initial_sidebar_state="expanded")
ui.inject_css()

# ------------------------------------------------------------------ data
try:
    ALL = load_orders()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load `clean_dataset_numeric.csv`: {exc}")
    st.stop()


@st.cache_data
def _measured(df: pd.DataFrame):
    return engine.measure_price_sensitivity(df)


MEASURED = _measured(ALL)
DATA_MIN, DATA_MAX = ALL["OrderDate"].min().date(), ALL["OrderDate"].max().date()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 💡 AI Price Optima")
    st.caption("Amazon sales analytics & pricing scenarios")

    st.markdown("**1 · Filter the data**")
    period = st.selectbox("Period", ["All time", "Last 12 months", "Last 6 months", "Custom range"], key="f_period")
    if period == "Last 12 months":
        start, end = DATA_MAX - dt.timedelta(days=364), DATA_MAX
    elif period == "Last 6 months":
        start, end = DATA_MAX - dt.timedelta(days=181), DATA_MAX
    elif period == "Custom range":
        picked = st.date_input("Dates", (DATA_MIN, DATA_MAX), min_value=DATA_MIN, max_value=DATA_MAX, key="f_dates")
        start, end = (picked[0], picked[1]) if len(picked) == 2 else (picked[0], DATA_MAX)
    else:
        start, end = DATA_MIN, DATA_MAX
    products = st.multiselect("Products", sorted(ALL["ProductID"].unique()), placeholder="All products", key="f_prod")
    cities = st.multiselect("Cities", sorted(ALL["City"].unique()), placeholder="All cities", key="f_city")

    st.markdown("**2 · Business assumptions**")
    with st.expander("Used by Simulator & Recommendations", expanded=True):
        elasticity = st.slider(
            "Customer price sensitivity", 0.0, 5.0, 2.5, 0.1, key="a_elast",
            help="How strongly units fall when price rises. 2.5 means +1% price -> about -2.5% units. "
                 "Online retail is typically 1.5-3.")
        if MEASURED:
            measured = 0.0 if abs(MEASURED["elasticity"]) < 0.005 else MEASURED["elasticity"]
            if MEASURED["significant"]:
                st.caption(f"Measured in your data: **{measured:.2f}** (statistically significant).")
            else:
                st.caption(f"Measured in your data: **≈ {measured:.2f}** - no detectable price-demand "
                           "link, so this slider is your call.")
        cost_pct = st.slider("Product cost (% of list price)", 20, 90, 60, 1, key="a_cost", format="%d%%",
                             help="Your data has no cost column; this sets the profit margin used for profit maths.")
        guardrail = st.slider("Max price move", 5, 40, 15, 1, key="a_guard", format="%d%%",
                              help="Price changes are searched within ± this range.")

# ------------------------------------------------------------------ filtering
mask = (ALL["OrderDate"].dt.date >= start) & (ALL["OrderDate"].dt.date <= end)
if products:
    mask &= ALL["ProductID"].isin(products)
if cities:
    mask &= ALL["City"].isin(cities)
DF = ALL[mask]

if DF.empty:
    st.warning("No orders match these filters. Widen the period or clear the product / city selection.")
    st.stop()

# previous period of equal length (for KPI arrows) - only when it exists inside the data
PREV = None
span = (end - start).days + 1
p_end, p_start = start - dt.timedelta(days=1), start - dt.timedelta(days=span)
if p_start >= DATA_MIN:
    pmask = (ALL["OrderDate"].dt.date >= p_start) & (ALL["OrderDate"].dt.date <= p_end)
    if products:
        pmask &= ALL["ProductID"].isin(products)
    if cities:
        pmask &= ALL["City"].isin(cities)
    PREV = ALL[pmask]

SET = Settings(elasticity=float(elasticity), cost_pct=cost_pct / 100, guardrail=guardrail / 100)

# ------------------------------------------------------------------ header
filt = []
if products:
    filt.append(f"{len(products)} product{'s' if len(products) > 1 else ''}")
if cities:
    filt.append(f"{len(cities)} cit{'ies' if len(cities) > 1 else 'y'}")
ui.hero(
    "AI Price Optima",
    "See where revenue comes from, what discounts really cost, and what a price change would do.",
    [f"{start:%b %Y} – {end:%b %Y}", f"{len(DF):,} orders", ui.cmoney(DF['TotalAmount'].sum()) + " revenue"]
    + (["Filtered: " + ", ".join(filt)] if filt else []),
)

# ------------------------------------------------------------------ tabs (left-to-right = the workflow)
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 Overview", "🏷️ Discount Insights", "🎛️ Price Simulator",
    "✅ Recommendations", "🔮 Revenue Predictor", "🧪 Model Health",
])
with t1:
    view_overview.render(DF, PREV)
with t2:
    view_discounts.render(DF)
with t3:
    view_simulator.render(DF, SET)
with t4:
    view_recommend.render(DF, SET)
with t5:
    view_predict.render(ALL)
with t6:
    view_model.render()

st.divider()
st.caption("AI Price Optima · XGBoost + LightGBM revenue models · Scenarios use the assumptions in the sidebar "
           "and are not forecasts.")
