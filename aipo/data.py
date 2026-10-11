"""Data loading and feature preparation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "clean_dataset_numeric.csv"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass(frozen=True)
class Settings:
    """Business assumptions chosen in the sidebar."""

    elasticity: float  # price sensitivity: 2.0 means +1% price -> -2% units
    cost_pct: float  # unit cost as a share of list price (0-1)
    guardrail: float  # max price move in either direction (0-1)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + price features exactly as used when the models were trained."""
    out = df.copy()
    out["OrderDate"] = pd.to_datetime(out["OrderDate"], errors="coerce")
    out["Year"] = out["OrderDate"].dt.year
    out["Month"] = out["OrderDate"].dt.month
    out["Quarter"] = out["OrderDate"].dt.quarter
    out["DayOfWeek"] = out["OrderDate"].dt.dayofweek
    out["DayOfMonth"] = out["OrderDate"].dt.day
    out["EffectivePrice"] = out["UnitPrice"] * (1 - out["Discount"])
    return out


@st.cache_data(show_spinner="Loading sales data...")
def load_orders() -> pd.DataFrame:
    """Load the cleaned Amazon orders and add business columns."""
    df = pd.read_csv(DATA_PATH)
    df = add_time_features(df)
    df["Gross"] = df["UnitPrice"] * df["Quantity"]  # before discount
    df["DiscountAmt"] = df["Gross"] * df["Discount"]
    df["NetSales"] = df["Gross"] - df["DiscountAmt"]
    df["Fees"] = df["TotalAmount"] - df["NetSales"]  # taxes / shipping on top of net sales
    df["MonthStart"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()
    return df
