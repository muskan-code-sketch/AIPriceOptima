"""Shared look & feel: CSS, callouts, chart styling and number formatting."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

INDIGO, TEAL, AMBER, ROSE, SLATE = "#4F46E5", "#0D9488", "#F59E0B", "#E11D48", "#64748B"
COLORWAY = [INDIGO, TEAL, AMBER, ROSE, "#8B5CF6", "#0EA5E9", SLATE]

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1300px;}
h1, h2, h3 {letter-spacing: -0.01em;}
/* hero */
.hero {background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #0D9488 120%);
       border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 18px;}
.hero h1 {color:#fff; margin:0 0 4px 0; font-size: 1.9rem; padding:0;}
.hero p {color: rgba(255,255,255,.88); margin:0; font-size: 1rem;}
.chips {margin-top: 14px; display:flex; gap:8px; flex-wrap:wrap;}
.chip {background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.28);
       padding: 4px 12px; border-radius: 999px; font-size: .82rem; color:#fff;}
/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {height: 46px; padding: 0 16px; border-radius: 10px 10px 0 0; font-weight: 600;}
/* callouts */
.callout {border-left: 5px solid #4F46E5; background:#F5F7FB; padding: 14px 18px;
          border-radius: 0 12px 12px 0; margin: 10px 0 16px 0; line-height: 1.55; color:#1E293B;}
.callout.good {border-color:#0D9488; background:#ECFDF5;}
.callout.warn {border-color:#F59E0B; background:#FFFBEB;}
.callout.bad  {border-color:#E11D48; background:#FFF1F2;}
.callout b.t {display:block; margin-bottom:4px; font-size:.95rem;}
.callout ul {margin: 6px 0 0 0; padding-left: 20px;}
.small-muted {color:#64748B; font-size:.86rem;}
[data-testid="stMetric"] {border-radius: 14px;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="chips">{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def callout(body: str, kind: str = "info", title: str | None = None) -> None:
    """kind: info | good | warn | bad.  `body` may contain simple HTML (<b>, <ul><li>)."""
    t = f'<b class="t">{title}</b>' if title else ""
    st.markdown(f'<div class="callout {kind}">{t}{body}</div>', unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.markdown(f'<div class="small-muted" style="margin-top:-8px;margin-bottom:8px">{caption}</div>',
                    unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, colorway=COLORWAY,
        margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
        font=dict(family="sans-serif", size=13, color="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EEF1F6", zeroline=False)
    return fig


def show(fig: go.Figure, height: int = 360, legend: bool = False) -> None:
    st.plotly_chart(style_fig(fig, height, legend), width="stretch", theme=None,
                    config={"displayModeBar": False})


# ---- number formatting ------------------------------------------------------------------
def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.{decimals}f}"


def cmoney(x: float) -> str:
    """Compact money: $1.23M, $456.7K, $89."""
    sign, a = ("-" if x < 0 else ""), abs(x)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def pct_delta(cur: float, prev: float | None) -> str | None:
    if prev is None or prev == 0:
        return None
    return f"{(cur - prev) / abs(prev) * 100:+.1f}%"
