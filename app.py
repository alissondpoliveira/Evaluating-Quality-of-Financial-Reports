"""
Advisor-Brain-FSA — Streamlit App
Bloomberg Terminal theme · Deploy no Streamlit Community Cloud
Run: streamlit run app.py

Performance design:
  - Only stdlib + streamlit imported before set_page_config (required by Streamlit)
  - ticker_map imported after set_page_config; all other advisor modules lazy
  - All heavy advisor_brain_fsa modules loaded lazily via @st.cache_resource
  - plotly.graph_objects loaded lazily on first chart render
  - market cache JSON read once per 5 min via @st.cache_data(ttl=300)
  - CSS pre-generated as a module-level constant (no f-string on every rerun)
  - Dropdown options split: static (instant) + full CVM (lazy, on demand)
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ── Only streamlit at module level (before set_page_config) ──────────────────
import streamlit as st

# ── Page config — must be the FIRST st call ───────────────────────────────────
st.set_page_config(
    page_title="Advisor-Brain-FSA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── No advisor imports at module level — all lazy via @st.cache_resource ──────

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_B = {
    "bg":     "#000000", "card":   "#1C1C1C", "border": "#30363D",
    "text":   "#D1D1D1", "green":  "#00FF00", "red":    "#FF3E3E",
    "orange": "#FFA500", "yellow": "#FFD700", "muted":  "#888888",
    "mono":   "'JetBrains Mono','Roboto Mono','Fira Code','Consolas',monospace",
}
_CY                = date.today().year
_YEAR_OPTS         = list(range(_CY - 1, _CY - 12, -1))
_FINANCIAL_SECTORS = {"Bancos", "Seguros", "Financeiro", "BDR"}
_FIN_ALL           = _FINANCIAL_SECTORS | {"Financeiro", "BDR", "Bancos", "Seguros"}


def _read_version() -> str:
    try:
        return "v" + (Path(__file__).parent / "VERSION").read_text().strip()
    except Exception:
        return "v?"


_APP_VERSION = _read_version()

# ─────────────────────────────────────────────────────────────────────────────
# CSS — pre-generated once at import time (no f-string rebuild on every rerun)
# ─────────────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="block-container"] {{
    background-color: {_B['bg']} !important;
    color: {_B['text']} !important;
    font-family: {_B['mono']} !important;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stDecoration"], [data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: {_B['bg']} !important;
    border-bottom: 1px solid {_B['border']} !important;
    gap: 0 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: {_B['bg']} !important;
    color: {_B['muted']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.78rem !important;
    border: 1px solid {_B['border']} !important;
    border-bottom: none !important;
    padding: 6px 18px !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {_B['orange']} !important;
    border-bottom: 2px solid {_B['orange']} !important;
}}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
    background-color: {_B['card']} !important;
    color: {_B['text']} !important;
    border: 1px solid {_B['border']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
}}
div[data-baseweb="select"] > div {{
    background-color: {_B['card']} !important;
    border: 1px solid {_B['border']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
    color: {_B['text']} !important;
}}
div[data-baseweb="popover"] * {{
    background-color: {_B['card']} !important;
    color: {_B['text']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
}}
[data-baseweb="tag"] {{
    background-color: #2a1a00 !important;
    color: {_B['orange']} !important;
}}
[data-testid="stButton"] > button {{
    background-color: {_B['card']} !important;
    color: {_B['orange']} !important;
    border: 1px solid {_B['orange']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    border-radius: 3px !important;
    padding: 6px 14px !important;
}}
[data-testid="stButton"] > button:hover {{
    background-color: #2a1a00 !important;
}}
[data-testid="stDownloadButton"] > button {{
    background-color: {_B['card']} !important;
    color: {_B['orange']} !important;
    border: 1px solid {_B['border']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.78rem !important;
}}
[data-testid="stExpander"] {{
    background-color: {_B['card']} !important;
    border: 1px solid {_B['border']} !important;
    border-radius: 4px !important;
}}
[data-testid="stExpander"] summary {{
    color: {_B['orange']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}}
[data-testid="stMetric"] label {{
    font-family: {_B['mono']} !important;
    font-size: 0.65rem !important;
    color: {_B['muted']} !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}
[data-testid="stMetricValue"] {{ font-family: {_B['mono']} !important; font-size: 1.4rem !important; }}
[data-testid="stCaptionContainer"] p {{
    color: {_B['muted']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.72rem !important;
}}
[data-testid="stAlert"] {{
    background-color: {_B['card']} !important;
    border: 1px solid {_B['border']} !important;
    font-family: {_B['mono']} !important;
    font-size: 0.82rem !important;
}}
[data-testid="stProgressBar"] > div > div {{ background-color: {_B['orange']} !important; }}
[data-testid="stSpinner"] > div {{ color: {_B['orange']} !important; }}
hr {{ border-color: {_B['border']} !important; margin: 10px 0 !important; }}
.bbg-card {{
    background: {_B['card']};
    border: 1px solid {_B['border']};
    border-radius: 6px;
    padding: 10px 16px;
    margin-bottom: 10px;
    font-family: {_B['mono']};
}}
.flag-item {{
    background: #1a0000;
    border: 1px solid #3d0000;
    border-left: 3px solid {_B['red']};
    padding: 6px 12px;
    margin-bottom: 6px;
    font-family: {_B['mono']};
    font-size: 0.80rem;
    color: {_B['text']};
    border-radius: 2px;
}}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Lazy module loaders — imported ONCE on first use, then cached in memory
# Keeps cold-start fast: advisor_brain_fsa modules only load when needed
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _go():
    """plotly.graph_objects — lazy so cold start doesn't pay the ~400ms import."""
    import plotly.graph_objects as go
    return go


@st.cache_resource(show_spinner=False)
def _rank_market():
    """rank_market module objects — loads numpy/pandas/data_fetcher stack once."""
    from advisor_brain_fsa.rank_market import (
        CompanyResult, _apply_sector_stats, _to_dataframe,
        get_home_dashboard_data, read_market_cache,
    )
    return CompanyResult, _apply_sector_stats, _to_dataframe, \
           get_home_dashboard_data, read_market_cache


@st.cache_resource(show_spinner=False)
def _scorer_mod():
    from advisor_brain_fsa.sector_scorer import get_scorer
    return get_scorer


@st.cache_resource(show_spinner=False)
def _mda_mod():
    from advisor_brain_fsa.mda_analyst import GeminiAnalyst, compute_grade
    return GeminiAnalyst, compute_grade


@st.cache_resource(show_spinner=False)
def _ticker_map():
    from advisor_brain_fsa.ticker_map import TICKER_TO_KEYWORD, TICKER_SECTOR, get_sector
    return TICKER_TO_KEYWORD, TICKER_SECTOR, get_sector


# ─────────────────────────────────────────────────────────────────────────────
# Market cache — read JSON at most once per 5 minutes
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _market_df():
    _, _, _, _, read_mc = _rank_market()
    df = read_mc()
    return df if (df is not None and not df.empty) else None


# ─────────────────────────────────────────────────────────────────────────────
# API key
# ─────────────────────────────────────────────────────────────────────────────
def _api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GOOGLE_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart builders — use lazy _go() so first chart doesn't block startup
# ─────────────────────────────────────────────────────────────────────────────
_BBG_LAYOUT = dict(
    paper_bgcolor=_B["bg"], plot_bgcolor=_B["bg"],
    font=dict(family=_B["mono"], color=_B["text"]),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _gauge(m_score: float):
    go   = _go()
    clr  = _B["red"] if m_score > -1.78 else _B["green"]
    fig  = go.Figure(go.Bar(
        x=[m_score], y=["M-Score"], orientation="h", marker_color=clr,
        text=[f"{m_score:+.4f}"], textposition="outside",
        textfont=dict(family=_B["mono"], color=clr, size=13),
    ))
    fig.add_vline(x=-1.78, line_dash="dash", line_color=_B["orange"], line_width=2)
    fig.update_layout(**_BBG_LAYOUT,
        title=dict(text="M-Score vs Limiar (−1.78)", font=dict(size=10, color=_B["muted"])),
        xaxis=dict(range=[-4, 1], gridcolor=_B["border"], zeroline=False),
        yaxis=dict(visible=False), height=170,
    )
    return fig


def _radar(ms):
    go   = _go()
    cats = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
    vals = [ms.dsri, ms.gmi, ms.aqi, ms.sgi, ms.depi, ms.sgai, ms.lvgi,
            max(0, ms.tata * 10 + 1)]
    thr  = [1.031, 1.014, 1.039, 1.134, 1.017, 1.054, 1.000, 1.000]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=thr + [thr[0]], theta=cats + [cats[0]], name="Limiar",
        line_color=_B["orange"], line_dash="dot", fill="none"))
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]], name="Empresa",
        line_color=_B["red"], fill="toself", fillcolor="rgba(255,62,62,0.13)"))
    fig.update_layout(**_BBG_LAYOUT,
        title=dict(text="Índices Beneish", font=dict(size=10, color=_B["muted"])),
        polar=dict(bgcolor=_B["card"],
                   radialaxis=dict(visible=True, range=[0, 2], gridcolor=_B["border"]),
                   angularaxis=dict(gridcolor=_B["border"])),
        legend=dict(font=dict(size=8, color=_B["muted"]), bgcolor=_B["bg"]),
        height=320,
    )
    return fig


def _sector_bar(df):
    import pandas as pd
    go   = _go()
    g    = df.groupby("Setor")["M-Score"].mean().dropna().sort_values()
    clrs = [_B["red"] if v > -1.78 else _B["green"] for v in g.values]
    fig  = go.Figure(go.Bar(
        x=g.values, y=g.index, orientation="h", marker_color=clrs,
        text=[f"{v:+.3f}" for v in g.values], textposition="outside",
        textfont=dict(family=_B["mono"], color=_B["text"], size=9),
    ))
    fig.add_vline(x=-1.78, line_dash="dash", line_color=_B["orange"], line_width=1.5)
    fig.update_layout(**_BBG_LAYOUT,
        title=dict(text="M-Score Médio por Setor", font=dict(size=10, color=_B["muted"])),
        xaxis=dict(range=[-4, 0.5], gridcolor=_B["border"]),
        yaxis=dict(gridcolor=_B["border"]),
        height=max(180, len(g) * 32 + 60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# BRL / delta helpers
# ─────────────────────────────────────────────────────────────────────────────
def _brl(v: float) -> str:
    a = abs(v)
    if a >= 1e12: return f"R$ {v/1e12:.2f} T"
    if a >= 1e9:  return f"R$ {v/1e9:.2f} Bn"
    if a >= 1e6:  return f"R$ {v/1e6:.2f} M"
    if a >= 1e3:  return f"R$ {v/1e3:.1f} K"
    return f"R$ {v:.0f}"


def _delta_pct(t1: float, t: float) -> tuple:
    if abs(t1) < 1e-6:
        return "—", _B["muted"]
    pct = (t - t1) / abs(t1) * 100
    return f"{pct:+.1f}%", (_B["green"] if pct >= 0 else _B["red"])


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────
def _pill_html(alert: str) -> str:
    m = {"Crítico": (_B["red"], "#2a0000"), "Alto Risco": (_B["orange"], "#2a1a00"),
         "Atenção": (_B["yellow"], "#2a2000"), "Normal": (_B["green"], "#002200")}
    c, bg = m.get(alert, (_B["muted"], "#111111"))
    return (f'<span style="display:inline-block;padding:2px 10px;border-radius:3px;'
            f'border:1px solid {c};color:{c};background:{bg};font-weight:700;'
            f'font-size:0.76rem;letter-spacing:0.04em;font-family:{_B["mono"]}">{alert}</span>')


def _grade_badge_html(grade: str) -> str:
    colors = {"A": "#00FF00", "B": "#7FFF00", "C": "#FFD700", "D": "#FFA500", "F": "#FF3E3E"}
    bg = colors.get(grade, "#888888")
    return (f'<div style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:64px;height:64px;border-radius:4px;font-size:2.2rem;font-weight:900;'
            f'color:#000000;background:{bg};font-family:{_B["mono"]}">{grade}</div>')


def _profile_card_html(profile: dict) -> str:
    denom_s   = profile.get("denom_social", "")
    denom_c   = profile.get("denom_comerc", "")
    cnpj_fmt  = profile.get("cnpj_fmt", "")
    setor     = profile.get("setor_ativ", "")
    sector_l  = profile.get("sector_label", "")
    sit_reg   = profile.get("sit_reg", "")
    tp_merc   = profile.get("tp_merc", "")
    name_main = denom_c or denom_s or "—"
    name_sub  = denom_s if denom_c else ""
    sit_color = _B["green"] if sit_reg.upper() == "ATIVO" else _B["red"]

    def kv(k: str, v: str, vc: str = _B["text"]) -> str:
        if not v: return ""
        return (f'<span style="display:inline-block;margin-right:20px;margin-bottom:3px">'
                f'<span style="color:{_B["muted"]};font-size:0.72rem;margin-right:4px">{k}:</span>'
                f'<span style="color:{vc};font-size:0.78rem;font-weight:600">{v}</span></span>')

    sub = (f'<span style="color:{_B["muted"]};font-size:0.78rem;margin-left:8px">{name_sub}</span>'
           if name_sub else "")
    return (f'<div class="bbg-card">'
            f'<div style="margin-bottom:6px">'
            f'<span style="color:{_B["text"]};font-size:1.05rem;font-weight:700">{name_main}</span>'
            f'{sub}</div>'
            f'<div>'
            f'{kv("CNPJ", cnpj_fmt)}'
            f'{kv("Setor CVM", setor)}'
            f'{kv("Classificação", sector_l, _B["orange"])}'
            f'{kv("Mercado", tp_merc)}'
            f'{kv("Status", sit_reg, sit_color)}'
            f'</div></div>')


def _audit_table_html(fd_t, fd_t1, year_t: int) -> str:
    yr1   = year_t - 1
    yr0   = year_t
    gp_t  = fd_t.revenues  - fd_t.cost_of_goods_sold
    gp_t1 = fd_t1.revenues - fd_t1.cost_of_goods_sold

    rows_spec = [
        ("DRE",                     None, None, False),
        ("Receita Líquida",         fd_t1.revenues,                    fd_t.revenues,                    False),
        ("CPV",                     fd_t1.cost_of_goods_sold,           fd_t.cost_of_goods_sold,           True),
        ("Lucro Bruto",             gp_t1,                              gp_t,                             False),
        ("SG&A",                    fd_t1.sales_general_admin_expenses, fd_t.sales_general_admin_expenses, True),
        ("Lucro Líquido",           fd_t1.net_income,                   fd_t.net_income,                  False),
        ("FCO",                     fd_t1.cash_from_operations,         fd_t.cash_from_operations,        False),
        ("D&A (add-back)",          fd_t1.depreciation,                 fd_t.depreciation,                False),
        ("BALANÇO — ATIVO",         None, None, False),
        ("Ativo Total",             fd_t1.total_assets,                 fd_t.total_assets,                False),
        ("Ativo Circulante",        fd_t1.current_assets,               fd_t.current_assets,              False),
        ("Contas a Receber (líq.)", fd_t1.receivables,                  fd_t.receivables,                 False),
        ("Imobilizado (PP&E)",      fd_t1.pp_and_e,                     fd_t.pp_and_e,                    False),
        ("Aplicações Financeiras",  fd_t1.securities,                   fd_t.securities,                  False),
        ("BALANÇO — PASSIVO",       None, None, False),
        ("Passivo Circulante",      fd_t1.current_liabilities,          fd_t.current_liabilities,         False),
        ("Dívida LP",               fd_t1.total_long_term_debt,         fd_t.total_long_term_debt,        False),
    ]

    TH  = (f'style="color:{_B["muted"]};font-family:{_B["mono"]};font-size:0.72rem;font-weight:700;'
           f'padding:5px 12px;border-bottom:1px solid {_B["border"]};letter-spacing:0.08em;'
           f'text-align:right;white-space:nowrap;background:{_B["card"]}"')
    THL = TH.replace("text-align:right", "text-align:left")
    TD  = (f'style="color:{_B["text"]};font-family:{_B["mono"]};font-size:0.80rem;'
           f'padding:4px 12px;text-align:right;white-space:nowrap"')
    TDL = TD.replace("text-align:right", "text-align:left")
    TDG = (f'style="color:{_B["orange"]};font-family:{_B["mono"]};font-size:0.69rem;'
           f'font-weight:700;padding:8px 12px 2px;text-align:left;letter-spacing:0.1em;'
           f'border-top:1px solid {_B["border"]}"')

    head = (f'<tr><th {THL}>Métrica Contábil</th>'
            f'<th {TH}>Ano {yr1}</th><th {TH}>Ano {yr0}</th>'
            f'<th {TH} style="color:{_B["orange"]}">Δ%</th></tr>')
    body, i = "", 0
    for label, v1, v0, is_cost in rows_spec:
        if v1 is None:
            body += f'<tr><td {TDG} colspan="4">{label}</td></tr>'
            continue
        dv1 = _brl(-v1 if is_cost else v1)
        dv0 = _brl(-v0 if is_cost else v0)
        pct, pc = _delta_pct(v1, v0)
        bg = _B["card"] if i % 2 == 0 else _B["bg"]
        body += (f'<tr style="border-bottom:1px solid {_B["border"]}22;background:{bg}">'
                 f'<td {TDL}>{label}</td>'
                 f'<td {TD} style="color:{_B["muted"]}">{dv1}</td>'
                 f'<td {TD}>{dv0}</td>'
                 f'<td {TD} style="color:{pc};font-weight:700">{pct}</td></tr>')
        i += 1

    return (f'<div style="overflow-x:auto;margin-top:8px">'
            f'<table style="width:100%;border-collapse:collapse;'
            f'border:1px solid {_B["border"]};border-radius:4px">'
            f'<thead>{head}</thead><tbody>{body}</tbody></table></div>')


# ─────────────────────────────────────────────────────────────────────────────
# Ticker option lists
# Static list: instant (ticker_map already imported at module level)
# Full list:   lazy, loaded only when user clicks "Carregar todas as empresas"
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _static_opts() -> tuple:
    """Fast path — builds option list from ticker_map only."""
    TICKER_TO_KEYWORD, TICKER_SECTOR, _ = _ticker_map()
    labels, values = [], []
    for t in sorted(TICKER_TO_KEYWORD.keys()):
        if TICKER_SECTOR.get(t) not in _FINANCIAL_SECTORS:
            labels.append(f"{t} — {TICKER_TO_KEYWORD[t]}")
            values.append(t)
    return labels, values, dict(zip(labels, values))


@st.cache_resource(show_spinner=False)
def _full_opts() -> tuple:
    """Slow path — merges static list with full CVM non-financial universe."""
    TICKER_TO_KEYWORD, TICKER_SECTOR, _ = _ticker_map()
    labels, values, lbl2val = _static_opts()
    labels, values = list(labels), list(values)
    try:
        from advisor_brain_fsa.cvm_registry import CVMRegistry
        non_fin_kws = {
            TICKER_TO_KEYWORD[t].upper()
            for t in TICKER_TO_KEYWORD
            if TICKER_SECTOR.get(t) not in _FINANCIAL_SECTORS
        }
        nf_df = CVMRegistry.get_instance().get_non_financial_df()
        extras: list[tuple] = []
        for _, row in nf_df.iterrows():
            denom = str(row.get("DENOM_COMERC") or row.get("DENOM_SOCIAL") or "").strip()
            cnpj  = str(row.get("CNPJ_CIA") or row.get("CNPJ") or "").strip()
            if not denom or not cnpj:
                continue
            norm = denom.upper()
            if any(kw in norm or norm in kw for kw in non_fin_kws):
                continue
            setor = str(row.get("_SECTOR_LABEL") or "Outros")
            extras.append((f"{denom} — {setor}", cnpj))
        for lbl, val in sorted(extras, key=lambda x: x[0]):
            labels.append(lbl)
            values.append(val)
    except Exception:
        pass
    return labels, values, dict(zip(labels, values))


@st.cache_resource(show_spinner=False)
def _ranking_opts() -> tuple:
    TICKER_TO_KEYWORD, _, _ = _ticker_map()
    tickers = sorted(TICKER_TO_KEYWORD.keys())
    labels  = [f"{t} — {TICKER_TO_KEYWORD.get(t, t)[:14]}" for t in tickers]
    return labels, dict(zip(labels, tickers))


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
def _init_state() -> None:
    for k, v in {
        "analyze_result": None,
        "last_result":    None,
        "selected_ticker": "",
        "gemini_report":  None,
        "ranking_df":     None,
        "use_full_opts":  False,   # whether Análise tab loaded full CVM list
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Result renderer
# ─────────────────────────────────────────────────────────────────────────────
def _render_result(ticker: str, sector: str, sr, year_t: int,
                   fd_t=None, fd_t1=None) -> None:
    _, compute_grade = _mda_mod()
    ms    = sr.mscore_result
    cfq   = sr.cfq_result
    alert = sr.alert_level.value
    flags = sr.red_flags or []

    grade, g_label = (compute_grade(ms.m_score, cfq.accrual_ratio)
                      if ms and cfq else ("N/D", "Dados insuficientes para análise"))

    m_color = _B["red"] if (ms and ms.m_score > -1.78) else _B["green"]
    ms_val  = f"{ms.m_score:+.4f}" if ms else "N/D"
    cfq_val = f"{cfq.accrual_ratio:+.4f}" if cfq else "N/D"
    classif = ms.classification if ms else "—"

    st.markdown(
        f'<div class="bbg-card">'
        f'<span style="color:{_B["orange"]};font-weight:700;font-size:1rem">{ticker}</span>'
        f'<span style="color:{_B["muted"]};font-size:0.78rem"> · {sector} · {year_t}</span>'
        f'<div style="font-size:0.72rem;color:{_B["muted"]};margin-top:2px">{g_label}</div>'
        f'</div>', unsafe_allow_html=True,
    )

    col_g, col_m, col_pill = st.columns([1, 4, 1])
    with col_g:
        st.markdown(_grade_badge_html(grade), unsafe_allow_html=True)
    with col_m:
        st.markdown(
            f'<div style="font-family:{_B["mono"]};line-height:1.7">'
            f'<span style="color:{_B["muted"]};font-size:0.8rem">M-Score: </span>'
            f'<span style="color:{m_color};font-size:1.1rem;font-weight:700">{ms_val}</span><br>'
            f'<span style="color:{_B["muted"]};font-size:0.75rem">Limiar: −1.78 → </span>'
            f'<span style="color:{_B["orange"]};font-size:0.75rem;font-weight:700">{classif}</span><br>'
            f'<span style="color:{_B["muted"]};font-size:0.75rem">Accrual Ratio: </span>'
            f'<span style="color:{_B["text"]};font-size:0.75rem">{cfq_val}</span>'
            f'</div>', unsafe_allow_html=True,
        )
    with col_pill:
        st.markdown(_pill_html(alert), unsafe_allow_html=True)

    st.divider()

    if ms:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(_gauge(ms.m_score), use_container_width=True,
                            config={"displayModeBar": False})
        with c2:
            st.plotly_chart(_radar(ms), use_container_width=True,
                            config={"displayModeBar": False})

    st.divider()

    st.markdown(
        f'<div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.12em;color:{_B["muted"]};font-family:{_B["mono"]};'
        f'margin-bottom:5px;margin-top:4px">🚩 Red Flags Detectados</div>',
        unsafe_allow_html=True,
    )
    if flags:
        for f in flags:
            st.markdown(f'<div class="flag-item">{f}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="color:{_B["green"]};font-size:0.83rem;'
            f'font-family:{_B["mono"]};padding:8px 0">'
            f'Nenhum red flag acima do limiar detectado.</div>',
            unsafe_allow_html=True,
        )

    if fd_t is not None and fd_t1 is not None:
        with st.expander("▶ Auditoria de Dados — Entradas do Modelo (YoY)", expanded=False):
            st.markdown(_audit_table_html(fd_t, fd_t1, year_t), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Dashboard Setorial
# ─────────────────────────────────────────────────────────────────────────────
def _tab_dashboard(year_t: int) -> None:
    st.caption("Empresas não-financeiras B3 · Beneish M-Score · dados via Portal CVM")
    col_refresh, _ = st.columns([1, 7])
    with col_refresh:
        force_refresh = st.button("↺ Atualizar", key="home_refresh")

    if force_refresh:
        _market_df.clear()       # invalidate TTL cache
        st.rerun()

    df = _market_df()

    if df is None:
        with st.spinner("Carregando dados ao vivo..."):
            try:
                _, _, _, get_home, _ = _rank_market()
                df = get_home(year_t=year_t, force=False, quick=True)
            except Exception as exc:
                st.error(f"Erro ao carregar dados: {exc}")
                return

    import pandas as pd
    ok = df[df["Score de Risco"].notna()].copy()
    if "Setor" in ok.columns:
        ok = ok[~ok["Setor"].isin(_FIN_ALL)].copy()

    if ok.empty:
        st.markdown(
            f'<div style="text-align:center;padding:48px 20px">'
            f'<div style="font-size:2.5rem;margin-bottom:14px">📊</div>'
            f'<div style="font-family:{_B["mono"]};color:{_B["text"]};font-size:0.92rem;'
            f'font-weight:700;margin-bottom:10px">Cache de mercado ainda não disponível</div>'
            f'<div style="font-family:{_B["mono"]};color:{_B["muted"]};font-size:0.80rem;'
            f'line-height:1.7;margin-bottom:24px">'
            f'O Dashboard exibe o ranking completo após o cache ser gerado.<br>'
            f'Para analisar uma empresa agora, use a aba '
            f'<strong style="color:{_B["orange"]}">🔍 Análise Individual</strong>.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    _icon   = {"Crítico": "🔴", "Alto Risco": "🟠", "Atenção": "🟡", "Normal": "🟢"}
    n_total = len(ok)
    n_risk  = int(ok["Nível de Alerta"].isin(["Crítico", "Alto Risco"]).sum())
    st.caption(f"📊 {n_total} empresas avaliadas · {n_risk} em risco elevado (M-Score > −1.78)")

    if st.session_state.get("selected_ticker"):
        nav = st.session_state["selected_ticker"]
        st.info(
            f"Ticker selecionado: **{nav}** — "
            f"vá para a aba **🔍 Análise Individual** para ver o resultado."
        )

    worst10 = ok.nlargest(10, "M-Score")
    best10  = ok.nsmallest(10, "M-Score")

    def _col_header(title: str, border: str) -> None:
        st.markdown(
            f'<div style="background:{_B["card"]};border:1px solid {border};'
            f'border-radius:6px 6px 0 0;padding:9px 14px">'
            f'<span style="font-size:0.7rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.1em;color:{border};font-family:{_B["mono"]}">{title}</span>'
            f'<span style="float:right;font-size:0.65rem;color:{_B["muted"]};'
            f'font-family:{_B["mono"]}">Top 10</span></div>',
            unsafe_allow_html=True,
        )

    col_w, col_b = st.columns(2)

    with col_w:
        _col_header("🔴 10 Maiores Riscos", "#ef4444")
        for _, row in worst10.iterrows():
            t     = str(row.get("Ticker", row.get("Nome", "—")))
            alert = row.get("Nível de Alerta", "—")
            score = f"{row.get('M-Score', float('nan')):+.3f}"
            setor = row.get("Setor", "")
            if st.button(
                f"{_icon.get(alert,'⚪')} **{t}** `{score}` · {setor}",
                key=f"hw_{t}", use_container_width=True,
            ):
                st.session_state["selected_ticker"] = t
                st.session_state["gemini_report"]   = None
                st.rerun()

    with col_b:
        _col_header("🟢 10 Menores Riscos", "#10b981")
        for _, row in best10.iterrows():
            t     = str(row.get("Ticker", row.get("Nome", "—")))
            alert = row.get("Nível de Alerta", "—")
            score = f"{row.get('M-Score', float('nan')):+.3f}"
            setor = row.get("Setor", "")
            if st.button(
                f"{_icon.get(alert,'⚪')} **{t}** `{score}` · {setor}",
                key=f"hb_{t}", use_container_width=True,
            ):
                st.session_state["selected_ticker"] = t
                st.session_state["gemini_report"]   = None
                st.rerun()

    if len(ok) > 1:
        st.divider()
        st.plotly_chart(_sector_bar(ok), use_container_width=True,
                        config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Análise Individual
# ─────────────────────────────────────────────────────────────────────────────
def _tab_analise(year_t: int) -> None:
    st.caption("Dados reais do Portal CVM · M-Score + Accruals + Narrativa IA")

    # Dropdown: start with fast static list; user can request full CVM universe
    if st.session_state.get("use_full_opts", False):
        with st.spinner("Carregando universo completo B3..."):
            all_labels, all_values, lbl_to_val = _full_opts()
    else:
        all_labels, all_values, lbl_to_val = _static_opts()

    nav_ticker  = st.session_state.get("selected_ticker", "")
    default_idx = 0
    if nav_ticker:
        for i, v in enumerate(all_values):
            if v.upper() == nav_ticker.upper():
                default_idx = i + 1
                break

    col_dd, col_cnpj, col_btn = st.columns([3, 2, 1])
    with col_dd:
        selected_label = st.selectbox(
            "Ticker B3", [""] + all_labels,
            index=default_idx,
            placeholder="Selecione um ticker B3...",
            key="analise_dd",
        )
    with col_cnpj:
        cnpj_input = st.text_input(
            "CNPJ / busca livre",
            value="",
            placeholder="ex: 33.000.167/0001-01",
            key="analise_cnpj",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        calc_clicked = st.button("Calcular", use_container_width=True, key="analise_btn")

    # Lazy-load full CVM list on demand — ação secundária, menor destaque
    if not st.session_state.get("use_full_opts", False):
        _, col_load, _ = st.columns([3, 2, 3])
        with col_load:
            if st.button("⬇ Ampliar lista (universo CVM completo)", key="load_full_opts"):
                st.session_state["use_full_opts"] = True
                st.rerun()

    ticker_val = lbl_to_val.get(selected_label, selected_label) if selected_label else ""
    query      = cnpj_input.strip() or ticker_val.strip()

    # ── Run analysis ─────────────────────────────────────────────────────────
    if calc_clicked and query:
        from advisor_brain_fsa.data_fetcher import CVMDataFetcher
        from advisor_brain_fsa.cvm_registry import CVMRegistry

        try:
            profile = CVMRegistry.get_instance().get_company_profile(query)
        except Exception:
            profile = None
        if profile:
            st.markdown(_profile_card_html(profile), unsafe_allow_html=True)
            _render_company_ai_desc(profile)

        with st.spinner("Calculando M-Score..."):
            try:
                fetcher     = CVMDataFetcher()
                fd_t, fd_t1 = fetcher.get_financial_data(query, year_t=year_t, year_t1=year_t - 1)
            except Exception as exc:
                st.error(f"Erro ao buscar dados: {exc}")
                return
            get_scorer = _scorer_mod()
            _, _, get_sector = _ticker_map()
            try:
                sector = get_sector(query)
                sr     = get_scorer(sector).score(fd_t, fd_t1)
            except Exception as exc:
                st.error(f"Erro no cálculo: {exc}")
                st.code(traceback.format_exc())
                return

        st.session_state["last_result"] = {
            "sr": sr, "fd_t": fd_t, "fd_t1": fd_t1,
            "query": query, "sector": sector, "year_t": year_t,
        }
        st.session_state["analyze_result"] = {
            "query": query, "sector": sector, "year_t": year_t,
            "ms_score": sr.mscore_result.m_score if sr.mscore_result else None,
            "accrual":  sr.cfq_result.accrual_ratio if sr.cfq_result else None,
            "flags": sr.red_flags, "alert": sr.alert_level.value,
        }
        st.session_state["gemini_report"] = None
        _render_result(query, sector, sr, year_t, fd_t=fd_t, fd_t1=fd_t1)

    elif not calc_clicked and st.session_state.get("last_result"):
        r = st.session_state["last_result"]
        try:
            from advisor_brain_fsa.cvm_registry import CVMRegistry
            profile = CVMRegistry.get_instance().get_company_profile(r["query"])
            if profile:
                st.markdown(_profile_card_html(profile), unsafe_allow_html=True)
        except Exception:
            pass
        st.caption("Resultado anterior — clique em Calcular para atualizar.")
        _render_result(r["query"], r["sector"], r["sr"], r["year_t"],
                       fd_t=r["fd_t"], fd_t1=r["fd_t1"])

    elif not query:
        st.markdown(
            f'<div style="color:{_B["muted"]};font-family:{_B["mono"]};'
            f'font-size:0.83rem;padding:20px 0">'
            f'Selecione um ticker ou informe um CNPJ.</div>',
            unsafe_allow_html=True,
        )

    # ── Gemini section ────────────────────────────────────────────────────────
    cache = st.session_state.get("analyze_result")
    if cache:
        st.divider()
        st.markdown(
            f'<div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.12em;color:{_B["muted"]};font-family:{_B["mono"]};'
            f'margin-bottom:8px">🤖 Tese de Risco — Narrativa Gemini</div>',
            unsafe_allow_html=True,
        )
        if st.button("Gerar Tese de Risco com Gemini", key="gemini_btn"):
            key = _api_key()
            if not key:
                st.error("Configure GOOGLE_API_KEY como Streamlit Secret ou variável de ambiente.")
            else:
                _run_gemini(cache, key)

        if st.session_state.get("gemini_report"):
            report, q, yt = st.session_state["gemini_report"]
            st.markdown(
                f'<div style="background:{_B["bg"]};color:{_B["text"]};'
                f'font-family:{_B["mono"]};font-size:0.83rem;padding:16px 20px;'
                f'border:1px solid {_B["border"]};border-left:3px solid {_B["orange"]};'
                f'border-radius:4px;line-height:1.7">{report}</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇️ Baixar .md", data=report,
                file_name=f"tese_{q}_{yt}.md", mime="text/markdown",
                key="gemini_dl",
            )


def _render_company_ai_desc(profile: dict) -> None:
    key = _api_key()
    if not key:
        return
    company_name = profile.get("denom_comerc") or profile.get("denom_social") or "a empresa"
    setor        = profile.get("setor_ativ") or profile.get("sector_label") or ""
    prompt = (
        f"Você é um analista de mercado brasileiro especializado em contabilidade forense. "
        f"Em exatamente 3 frases curtas e objetivas, descreva '{company_name}': "
        f"(1) o que ela faz e em qual segmento atua; "
        f"(2) sua relevância/porte no mercado brasileiro; "
        f"(3) seu principal modelo de geração de receita. "
        f"Contexto adicional — setor CVM: {setor}. "
        f"Não mencione preço de ação, não use disclaimers."
    )
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        resp = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        desc = resp.text.strip()
        if desc:
            st.markdown(
                f'<div style="margin-top:8px;margin-bottom:10px">'
                f'<div style="color:{_B["muted"]};font-family:{_B["mono"]};font-size:0.70rem;'
                f'font-weight:700;letter-spacing:0.1em;margin-bottom:5px">PERFIL DA EMPRESA</div>'
                f'<div style="color:{_B["text"]};font-family:{_B["mono"]};font-size:0.83rem;'
                f'line-height:1.55;border-left:3px solid {_B["orange"]};padding-left:10px">'
                f'{desc}</div></div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass


def _run_gemini(cache: dict, key: str) -> None:
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    GeminiAnalyst, _ = _mda_mod()
    get_scorer       = _scorer_mod()
    q, yt, sec       = cache["query"], cache["year_t"], cache["sector"]
    with st.spinner("Gerando narrativa Gemini..."):
        try:
            fd_t, fd_t1 = CVMDataFetcher().get_financial_data(q, year_t=yt, year_t1=yt - 1)
            sr          = get_scorer(sec).score(fd_t, fd_t1)
            analyst     = GeminiAnalyst(api_key=key)
            report      = "".join(analyst.analyze_streaming(
                ticker=q, sector=sec, year=yt,
                mscore_result=sr.mscore_result,
                cfq_result=sr.cfq_result,
                red_flags=sr.red_flags or [],
            ))
            st.session_state["gemini_report"] = (report, q, yt)
        except Exception as exc:
            st.error(f"Erro na API Gemini: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Ranking de Mercado
# ─────────────────────────────────────────────────────────────────────────────
def _tab_ranking(year_t: int) -> None:
    st.caption("Processa múltiplos tickers e ordena por nível de risco.")

    rank_labels, lbl_to_ticker = _ranking_opts()
    default_lbl = [
        lbl for lbl in rank_labels
        if any(lbl.startswith(t) for t in
               ["PETR4", "VALE3", "ABEV3", "ELET3", "WEGE3", "RENT3", "MGLU3"])
    ]

    selected = st.multiselect(
        "Tickers", rank_labels, default=default_lbl,
        label_visibility="collapsed", key="rank_tickers",
    )
    run = st.button("Calcular Ranking", key="rank_btn")

    if run:
        if not selected:
            st.warning("Selecione ao menos um ticker.")
            return
        _compute_ranking([lbl_to_ticker[l] for l in selected if l in lbl_to_ticker], year_t)

    rdf = st.session_state.get("ranking_df")
    if rdf is not None and not run:
        st.caption("Resultado anterior:")
        _show_ranking(rdf)


def _compute_ranking(tickers: list, year_t: int) -> None:
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    CompanyResult, apply_stats, to_df, _, _ = _rank_market()
    get_scorer = _scorer_mod()
    _, _, get_sector = _ticker_map()

    fetcher = CVMDataFetcher()
    results = []
    prog    = st.progress(0, text="Iniciando...")
    for i, ticker in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Analisando {ticker}…")
        sector = get_sector(ticker)
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t - 1)
            sr = get_scorer(sector).score(fd_t, fd_t1)
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, sector_risk=sr))
        except Exception as exc:
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, error=str(exc)))
        time.sleep(0.05)
    prog.empty()

    apply_stats(results)
    df = to_df(results, top_flags=3)
    st.session_state["ranking_df"] = df
    _show_ranking(df)


def _show_ranking(df) -> None:
    ok  = df[df["Nível de Alerta"] != "N/D"].copy()
    err = df[df["Nível de Alerta"] == "N/D"].copy()

    if not ok.empty:
        counts = ok["Nível de Alerta"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 CRÍTICO",    str(counts.get("Crítico",    0)))
        c2.metric("🟠 ALTO RISCO", str(counts.get("Alto Risco", 0)))
        c3.metric("🟡 ATENÇÃO",    str(counts.get("Atenção",    0)))
        c4.metric("🟢 NORMAL",     str(counts.get("Normal",     0)))
        st.divider()

        want     = ["Ticker", "Setor", "Score de Risco", "M-Score",
                    "Nível de Alerta", "Accrual Ratio", "Δ vs Grupo", "Red Flag 1"]
        cols_use = [c for c in want if c in ok.columns]
        display  = ok[cols_use].copy()

        def _style_row(row):
            styles  = [""] * len(row)
            idx_map = {c: i for i, c in enumerate(row.index)}
            alert   = row.get("Nível de Alerta", "")
            ms      = row.get("M-Score")
            if "M-Score" in idx_map and ms is not None:
                try:
                    styles[idx_map["M-Score"]] = (
                        "color:#FF3E3E;font-weight:700"
                        if float(ms) > -1.78 else "color:#00FF00;font-weight:700"
                    )
                except Exception:
                    pass
            if "Ticker" in idx_map:
                styles[idx_map["Ticker"]] = "color:#FFA500;font-weight:700"
            alert_style = {
                "Crítico":    "color:#FF3E3E;font-weight:700;background-color:#2a0000",
                "Alto Risco": "color:#FFA500;font-weight:700;background-color:#2a1a00",
                "Atenção":    "color:#FFD700;font-weight:700;background-color:#2a2000",
                "Normal":     "color:#00FF00;font-weight:700;background-color:#002200",
            }
            if "Nível de Alerta" in idx_map:
                styles[idx_map["Nível de Alerta"]] = alert_style.get(alert, "")
            return styles

        numeric_cols = {c: lambda x: f"{x:+.4f}" if x is not None else "—"
                        for c in ["Score de Risco", "M-Score", "Accrual Ratio", "Δ vs Grupo"]
                        if c in display.columns}
        styled = display.style.apply(_style_row, axis=1).format(numeric_cols)
        st.dataframe(styled, use_container_width=True,
                     height=min(600, len(display) * 38 + 60))

        b_df = ok[ok["Scorer"] == "beneish"] if "Scorer" in ok.columns else ok
        if len(b_df) > 1:
            st.divider()
            st.plotly_chart(_sector_bar(b_df), use_container_width=True,
                            config={"displayModeBar": False})

    if not err.empty:
        with st.expander(f"⚠️ {len(err)} ticker(s) sem dados"):
            for _, r in err.iterrows():
                st.caption(f"{r.get('Ticker','?')} — {r.get('Erro','erro desconhecido')}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)   # pre-generated — no f-string overhead
    _init_state()

    t_col, y_col, api_col, v_col = st.columns([5, 1, 1, 1])
    with t_col:
        st.markdown(
            f'<span style="font-family:{_B["mono"]};font-size:0.95rem;font-weight:700;'
            f'color:{_B["orange"]};letter-spacing:0.08em">ADVISOR-BRAIN-FSA</span>'
            f'<span style="font-size:0.68rem;color:{_B["muted"]};margin-left:10px">'
            f'· Qualidade de Relatórios Financeiros · B3</span>',
            unsafe_allow_html=True,
        )
    with y_col:
        st.markdown(
            f'<div style="font-family:{_B["mono"]};font-size:0.60rem;color:{_B["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-14px">ANO</div>',
            unsafe_allow_html=True,
        )
        year_t = st.selectbox(
            "Ano", _YEAR_OPTS, index=0,
            label_visibility="collapsed", key="year_sel",
        )
    with api_col:
        ok_key = bool(_api_key())
        label  = "■ Gemini: ativo" if ok_key else "■ Gemini: inativo"
        color  = "#00FF00" if ok_key else "#FF3E3E"
        tip    = "" if ok_key else ' title="Configure GOOGLE_API_KEY nos Secrets do Streamlit"'
        st.markdown(
            f'<span{tip} style="color:{color};font-family:{_B["mono"]};font-size:0.72rem;'
            f'cursor:{"default" if ok_key else "help"}">{label}</span>',
            unsafe_allow_html=True,
        )
    with v_col:
        st.markdown(
            f'<span style="font-size:0.68rem;color:{_B["orange"]};font-family:{_B["mono"]};'
            f'font-weight:700;border:1px solid {_B["border"]};padding:1px 7px;'
            f'border-radius:3px">{_APP_VERSION}</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<hr style="border-top:1px solid {_B["border"]};margin:6px 0 10px 0">',
        unsafe_allow_html=True,
    )

    tab_home, tab_analise, tab_ranking = st.tabs([
        "🏠 Dashboard", "🔍 Análise Individual", "📊 Ranking"
    ])
    with tab_home:
        try:
            _tab_dashboard(year_t)
        except Exception:
            st.error("Dashboard error:")
            st.code(traceback.format_exc())
    with tab_analise:
        try:
            _tab_analise(year_t)
        except Exception:
            st.error("Análise error:")
            st.code(traceback.format_exc())
    with tab_ranking:
        try:
            _tab_ranking(year_t)
        except Exception:
            st.error("Ranking error:")
            st.code(traceback.format_exc())

    st.markdown(
        f'<hr style="border-top:1px solid {_B["border"]};margin:32px 0 8px 0">',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="text-align:center;font-family:{_B["mono"]};font-size:0.65rem;'
        f'color:{_B["muted"]};line-height:1.6;padding:4px 0 12px">'
        f'CFA Institute: indicadores quantitativos são sinais de alerta, não prova de fraude. '
        f'IA: narrativas geradas por Gemini 2.5 Flash — uso educacional. '
        f'Não é recomendação de investimento.'
        f'</div>',
        unsafe_allow_html=True,
    )


try:
    main()
except Exception:
    st.error("Erro interno — veja o traceback abaixo:")
    st.code(traceback.format_exc())
