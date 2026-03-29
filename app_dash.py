"""
Advisor-Brain-FSA — Dash App (produção)
=========================================
Bloomberg Terminal theme · Deploy no Render via gunicorn

Deploy:    gunicorn app_dash:server
Local dev: python app_dash.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, no_update, ctx
from dash.exceptions import PreventUpdate

from advisor_brain_fsa.mda_analyst import GeminiAnalyst, compute_grade, compute_grade_financial
from advisor_brain_fsa.rank_market import (
    DEFAULT_WATCHLIST, CompanyResult, _apply_sector_stats, _to_dataframe,
    get_home_dashboard_data,
)
from advisor_brain_fsa.sector_scorer import SectorRiskResult, get_scorer
from advisor_brain_fsa.ticker_map import (
    TICKER_TO_KEYWORD, SECTOR_LABELS, TICKER_SECTOR, FINANCIAL_GROUP, get_sector,
)


# ─────────────────────────────────────────────────────────────────────────────
# App + server  (essencial para gunicorn / Render)
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Advisor-Brain-FSA",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # ← ponto de entrada para gunicorn / Railway / Render

# ── /health — rota Flask leve para Railway healthcheck ───────────────────────
# Railway bate em healthcheckPath: "/health" (railway.json) antes do Dash
# responder. Esta rota retorna imediatamente, sem depender do layout Dash.
@server.route("/health")
def _health():
    return {"status": "ok", "app": "advisor-brain-fsa"}, 200

# ─────────────────────────────────────────────────────────────────────────────
# Constantes Bloomberg
# ─────────────────────────────────────────────────────────────────────────────

_B = {
    "bg":     "#000000", "card":   "#1C1C1C", "border": "#30363D",
    "text":   "#D1D1D1", "green":  "#00FF00", "red":    "#FF3E3E",
    "orange": "#FFA500", "yellow": "#FFD700", "muted":  "#888888",
    "mono":   "'JetBrains Mono','Roboto Mono','Fira Code','Consolas',monospace",
}

_CY = date.today().year
_YEAR_OPTS  = [{"label": str(y), "value": y} for y in range(_CY - 1, _CY - 6, -1)]
_TICK_OPTS  = [
    {"label": f"{t} — {TICKER_TO_KEYWORD.get(t, t)}", "value": t}
    for t in sorted(TICKER_TO_KEYWORD.keys())
]
_MULTI_OPTS = [{"label": f"{t} — {TICKER_TO_KEYWORD.get(t, t)}", "value": t}
               for t in sorted(TICKER_TO_KEYWORD.keys())]

_DEFAULT_WL = DEFAULT_WATCHLIST[:24]  # watchlist padrao

def _api_key() -> str:
    return os.environ.get("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Plotly — layout Bloomberg
# ─────────────────────────────────────────────────────────────────────────────

_BBG_LAYOUT = dict(
    paper_bgcolor=_B["bg"], plot_bgcolor=_B["bg"],
    font=dict(family=_B["mono"], color=_B["text"]),
    margin=dict(l=40, r=20, t=40, b=40),
)

def _gauge(m_score: float) -> go.Figure:
    clr = _B["red"] if m_score > -1.78 else _B["green"]
    fig = go.Figure(go.Bar(
        x=[m_score], y=["M-Score"], orientation="h",
        marker_color=clr,
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

def _radar(ms) -> go.Figure:
    cats  = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA"]
    vals  = [ms.dsri, ms.gmi, ms.aqi, ms.sgi, ms.depi, ms.sgai, ms.lvgi,
             max(0, ms.tata * 10 + 1)]
    thr   = [1.031, 1.014, 1.039, 1.134, 1.017, 1.054, 1.000, 1.000]
    fig   = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=thr+[thr[0]], theta=cats+[cats[0]], name="Limiar",
        line_color=_B["orange"], line_dash="dot", fill="none"))
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], name="Empresa",
        line_color=_B["red"], fill="toself", fillcolor=_B["red"]+"22"))
    fig.update_layout(**_BBG_LAYOUT,
        title=dict(text="Índices Beneish", font=dict(size=10, color=_B["muted"])),
        polar=dict(bgcolor=_B["card"],
                   radialaxis=dict(visible=True, range=[0,2], gridcolor=_B["border"]),
                   angularaxis=dict(gridcolor=_B["border"])),
        legend=dict(font=dict(size=8, color=_B["muted"]), bgcolor=_B["bg"]),
        height=320,
    )
    return fig

def _sector_bar(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Setor")["M-Score"].mean().dropna().sort_values()
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
# DataTable Bloomberg
# ─────────────────────────────────────────────────────────────────────────────

_DT_TABLE  = {"backgroundColor": _B["bg"], "border": f"1px solid {_B['border']}",
              "borderRadius": "4px", "overflowX": "auto"}
_DT_HEADER = {"backgroundColor": _B["card"], "color": _B["orange"],
              "fontFamily": _B["mono"], "fontWeight": "700",
              "fontSize": "0.68rem", "textTransform": "uppercase",
              "letterSpacing": "0.08em", "border": f"1px solid {_B['border']}",
              "padding": "10px 12px"}
_DT_CELL   = {"backgroundColor": _B["bg"], "color": _B["text"],
              "fontFamily": _B["mono"], "fontSize": "0.82rem",
              "border": f"1px solid {_B['border']}", "padding": "7px 12px",
              "textAlign": "left"}
_DT_COND   = [
    {"if": {"filter_query": '{Nível de Alerta} = "Crítico"',
            "column_id": "Nível de Alerta"},
     "backgroundColor": "#2a0000", "color": _B["red"], "fontWeight": "700"},
    {"if": {"filter_query": '{Nível de Alerta} = "Alto Risco"',
            "column_id": "Nível de Alerta"},
     "backgroundColor": "#2a1a00", "color": _B["orange"], "fontWeight": "700"},
    {"if": {"filter_query": '{Nível de Alerta} = "Atenção"',
            "column_id": "Nível de Alerta"},
     "backgroundColor": "#2a2000", "color": _B["yellow"], "fontWeight": "700"},
    {"if": {"filter_query": '{Nível de Alerta} = "Normal"',
            "column_id": "Nível de Alerta"},
     "backgroundColor": "#002200", "color": _B["green"], "fontWeight": "700"},
    {"if": {"filter_query": "{M-Score} > -1.78", "column_id": "M-Score"},
     "color": _B["red"], "fontWeight": "700"},
    {"if": {"filter_query": "{M-Score} <= -1.78", "column_id": "M-Score"},
     "color": _B["green"], "fontWeight": "700"},
    {"if": {"column_id": "Ticker"},
     "color": _B["orange"], "fontWeight": "700"},
    {"if": {"row_index": "odd"}, "backgroundColor": "#050505"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Pequenos helpers HTML
# ─────────────────────────────────────────────────────────────────────────────

def _card(children, extra=None):
    s = {"background": _B["card"], "border": f"1px solid {_B['border']}",
         "borderRadius": "6px", "padding": "12px 16px", "marginBottom": "10px"}
    if extra: s.update(extra)
    return html.Div(children, style=s)

def _pill(alert: str) -> html.Span:
    m = {"Crítico": (_B["red"],"#2a000044"), "Alto Risco": (_B["orange"],"#2a1a0044"),
         "Atenção": (_B["yellow"],"#2a200044"), "Normal": (_B["green"],"#00220044")}
    c, bg = m.get(alert, (_B["muted"], "#11111144"))
    return html.Span(alert, style={"display":"inline-block","padding":"2px 10px",
        "borderRadius":"3px","border":f"1px solid {c}","color":c,"background":bg,
        "fontWeight":"700","fontSize":"0.76rem","letterSpacing":"0.04em",
        "fontFamily":_B["mono"]})

def _lbl(txt):
    return html.Div(txt, style={"fontSize":"0.62rem","fontWeight":"700",
        "textTransform":"uppercase","letterSpacing":"0.12em",
        "color":_B["muted"],"fontFamily":_B["mono"],"marginBottom":"5px","marginTop":"12px"})

def _mono(txt, color=None, size="0.88rem", bold=False):
    return html.Span(str(txt), style={"fontFamily":_B["mono"],
        "color": color or _B["text"], "fontSize":size,
        "fontWeight":"700" if bold else "normal"})

def _grade_badge(grade: str) -> html.Div:
    colors = {"A":"#00FF00","B":"#7FFF00","C":"#FFD700","D":"#FFA500","F":"#FF3E3E"}
    return html.Div(grade, style={"display":"inline-flex","alignItems":"center",
        "justifyContent":"center","width":"64px","height":"64px","borderRadius":"4px",
        "fontSize":"2.2rem","fontWeight":"900","color":"#000000",
        "background": colors.get(grade,"#888888"),"fontFamily":_B["mono"]})

def _divider():
    return html.Hr(style={"borderTop":f"1px solid {_B['border']}","margin":"14px 0"})

def _metric_card(label, value, delta="", status="ok"):
    c = {"ok":_B["green"],"warn":_B["yellow"],"crit":_B["red"]}.get(status, _B["text"])
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className=f"metric-value {status}"),
        *([ html.Div(delta, className="metric-delta") ] if delta else []),
    ], className="metric-card")

# ─────────────────────────────────────────────────────────────────────────────
# Layout principal
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    # Stores de sessão (persistem entre callbacks, resolvem o problema do Gemini)
    dcc.Store(id="analyze-store",  storage_type="session"),
    dcc.Store(id="ranking-store",  storage_type="session"),
    dcc.Store(id="home-sel-store", storage_type="session"),

    # ── Topbar ───────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("ADVISOR-BRAIN-FSA", style={
                "fontFamily":_B["mono"],"fontSize":"0.95rem",
                "fontWeight":"700","color":_B["orange"],"letterSpacing":"0.08em"}),
            html.Span(" · Qualidade de Relatórios Financeiros · B3",
                      style={"fontSize":"0.68rem","color":_B["muted"],"marginLeft":"10px"}),
        ], style={"flex":"1"}),
        # Seletor de ano inline
        # NOTA: width deve ser definido no wrapper div, NAO no style do Dropdown.
        # React-Select herda o container width; se <=90px o arrow (~30px) +
        # padding interno (~16px) deixam <44px para o texto e o overflow:hidden
        # do .Select-value-label clipa "2025" para apenas "2".
        html.Div([
            html.Span("ANO ", style={"fontSize":"0.62rem","color":_B["muted"],
                                     "fontFamily":_B["mono"]}),
            html.Div(
                dcc.Dropdown(id="year-dd", options=_YEAR_OPTS, value=_CY-1,
                    clearable=False, searchable=False,
                    style={"backgroundColor":_B["card"],"border":"none",
                           "fontFamily":_B["mono"],"fontSize":"0.82rem"}),
                style={"width":"116px","minWidth":"116px"},
            ),
        ], style={"display":"flex","alignItems":"center","gap":"6px","marginRight":"18px"}),
        html.Div(id="api-status",
                 style={"fontSize":"0.72rem","fontFamily":_B["mono"],"marginRight":"18px"}),
        html.Span("v0.7.0", style={"fontSize":"0.62rem","color":"#333333","fontFamily":_B["mono"]}),
    ], style={"display":"flex","alignItems":"center","padding":"8px 20px",
              "background":_B["bg"],"borderBottom":f"1px solid {_B['border']}",
              "position":"sticky","top":"0","zIndex":"1000"}),

    # ── Tabs ─────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="home",
        children=[
            dcc.Tab(label="🏠 Dashboard",          value="home",
                    className="bbg-tab", selected_className="bbg-tab-selected"),
            dcc.Tab(label="🔍 Análise Individual", value="analise",
                    className="bbg-tab", selected_className="bbg-tab-selected"),
            dcc.Tab(label="📊 Ranking",             value="ranking",
                    className="bbg-tab", selected_className="bbg-tab-selected"),
        ],
        style={"borderBottom":f"1px solid {_B['border']}","backgroundColor":_B["bg"]},
        colors={"border":_B["border"],"primary":_B["orange"],"background":_B["bg"]},
    ),
    html.Div(id="tab-content",
             style={"padding":"14px 20px","paddingBottom":"80px","backgroundColor":_B["bg"]}),

    # ── Footer legal fixo ────────────────────────────────────────────────────
    html.Div([
        html.Span("CFA Institute: ", style={"color":_B["orange"],"fontWeight":"700"}),
        "Indicadores quantitativos são sinais de alerta, não prova de fraude. ",
        html.Span("IA: ", style={"color":_B["orange"],"fontWeight":"700"}),
        "Narrativas geradas por Gemini 2.5 Flash — uso educacional. ",
        html.Span("Não é recomendação de investimento.",
                  style={"fontWeight":"700"}),
    ], className="legal-footer"),

], style={"backgroundColor":_B["bg"],"minHeight":"100vh"})

# ─────────────────────────────────────────────────────────────────────────────
# Callbacks globais
# ─────────────────────────────────────────────────────────────────────────────

@callback(Output("api-status","children"), Input("tabs","value"))
def _api_status(_):
    if _api_key():
        return html.Span("■ GEMINI OK",  style={"color":_B["green"]})
    return html.Span("■ SEM API KEY", style={"color":_B["red"]})


@callback(Output("tab-content","children"),
          Input("tabs","value"),
          State("year-dd","value"))
def _render_tab(tab, year_t):
    if   tab == "home":    return _layout_home()
    elif tab == "analise": return _layout_analise()
    elif tab == "ranking": return _layout_ranking()
    return html.Div("Tab desconhecida")


# ─────────────────────────────────────────────────────────────────────────────
# TAB HOME — Dashboard Setorial auto-load
# ─────────────────────────────────────────────────────────────────────────────

def _layout_home():
    return html.Div([
        html.Div([
            html.Div([
                html.Span("Dashboard Setorial B3",
                          style={"fontSize":"1.15rem","fontWeight":"700","color":_B["text"]}),
                html.Span(" · Top 5 por risco · Industriais / Bancos / Seguros · clique ↺ para análise completa (121 tickers)",
                          style={"fontSize":"0.75rem","color":_B["muted"],"marginLeft":"10px"}),
            ]),
            html.Button("↺ Atualizar", id="home-refresh", n_clicks=0,
                        style={"marginLeft":"auto","padding":"5px 14px",
                               "fontSize":"0.75rem","cursor":"pointer"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"12px"}),
        # Intervalo: dispara UMA vez em 200ms para auto-load
        dcc.Interval(id="home-init", interval=200, max_intervals=1),
        dcc.Loading(type="circle", color=_B["orange"],
            children=html.Div(id="home-content")),
        html.Div(id="home-drill"),
    ])


@callback(
    Output("home-content","children"),
    Input("home-init","n_intervals"),
    Input("home-refresh","n_clicks"),
    State("year-dd","value"),
    prevent_initial_call=False,
)
def _load_home(_ni, _nb, year_t):
    year_t = year_t or (_CY - 1)

    # Force rebuild only when the ↺ button triggered this callback.
    # Initial auto-load (home-init) uses the disk cache if it is fresh.
    force_refresh = (ctx.triggered_id == "home-refresh")
    # On auto-load (home-init), use quick mode so gunicorn never blocks.
    # quick=True scores only DEFAULT_WATCHLIST (~27 tickers, ~10s) when no
    # disk cache exists.  Full 121-ticker rebuild is triggered only by the ↺ button.
    quick_mode = not force_refresh

    try:
        df = get_home_dashboard_data(year_t=year_t, force=force_refresh, quick=quick_mode)
    except Exception as exc:
        return html.Div(
            f"Erro ao carregar dados: {exc}",
            style={"color": _B["red"], "fontFamily": _B["mono"], "fontSize": "0.83rem"},
        )

    ok = df[df["Score de Risco"].notna()].copy()
    if ok.empty:
        return html.Div("Nenhum dado disponível.", style={"color":_B["muted"]})

    def _top5_col(scorer_type, title, icon, border):
        sub = ok[ok["Scorer"] == scorer_type]
        top = sub.nlargest(5, "Score de Risco") if not sub.empty else sub
        _icon = {"Crítico":"🔴","Alto Risco":"🟠","Atenção":"🟡","Normal":"🟢"}
        rows = []
        for _, row in top.iterrows():
            t     = row["Ticker"]
            alert = row.get("Nível de Alerta","—")
            score = (f"{row.get('M-Score',float('nan')):+.3f}"
                     if scorer_type=="beneish" else
                     f"{row.get('Score de Risco',0):.1f}/10")
            setor = row.get("Setor","")
            rows.append(html.Div([
                html.Span(f"{_icon.get(alert,'⚪')} ", style={"marginRight":"4px"}),
                html.Span(t, style={"color":_B["orange"],"fontWeight":"700","marginRight":"8px"}),
                html.Span(f"`{score}`", style={"color":_B["text"],"fontSize":"0.8rem"}),
                html.Span(f" · {setor}", style={"color":_B["muted"],"fontSize":"0.75rem"}),
            ], style={"padding":"7px 14px","background":_B["bg"],
                      "border":f"1px solid {_B['border']}","borderTop":"none",
                      "fontFamily":_B["mono"],"fontSize":"0.83rem"}))
        return html.Div([
            html.Div([
                html.Span(f"{icon} {title}",
                          style={"fontSize":"0.7rem","fontWeight":"700",
                                 "textTransform":"uppercase","letterSpacing":"0.1em",
                                 "color":border}),
                html.Span("Top 5", style={"float":"right","fontSize":"0.65rem","color":_B["muted"]}),
            ], style={"background":_B["card"],"border":f"1px solid {border}",
                      "borderRadius":"6px 6px 0 0","padding":"9px 14px"}),
            *rows,
        ])

    col_b, col_bk, col_i = _top5_col("beneish","Industriais","🏭","#3b82f6"),                             _top5_col("banking","Bancos & Financeiro","🏦","#f59e0b"),                             _top5_col("insurance","Seguros","🛡️","#10b981")
    charts = []
    b_df = ok[ok["Scorer"] == "beneish"]
    if len(b_df) > 1:
        charts = [_divider(),
                  html.Div("M-Score Médio por Setor (Industriais)",
                           style={"fontSize":"0.7rem","fontWeight":"700","textTransform":"uppercase",
                                  "letterSpacing":"0.08em","color":_B["muted"],"fontFamily":_B["mono"]}),
                  dcc.Graph(figure=_sector_bar(b_df), config={"displayModeBar":False})]
    return html.Div([
        html.Div([
            html.Div(col_b,  style={"flex":"1","minWidth":"0"}),
            html.Div(col_bk, style={"flex":"1","minWidth":"0"}),
            html.Div(col_i,  style={"flex":"1","minWidth":"0"}),
        ], style={"display":"flex","gap":"12px"}),
        *charts,
    ])

# ─────────────────────────────────────────────────────────────────────────────
# TAB ANALISE INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def _layout_analise():
    return html.Div([
        html.Div([
            html.Span("Análise Individual",
                      style={"fontSize":"1.15rem","fontWeight":"700","color":_B["text"]}),
            html.Div("Dados reais do Portal CVM · M-Score + Accruals + Narrativa IA",
                     style={"fontSize":"0.75rem","color":_B["muted"],"marginTop":"2px"}),
        ], style={"marginBottom":"14px"}),
        # ── Barra de entrada ────────────────────────────────────────────────
        html.Div([
            html.Div([
                dcc.Dropdown(id="analise-dd", options=_TICK_OPTS, placeholder="Selecione um ticker B3...",
                    clearable=True,
                    style={"backgroundColor":_B["card"],"color":_B["text"],
                           "fontFamily":_B["mono"],"fontSize":"0.85rem","border":"none"}),
            ], style={"flex":"3","minWidth":"0"}),
            html.Div([
                dcc.Input(id="analise-cnpj", type="text",
                    placeholder="ou CNPJ / nome livre...",
                    debounce=False,
                    style={"width":"100%","backgroundColor":_B["card"],"border":f"1px solid {_B['border']}",
                           "color":_B["text"],"fontFamily":_B["mono"],"fontSize":"0.82rem",
                           "padding":"8px 12px","borderRadius":"3px"}),
            ], style={"flex":"2","minWidth":"0"}),
            html.Div([
                html.Button("Calcular", id="analise-btn", n_clicks=0,
                    style={"width":"100%","padding":"8px 0","backgroundColor":_B["card"],
                           "border":f"1px solid {_B['orange']}","color":_B["orange"],
                           "fontFamily":_B["mono"],"fontSize":"0.82rem",
                           "fontWeight":"700","cursor":"pointer","borderRadius":"3px"}),
            ], style={"flex":"1","minWidth":"80px"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"14px"}),
        # ── Loading + resultado ──────────────────────────────────────────────
        dcc.Loading(type="circle", color=_B["orange"],
            children=html.Div(id="analise-result")),
        # ── Gemini section ───────────────────────────────────────────────────
        html.Div(id="analise-ai-section"),
    ])


def _render_result_layout(ticker, sector, sr: SectorRiskResult, year_t) -> html.Div:
    """Renderiza o painel de resultado para qualquer tipo de scorer."""
    stype = sr.scorer_type
    ms    = sr.mscore_result
    cfq   = sr.cfq_result
    alert = sr.alert_level.value
    flags = sr.red_flags or []

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    if stype == "beneish" and ms and cfq:
        grade, g_label = compute_grade(ms.m_score, cfq.accrual_ratio)
        header = html.Div([
            html.Div(_grade_badge(grade), style={"marginRight":"16px"}),
            html.Div([
                html.Div([_mono(f"M-Score: ", _B["muted"],"0.8rem"),
                          _mono(f"{ms.m_score:+.4f}",
                                _B["red"] if ms.m_score > -1.78 else _B["green"],
                                "1.1rem", bold=True)]),
                html.Div([_mono("Limiar: −1.78 → ", _B["muted"], "0.75rem"),
                          _mono(ms.classification, _B["orange"], "0.75rem", bold=True)]),
                html.Div([_mono("Accrual Ratio: ", _B["muted"], "0.75rem"),
                          _mono(f"{cfq.accrual_ratio:+.4f}", _B["text"], "0.75rem")],
                         style={"marginTop":"3px"}),
            ], style={"flex":"1"}),
            html.Div([_pill(alert)], style={"marginLeft":"auto","alignSelf":"center"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"12px"})
    else:
        risk_color = (_B["red"] if sr.risk_score >= 5 else
                      _B["orange"] if sr.risk_score >= 3 else _B["green"])
        grade, g_label = compute_grade_financial(sr.risk_score, alert)
        header = html.Div([
            html.Div([
                _mono(f"{sr.risk_score:.1f}", risk_color, "2rem", bold=True),
                _mono("/10", _B["muted"], "0.9rem"),
            ], style={"marginRight":"16px"}),
            html.Div([
                html.Div(_mono(sr.classification, _B["orange"], "0.88rem", bold=True)),
                html.Div(_mono({"banking":"Modelo Bancário","insurance":"Índice Combinado SUSEP"}
                               .get(stype, stype), _B["muted"], "0.75rem")),
            ], style={"flex":"1"}),
            html.Div(_pill(alert), style={"marginLeft":"auto"}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"12px"})

    # ── Gráficos / métricas ─────────────────────────────────────────────────
    if stype == "beneish" and ms:
        charts = html.Div([
            html.Div(dcc.Graph(figure=_gauge(ms.m_score), config={"displayModeBar":False}),
                     style={"flex":"1"}),
            html.Div(dcc.Graph(figure=_radar(ms), config={"displayModeBar":False}),
                     style={"flex":"1"}),
        ], style={"display":"flex","gap":"12px"})
    else:
        def _fmt(v):
            return "N/D" if (v != v) else f"{v:.4f}"
        m = sr.metrics or {}
        if stype == "banking":
            items = [("ROA",_fmt(m.get("roa",float("nan"))),"ok" if m.get("roa",0)>0.01 else "warn"),
                     ("Cost-to-Income",_fmt(m.get("cost_income",float("nan"))),"ok" if m.get("cost_income",1)<0.5 else "warn"),
                     ("CFO Quality",_fmt(m.get("cfo_quality",float("nan"))),"ok" if m.get("cfo_quality",0)>0 else "crit"),
                     ("Crescimento Rev.",_fmt(m.get("rev_growth",float("nan"))),"ok"),
                     ("Spread",_fmt(m.get("spread",float("nan"))),"ok"),
                     ("Alavancagem",_fmt(m.get("leverage",float("nan"))),"ok" if m.get("leverage",0)<12 else "crit"),]
        else:
            items = [("Sinistralidade",_fmt(m.get("loss_ratio",float("nan"))),"ok" if m.get("loss_ratio",1)<0.7 else "crit"),
                     ("Índ. Despesas",_fmt(m.get("expense_ratio",float("nan"))),"ok" if m.get("expense_ratio",1)<0.3 else "warn"),
                     ("Índ. Combinado",_fmt(m.get("combined_ratio",float("nan"))),"ok" if m.get("combined_ratio",1)<1.0 else "crit"),
                     ("Crescimento Rev.",_fmt(m.get("rev_growth",float("nan"))),"ok"),
                     ("ROA",_fmt(m.get("roa",float("nan"))),"ok" if m.get("roa",0)>0.02 else "warn"),
                     ("CFO Quality",_fmt(m.get("cfo_quality",float("nan"))),"ok")]
        charts = html.Div(
            [html.Div(_metric_card(l, v, status=s), style={"flex":"1","minWidth":"140px"})
             for l,v,s in items],
            style={"display":"flex","flexWrap":"wrap","gap":"8px"})

    # ── Red flags ───────────────────────────────────────────────────────────
    if flags:
        flag_items = [html.Div(f, className="flag-item") for f in flags]
    else:
        flag_items = [html.Div("Nenhum red flag acima do limiar detectado.",
                               style={"color":_B["green"],"fontSize":"0.83rem",
                                      "fontFamily":_B["mono"],"padding":"8px 0"})]

    return html.Div([
        html.Div([
            html.Span(ticker, style={"color":_B["orange"],"fontWeight":"700",
                                     "fontFamily":_B["mono"],"fontSize":"1rem"}),
            html.Span(f" · {sector} · {year_t}",
                      style={"color":_B["muted"],"fontSize":"0.78rem","fontFamily":_B["mono"]}),
            html.Div(g_label, style={"fontSize":"0.72rem","color":_B["muted"],"marginTop":"2px"}),
        ], style={"background":_B["card"],"border":f"1px solid {_B['border']}",
                  "borderRadius":"6px","padding":"8px 16px","marginBottom":"10px"}),
        header,
        _divider(), charts, _divider(),
        _lbl("🚩 Red Flags Detectados"),
        html.Div(flag_items),
    ])


@callback(
    Output("analise-result",      "children"),
    Output("analise-ai-section",  "children"),
    Output("analyze-store",       "data"),
    Input("analise-btn",          "n_clicks"),
    State("analise-dd",           "value"),
    State("analise-cnpj",         "value"),
    State("year-dd",              "value"),
    State("analyze-store",        "data"),
    prevent_initial_call=True,
)
def _run_analysis(n, ticker_sel, cnpj, year_t, cache):
    if not n:
        raise PreventUpdate
    year_t = year_t or (_CY - 1)
    query  = (cnpj or "").strip() or (ticker_sel or "").strip()
    if not query:
        return (html.Div("Selecione um ticker ou informe um CNPJ.",
                         style={"color":_B["muted"],"fontFamily":_B["mono"]}),
                no_update, no_update)
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    from advisor_brain_fsa.sector_scorer import get_scorer
    try:
        fetcher = CVMDataFetcher()
        fd_t, fd_t1 = fetcher.get_financial_data(query, year_t=year_t, year_t1=year_t-1)
    except Exception as exc:
        return (html.Div([
                    html.Span("Erro ao buscar dados: ", style={"color":_B["red"],"fontWeight":"700"}),
                    html.Span(str(exc), style={"color":_B["text"]}),
                ], style={"fontFamily":_B["mono"],"fontSize":"0.83rem"}),
                no_update, no_update)
    sector = get_sector(query)
    sr     = get_scorer(sector).score(fd_t, fd_t1)

    # Persiste no Store para o botão Gemini não perder o estado
    new_cache = {"query": query, "sector": sector, "year_t": year_t,
                 "stype": sr.scorer_type,
                 "ms_score": sr.mscore_result.m_score if sr.mscore_result else None,
                 "accrual":  sr.cfq_result.accrual_ratio if sr.cfq_result else None,
                 "risk_score": sr.risk_score,
                 "classification": sr.classification,
                 "metrics": sr.metrics,
                 "flags": sr.red_flags,
                 "alert": sr.alert_level.value}

    result_panel = _render_result_layout(query, sector, sr, year_t)
    ai_section   = _build_ai_section(query, year_t)
    return result_panel, ai_section, new_cache


def _build_ai_section(ticker, year_t):
    return html.Div([
        _divider(),
        _lbl("🤖 Tese de Risco — Narrativa Gemini"),
        html.Button("Gerar Tese de Risco com Gemini", id="gemini-btn", n_clicks=0,
            style={"padding":"9px 22px","backgroundColor":_B["card"],
                   "border":f"1px solid {_B['orange']}","color":_B["orange"],
                   "fontFamily":_B["mono"],"fontSize":"0.82rem","fontWeight":"700",
                   "cursor":"pointer","borderRadius":"3px","marginBottom":"12px"}),
        dcc.Loading(type="circle", color=_B["orange"],
            children=html.Div(id="gemini-output")),
    ])


@callback(
    Output("gemini-output", "children"),
    Input("gemini-btn",     "n_clicks"),
    State("analyze-store",  "data"),
    State("year-dd",        "value"),
    prevent_initial_call=True,
)
def _run_gemini(n, cache, year_t):
    if not n or not cache:
        raise PreventUpdate
    key = _api_key()
    if not key:
        return html.Div("Configure GOOGLE_API_KEY como variável de ambiente no Render.",
                        style={"color":_B["red"],"fontFamily":_B["mono"],"fontSize":"0.82rem"})
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    from advisor_brain_fsa.sector_scorer import get_scorer
    from advisor_brain_fsa.beneish_mscore import MScoreResult
    from advisor_brain_fsa.accruals import CashFlowQualityResult, AlertLevel

    year_t = cache.get("year_t") or year_t or (_CY - 1)
    query  = cache.get("query","")
    sector = cache.get("sector","")
    flags  = cache.get("flags") or []
    stype  = cache.get("stype","beneish")

    try:
        fetcher = CVMDataFetcher()
        fd_t, fd_t1 = fetcher.get_financial_data(query, year_t=year_t, year_t1=year_t-1)
        sr = get_scorer(sector).score(fd_t, fd_t1)
    except Exception as exc:
        return html.Div(f"Erro ao recarregar dados: {exc}",
                        style={"color":_B["red"],"fontFamily":_B["mono"]})

    try:
        analyst = GeminiAnalyst(api_key=key)
        report  = "".join(analyst.analyze_streaming(
            ticker=query, sector=sector, year=year_t,
            mscore_result=sr.mscore_result, cfq_result=sr.cfq_result,
            red_flags=sr.red_flags or [], sector_risk=sr,
        ))
    except Exception as exc:
        return html.Div(f"Erro na API Gemini: {exc}",
                        style={"color":_B["red"],"fontFamily":_B["mono"]})

    return html.Div([
        dcc.Markdown(report, className="dash-markdown",
                     style={"backgroundColor":_B["bg"],"color":_B["text"],
                            "fontFamily":_B["mono"],"fontSize":"0.83rem",
                            "padding":"16px 20px","border":f"1px solid {_B['border']}",
                            "borderLeft":f"3px solid {_B['orange']}","borderRadius":"4px",
                            "lineHeight":"1.7"}),
        html.A("⬇️ Baixar .md", href=f"data:text/markdown;charset=utf-8,{report}",
               download=f"tese_{query}_{year_t}.md",
               style={"display":"inline-block","marginTop":"10px","color":_B["orange"],
                      "fontFamily":_B["mono"],"fontSize":"0.78rem",
                      "textDecoration":"none","border":f"1px solid {_B['orange']}",
                      "padding":"5px 14px","borderRadius":"3px"}),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# TAB RANKING
# ─────────────────────────────────────────────────────────────────────────────

def _layout_ranking():
    return html.Div([
        html.Div([
            html.Span("Ranking de Mercado",
                      style={"fontSize":"1.15rem","fontWeight":"700","color":_B["text"]}),
            html.Div("Processa múltiplos tickers e ordena por nível de risco.",
                     style={"fontSize":"0.75rem","color":_B["muted"],"marginTop":"2px"}),
        ], style={"marginBottom":"14px"}),
        html.Div([
            html.Div([
                dcc.Dropdown(id="rank-tickers", options=_MULTI_OPTS, multi=True,
                    value=["PETR4","VALE3","ITUB4","ABEV3","ELET3","BBDC4","WEGE3","RENT3"],
                    placeholder="Selecione tickers...",
                    style={"backgroundColor":_B["card"],"color":_B["text"],
                           "fontFamily":_B["mono"],"fontSize":"0.82rem","border":"none"}),
            ], style={"flex":"3","minWidth":"0"}),
            html.Div([
                html.Button("Calcular Ranking", id="rank-btn", n_clicks=0,
                    style={"width":"100%","padding":"8px 14px","backgroundColor":_B["card"],
                           "border":f"1px solid {_B['orange']}","color":_B["orange"],
                           "fontFamily":_B["mono"],"fontSize":"0.82rem",
                           "fontWeight":"700","cursor":"pointer","borderRadius":"3px"}),
            ], style={"flex":"1","minWidth":"120px"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"14px"}),
        dcc.Loading(type="circle", color=_B["orange"],
            children=html.Div(id="rank-result")),
    ])


@callback(
    Output("rank-result",   "children"),
    Output("ranking-store", "data"),
    Input("rank-btn",       "n_clicks"),
    State("rank-tickers",   "value"),
    State("year-dd",        "value"),
    prevent_initial_call=True,
)
def _run_ranking(n, tickers, year_t):
    if not n or not tickers:
        raise PreventUpdate
    year_t = year_t or (_CY - 1)
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    from advisor_brain_fsa.sector_scorer import get_scorer as _gs

    fetcher = CVMDataFetcher()
    results = []
    for ticker in tickers:
        sector = get_sector(ticker)
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t-1)
            sr = _gs(sector).score(fd_t, fd_t1)
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, sector_risk=sr))
        except Exception as exc:
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, error=str(exc)))
        time.sleep(0.1)
    _apply_sector_stats(results)
    df  = _to_dataframe(results, top_flags=3)
    ok  = df[df["Nível de Alerta"] != "N/D"].copy()
    err = df[df["Nível de Alerta"] == "N/D"].copy()

    # Summary counts
    counts = ok["Nível de Alerta"].value_counts() if not ok.empty else {}
    summary = html.Div([
        html.Div([
            html.Div([
                html.Div(str(counts.get("Crítico",0)),
                         style={"fontSize":"1.6rem","fontWeight":"700","color":_B["red"],"fontFamily":_B["mono"]}),
                html.Div("CRÍTICO", style={"fontSize":"0.6rem","color":_B["muted"],"fontFamily":_B["mono"]}),
            ], style={"flex":"1","textAlign":"center","padding":"12px",
                      "background":_B["card"],"border":f"1px solid #2a0000","borderRadius":"4px"}),
            html.Div([
                html.Div(str(counts.get("Alto Risco",0)),
                         style={"fontSize":"1.6rem","fontWeight":"700","color":_B["orange"],"fontFamily":_B["mono"]}),
                html.Div("ALTO RISCO", style={"fontSize":"0.6rem","color":_B["muted"],"fontFamily":_B["mono"]}),
            ], style={"flex":"1","textAlign":"center","padding":"12px",
                      "background":_B["card"],"border":f"1px solid #2a1a00","borderRadius":"4px"}),
            html.Div([
                html.Div(str(counts.get("Atenção",0)),
                         style={"fontSize":"1.6rem","fontWeight":"700","color":_B["yellow"],"fontFamily":_B["mono"]}),
                html.Div("ATENÇÃO", style={"fontSize":"0.6rem","color":_B["muted"],"fontFamily":_B["mono"]}),
            ], style={"flex":"1","textAlign":"center","padding":"12px",
                      "background":_B["card"],"border":f"1px solid #2a2000","borderRadius":"4px"}),
            html.Div([
                html.Div(str(counts.get("Normal",0)),
                         style={"fontSize":"1.6rem","fontWeight":"700","color":_B["green"],"fontFamily":_B["mono"]}),
                html.Div("NORMAL", style={"fontSize":"0.6rem","color":_B["muted"],"fontFamily":_B["mono"]}),
            ], style={"flex":"1","textAlign":"center","padding":"12px",
                      "background":_B["card"],"border":f"1px solid #002200","borderRadius":"4px"}),
        ], style={"display":"flex","gap":"10px","marginBottom":"14px"}),
    ])

    # Bloomberg DataTable
    if not ok.empty:
        want = ["Ticker","Setor","Scorer","Score de Risco","M-Score",
                "Nível de Alerta","Accrual Ratio","Δ vs Grupo","Red Flag 1"]
        cols_use = [c for c in want if c in ok.columns]

        def _safe(v):
            import math
            if v is None: return None
            try:
                if math.isnan(float(v)): return None
            except Exception: pass
            return v

        data = [{c: _safe(row.get(c)) for c in cols_use} for _, row in ok[cols_use].iterrows()]
        fmt_cols = []
        for c in cols_use:
            if c in ("Score de Risco","M-Score","Accrual Ratio","Δ vs Grupo"):
                fmt_cols.append({"id":c,"name":c,"type":"numeric","format":{"specifier":"+.4f"}})
            else:
                fmt_cols.append({"id":c,"name":c})

        table = dash_table.DataTable(
            id="rank-table",
            data=data,
            columns=fmt_cols,
            style_table=_DT_TABLE,
            style_header=_DT_HEADER,
            style_cell=_DT_CELL,
            style_data_conditional=_DT_COND,
            sort_action="native",
            filter_action="native",
            page_size=25,
            export_format="csv",
        )
        chart = []
        b_df  = ok[ok["Scorer"]=="beneish"]
        if len(b_df) > 1:
            chart = [_divider(),
                     dcc.Graph(figure=_sector_bar(b_df), config={"displayModeBar":False})]
    else:
        table = html.Div("Nenhum resultado disponível.",
                         style={"color":_B["muted"],"fontFamily":_B["mono"]})
        chart = []

    err_block = []
    if not err.empty:
        err_block = [_divider(),
                     html.Details([
                         html.Summary(f"⚠️ {len(err)} ticker(s) sem dados",
                                      style={"color":_B["yellow"],"fontFamily":_B["mono"],
                                             "cursor":"pointer","fontSize":"0.82rem"}),
                         html.Div([html.Div(f"{r.Ticker} — {r.get('Erro','erro desconhecido')}",
                                            style={"color":_B["muted"],"fontSize":"0.78rem",
                                                   "fontFamily":_B["mono"],"padding":"3px 0"})
                                   for _, r in err.iterrows()],
                                  style={"padding":"8px 12px"}),
                     ])]

    return html.Div([summary, table, *chart, *err_block]), df.to_json(orient="records")



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
