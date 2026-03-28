"""
Advisor-Brain-FSA — Streamlit App v2
=====================================
Interface web para análise de qualidade de relatórios financeiros.

Rodar:
    streamlit run app.py

API Key (Gemini):
    Crie .streamlit/secrets.toml com:
        GOOGLE_API_KEY = "AIzaSy..."
    Ou exporte a variável de ambiente:
        export GOOGLE_API_KEY="AIzaSy..."
"""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advisor_brain_fsa import BeneishMScore, CashFlowQuality
from advisor_brain_fsa.beneish_mscore import FinancialData
from advisor_brain_fsa.mda_analyst import GeminiAnalyst, compute_grade
from advisor_brain_fsa.rank_market import (
    DEFAULT_WATCHLIST,
    CompanyResult,
    _apply_sector_stats,
    _to_dataframe,
    detect_red_flags,
)
from advisor_brain_fsa.sector_scorer import SectorRiskResult
from advisor_brain_fsa.ticker_map import (
    FINANCIAL_GROUP,
    SECTOR_LABELS,
    TICKER_TO_KEYWORD,
    get_sector,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Advisor-Brain-FSA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ══════════════════════════════════════════════════════════════════
       BLOOMBERG TERMINAL PALETTE
       BG:#000000  CARD:#1C1C1C  BORDER:#30363D  TEXT:#D1D1D1
       GREEN:#00FF00  RED:#FF3E3E  ORANGE:#FFA500  YELLOW:#FFD700
    ══════════════════════════════════════════════════════════════════ */

    /* ── App-wide background override ───────────────────────────────── */
    .stApp, .stApp > div,
    .main, .main .block-container {
        background-color: #000000 !important;
        color: #D1D1D1 !important;
    }

    /* Force Streamlit tab background */
    [data-testid="stTabs"] { background-color: #000000 !important; }
    [data-testid="stTabContent"] { background-color: #000000 !important; }

    /* Global font stack — monospace for the terminal feel */
    html, body, .stApp {
        font-family: 'JetBrains Mono','Roboto Mono','Fira Code','Consolas',monospace !important;
    }

    /* Streamlit default text elements */
    p, li, span, label, .stMarkdown {
        color: #D1D1D1 !important;
    }
    h1, h2, h3, h4 { color: #D1D1D1 !important; }

    /* Dividers */
    hr { border-color: #30363D !important; }

    /* Streamlit info/success/warning boxes → terminal style */
    [data-testid="stAlert"] {
        background: #1C1C1C !important;
        border: 1px solid #30363D !important;
        color: #D1D1D1 !important;
    }

    /* ── Grade badge ─────────────────────────────────────────────────── */
    .grade-badge {
        display:inline-block; font-size:3rem; font-weight:900;
        width:90px; height:90px; line-height:90px;
        text-align:center; border-radius:4px; color:#000000;
        font-family:'JetBrains Mono',monospace;
    }
    .grade-A { background:#00FF00; }
    .grade-B { background:#7FFF00; }
    .grade-C { background:#FFD700; }
    .grade-D { background:#FFA500; }
    .grade-F { background:#FF3E3E; }

    /* ── Alert pills — Bloomberg palette ────────────────────────────── */
    .pill {
        display:inline-block; padding:3px 12px; border-radius:3px;
        font-weight:700; font-size:.82rem; letter-spacing:.05em;
        font-family:'JetBrains Mono',monospace;
    }
    .pill-critico { background:#FF3E3E22; border:1px solid #FF3E3E; color:#FF3E3E; }
    .pill-alto    { background:#FFA50022; border:1px solid #FFA500; color:#FFA500; }
    .pill-atencao { background:#FFD70022; border:1px solid #FFD700; color:#FFD700; }
    .pill-normal  { background:#00FF0022; border:1px solid #00FF00; color:#00FF00; }

    /* ── Red flag rows ───────────────────────────────────────────────── */
    .flag-item {
        background: #1C1C1C;
        border-left: 3px solid #FF3E3E;
        padding: 8px 14px; margin: 5px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: .88rem; color: #FF3E3E;
    }

    /* ── Index explanation cards ─────────────────────────────────────── */
    .idx-explain {
        background: #1C1C1C; border: 1px solid #30363D;
        border-radius: 4px; padding: 12px 16px; margin: 5px 0;
        color: #D1D1D1; font-family: 'JetBrains Mono', monospace;
        font-size: .85rem;
    }

    /* ── Bloomberg metric cards ──────────────────────────────────────── */
    .metric-card {
        background: #1C1C1C;
        border: 1px solid #30363D;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.68rem; color: #888888;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-value {
        font-family: 'JetBrains Mono','Roboto Mono','Consolas', monospace;
        font-size: 1.3rem; font-weight: 700;
        color: #D1D1D1; letter-spacing: 0;
    }
    .metric-value.ok   { color: #00FF00; }
    .metric-value.warn { color: #FFA500; }
    .metric-value.crit { color: #FF3E3E; }
    .metric-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem; color: #888888; margin-top: 3px;
    }

    /* ── Home scorer column header ───────────────────────────────────── */
    .scorer-header {
        background: #1C1C1C; border: 1px solid #30363D;
        border-radius: 4px 4px 0 0; padding: 10px 14px;
    }
    .scorer-title {
        font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.12em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Ticker rows in top-5 ────────────────────────────────────────── */
    .ticker-row {
        background: #1C1C1C; border: 1px solid #30363D; border-top: none;
        padding: 8px 14px; font-family: 'JetBrains Mono', monospace;
        font-size: 0.86rem; color: #D1D1D1;
    }
    .ticker-row:hover { background: #252525; }

    /* ── Sidebar — completamente oculta ─────────────────────────────── */
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"] {
        display: none !important;
    }

    /* ── Streamlit tab bar ───────────────────────────────────────────── */
    [data-testid="stTabs"] button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .78rem !important; letter-spacing: .06em !important;
        color: #888888 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFA500 !important;
        border-bottom: 2px solid #FFA500 !important;
    }

    /* ── st.metric widget ────────────────────────────────────────────── */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono','Roboto Mono',monospace !important;
        color: #D1D1D1 !important;
    }
    [data-testid="stMetricDelta"] { font-family: 'JetBrains Mono',monospace !important; }

    /* ── Dataframes / tables ─────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #30363D !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stDataFrameGlideDataEditor"] * {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────────── */
    .stButton > button {
        background: #1C1C1C !important; border: 1px solid #30363D !important;
        color: #D1D1D1 !important; border-radius: 3px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .8rem !important; letter-spacing: .04em !important;
    }
    .stButton > button:hover {
        border-color: #FFA500 !important; color: #FFA500 !important;
    }
    .stButton > button[kind="primary"] {
        border-color: #FFA500 !important; color: #FFA500 !important;
    }

    /* ── Selectbox / input ───────────────────────────────────────────── */
    [data-testid="stSelectbox"] > div,
    [data-testid="stTextInput"] > div > div {
        background: #1C1C1C !important; border-color: #30363D !important;
        color: #D1D1D1 !important; font-family: 'JetBrains Mono',monospace !important;
    }

    /* ── Spinner ─────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] { color: #FFA500 !important; }

    /* ── Fixed legal footer ──────────────────────────────────────────── */
    .legal-footer {
        position: fixed; bottom: 0; left: 0; right: 0;
        background: rgba(0,0,0,0.97);
        border-top: 1px solid #30363D;
        padding: 6px 24px;
        font-size: 0.65rem; color: #888888;
        font-family: 'JetBrains Mono', monospace;
        z-index: 9999; backdrop-filter: blur(4px); line-height: 1.6;
    }
    .legal-footer b { color: #FFA500; }

    /* Compensate for fixed footer + remove default sidebar padding */
    .main .block-container {
        padding-bottom: 72px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1400px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# API key — lida do backend (secrets ou env), nunca do usuário
# ─────────────────────────────────────────────────────────────────────────────

def _get_api_key() -> str:
    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Barra superior — sem sidebar
# ─────────────────────────────────────────────────────────────────────────────

current_year = date.today().year

_tb_brand, _tb_year, _tb_ai, _tb_ver = st.columns([5, 2, 2, 1])
with _tb_brand:
    st.markdown(
        '<span style="font-family:\'JetBrains Mono\',monospace;font-size:1rem;'
        'font-weight:700;color:#FFA500;letter-spacing:.1em">ADVISOR-BRAIN-FSA</span>'
        '<span style="color:#888888;font-size:.72rem;margin-left:12px">'
        'Qualidade de Relatórios Financeiros · B3</span>',
        unsafe_allow_html=True,
    )
with _tb_year:
    _prev_year = st.session_state.get("_last_year_t")
    year_t = st.selectbox(
        "Ano de Balanço",
        options=list(range(current_year - 1, current_year - 6, -1)),
        index=0,
        key="global_year_t",
        label_visibility="collapsed",
        help="Ano de referência do balanço (DFP CVM)",
    )
    if _prev_year is not None and _prev_year != year_t:
        # Ano mudou — limpa cache de análise para forçar recálculo
        st.session_state.pop("_analyze_cache", None)
        st.cache_data.clear()
    st.session_state["_last_year_t"] = year_t
with _tb_ai:
    _key_status = _get_api_key()
    if _key_status:
        st.markdown(
            '<span style="font-size:.72rem;color:#00FF00;font-family:\'JetBrains Mono\',monospace">'
            '■ GEMINI OK</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span style="font-size:.72rem;color:#FF3E3E;font-family:\'JetBrains Mono\',monospace">'
            '■ SEM API KEY</span>',
            unsafe_allow_html=True,
        )
with _tb_ver:
    st.markdown(
        '<span style="font-size:.65rem;color:#444444;font-family:\'JetBrains Mono\',monospace">'
        'v0.7.0</span>',
        unsafe_allow_html=True,
    )

st.markdown('<hr style="margin:4px 0 10px 0;border-color:#30363D">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de UI
# ─────────────────────────────────────────────────────────────────────────────

_ALERT_PILL = {
    "Crítico":    ("pill-critico",  "🔴 Crítico"),
    "Alto Risco": ("pill-alto",     "🟠 Alto Risco"),
    "Atenção":    ("pill-atencao",  "🟡 Atenção"),
    "Normal":     ("pill-normal",   "🟢 Normal"),
}

_THRESHOLDS = {
    "DSRI": 1.031, "GMI": 1.014, "AQI": 1.039, "SGI": 1.134,
    "DEPI": 1.017, "SGAI": 1.054, "LVGI": 1.000, "TATA": -0.012,
}

# ─────────────────────────────────────────────────────────────────────────────
# Explicações detalhadas dos 8 índices
# ─────────────────────────────────────────────────────────────────────────────

_INDEX_INFO = {
    "DSRI": {
        "full":     "Days Sales in Receivables Index",
        "formula":  "(Recebíveis_t / Receita_t) ÷ (Recebíveis_t₋₁ / Receita_t₋₁)",
        "what":     "Mede a variação relativa no prazo médio de recebimento entre dois exercícios. "
                    "Valor > 1 indica que contas a receber cresceram mais rápido que a receita.",
        "suggests": "DSRI > 1,031 pode indicar reconhecimento prematuro de receitas (vendas 'na "
                    "virada do período'), concessão de prazos maiores para inflar vendas, ou "
                    "dificuldades de cobrança. É o sinal mais clássico de manipulação de receitas.",
    },
    "GMI": {
        "full":     "Gross Margin Index",
        "formula":  "[(Receita_t₋₁ − CPV_t₋₁) / Receita_t₋₁] ÷ [(Receita_t − CPV_t) / Receita_t]",
        "what":     "Compara a margem bruta do período anterior com a do corrente. "
                    "Valor > 1 indica deterioração da rentabilidade bruta.",
        "suggests": "Deterioração de margem (GMI > 1,014) cria incentivo para manipulação: a empresa "
                    "está sob pressão competitiva e pode subreportar o CPV ou superestimar receitas "
                    "para manter aparência de rentabilidade.",
    },
    "AQI": {
        "full":     "Asset Quality Index",
        "formula":  "[1 − (Ativo Circ._t + AF_t) / AT_t] ÷ [1 − (Ativo Circ._t₋₁ + AF_t₋₁) / AT_t₋₁]",
        "what":     "Mede a variação na proporção de ativos 'não produtivos' (excluindo circulante e "
                    "ativo fixo tangível). Acima de 1 indica crescimento de ativos intangíveis/diferidos.",
        "suggests": "AQI > 1,039 sugere capitalização agressiva de gastos que deveriam ser despesas — "
                    "P&D, software interno, custos de aquisição de clientes. Prática que infla ativos "
                    "e eleva o lucro contábil artificialmente.",
    },
    "SGI": {
        "full":     "Sales Growth Index",
        "formula":  "Receita_t ÷ Receita_t₋₁",
        "what":     "Razão simples de crescimento de receita. Por si só não indica fraude, mas "
                    "empresas de alto crescimento têm maior incentivo e oportunidade para manipular.",
        "suggests": "SGI > 1,134 (acima da média histórica Beneish 1999) em conjunto com outros "
                    "índices elevados aumenta o risco. Crescimento agressivo pode mascarar antecipação "
                    "de receitas ou reconhecimento de contratos incompletos.",
    },
    "DEPI": {
        "full":     "Depreciation Index",
        "formula":  "[Dep_t₋₁ / (Dep_t₋₁ + AF bruto_t₋₁)] ÷ [Dep_t / (Dep_t + AF bruto_t)]",
        "what":     "Compara a taxa de depreciação em relação ao ativo fixo bruto entre períodos. "
                    "Valor > 1 indica que a empresa passou a depreciar mais lentamente.",
        "suggests": "DEPI > 1,017 sugere alongamento da vida útil estimada dos ativos — reduz a "
                    "despesa de depreciação e infla o lucro sem impacto no caixa. Típico de empresas "
                    "que 'revisaram premissas' de vida útil em momentos de pressão sobre resultados.",
    },
    "SGAI": {
        "full":     "SG&A Expenses Index",
        "formula":  "(Desp. SGA_t / Receita_t) ÷ (Desp. SGA_t₋₁ / Receita_t₋₁)",
        "what":     "Mede a variação das despesas gerais e administrativas em proporção à receita. "
                    "Acima de 1 indica crescimento desproporcional de overhead.",
        "suggests": "Coeficiente negativo no modelo (−0,172): empresas manipuladoras tendem a ter "
                    "SGA relativamente menor, pois inflam receita sem correspondente aumento de overhead. "
                    "SGAI > 1,054 sinaliza ineficiência operacional.",
    },
    "LVGI": {
        "full":     "Leverage Index",
        "formula":  "[(Dívida LP_t + PC_t) / AT_t] ÷ [(Dívida LP_t₋₁ + PC_t₋₁) / AT_t₋₁]",
        "what":     "Mede a variação do nível de endividamento total em relação ao ativo. "
                    "Acima de 1 indica aumento da alavancagem financeira.",
        "suggests": "Alta alavancagem cria pressão para cumprir covenants financeiros (cobertura de "
                    "juros, D/EBITDA), motivando gerenciamento de resultados. Coeficiente negativo "
                    "(−0,327) — empresas mais alavancadas reduzem levemente o M-Score calculado.",
    },
    "TATA": {
        "full":     "Total Accruals to Total Assets",
        "formula":  "(Lucro Líquido_t − Fluxo de Caixa Operacional_t) ÷ Ativo Total_t",
        "what":     "Mede o componente não-caixa (accruals) do lucro em relação ao ativo total. "
                    "Equivale ao Accrual Ratio do CFA Level 2 (Sloan, 1996).",
        "suggests": "Maior coeficiente do modelo (4,679). TATA elevado indica lucro 'accrual-driven' "
                    "— menos persistente e com alta probabilidade de reversão futura. Lucros de alta "
                    "qualidade são 'cash-driven': caixa gerado ≈ lucro reportado.",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ─────────────────────────────────────────────────────────────────────────────

def _pill_html(alert: str) -> str:
    css, label = _ALERT_PILL.get(alert, ("", alert))
    return f'<span class="pill {css}">{label}</span>'

def _grade_html(grade: str) -> str:
    return f'<div class="grade-badge grade-{grade}">{grade}</div>'

def _fmt_mscore(val: float) -> str:
    color = "#FF3E3E" if val > -1.78 else "#00FF00"
    return (
        f'<span style="color:{color};font-size:2rem;font-weight:900;'
        f'font-family:\'JetBrains Mono\',monospace">{val:+.4f}</span>'
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gráficos Plotly
# ─────────────────────────────────────────────────────────────────────────────

_BBG_LAYOUT = dict(
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(family="'JetBrains Mono','Roboto Mono',monospace", color="#D1D1D1"),
)

def _gauge_chart(m_score: float) -> go.Figure:
    is_bad = m_score > -1.78
    bar_color = "#FF3E3E" if is_bad else "#00FF00"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=m_score,
        delta={"reference": -1.78, "valueformat": ".3f",
               "increasing": {"color": "#FF3E3E"}, "decreasing": {"color": "#00FF00"}},
        number={"valueformat": "+.4f", "font": {"size": 26, "color": bar_color}},
        gauge={
            "axis": {"range": [-5, 2], "tickwidth": 1,
                     "tickcolor": "#888888", "tickfont": {"color": "#888888"}},
            "bgcolor": "#1C1C1C",
            "bar": {"color": bar_color, "thickness": 0.22},
            "bordercolor": "#30363D", "borderwidth": 1,
            "steps": [
                {"range": [-5, -1.78], "color": "#001a00"},
                {"range": [-1.78, 2],  "color": "#1a0000"},
            ],
            "threshold": {
                "line": {"color": "#FF3E3E", "width": 2},
                "thickness": 0.8, "value": -1.78,
            },
        },
        title={"text": "M-SCORE  |  LIMIAR −1.78",
               "font": {"size": 11, "color": "#888888"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10), **_BBG_LAYOUT)
    return fig

def _radar_chart(mscore) -> go.Figure:
    idx    = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA"]
    vals   = [mscore.dsri, mscore.gmi, mscore.aqi, mscore.sgi,
              mscore.depi, mscore.sgai, mscore.lvgi, mscore.tata]
    thresh = [_THRESHOLDS[i] for i in idx]
    colors = ["#FF3E3E" if v > t else "#00FF00" for v, t in zip(vals, thresh)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=thresh+[thresh[0]], theta=idx+[idx[0]], fill="toself",
        fillcolor="rgba(255,165,0,0.06)", line=dict(color="#FFA500", dash="dot", width=1),
        name="Limiar",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=idx+[idx[0]], fill="toself",
        fillcolor="rgba(255,62,62,0.08)", line=dict(color="#FF3E3E", width=1.5),
        marker=dict(color=colors, size=7), name="Empresa",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1C1C1C",
            radialaxis=dict(visible=True, showticklabels=False,
                            gridcolor="#30363D", linecolor="#30363D"),
            angularaxis=dict(gridcolor="#30363D", linecolor="#30363D",
                             tickfont=dict(color="#D1D1D1")),
        ),
        showlegend=True,
        legend=dict(font=dict(color="#888888", size=10)),
        height=340, margin=dict(l=30, r=30, t=40, b=20),
        **_BBG_LAYOUT,
    )
    return fig

def _sector_bar(df: pd.DataFrame) -> go.Figure:
    ok = df[df["M-Score"].notna()]
    if ok.empty:
        return go.Figure()
    avg    = ok.groupby("Setor")["M-Score"].mean().sort_values()
    colors = ["#FF3E3E" if v > -1.78 else "#00FF00" for v in avg.values]
    fig = go.Figure(go.Bar(
        x=avg.values, y=avg.index, orientation="h",
        marker_color=colors, marker_line_width=0,
        text=[f"{v:+.3f}" for v in avg.values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color="#D1D1D1"),
    ))
    fig.add_vline(x=-1.78, line_dash="dot", line_color="#FF3E3E", line_width=1,
                  annotation_text="−1.78", annotation_font_color="#FF3E3E",
                  annotation_position="top")
    fig.update_layout(
        height=max(220, len(avg) * 36),
        margin=dict(l=10, r=70, t=16, b=24),
        xaxis=dict(gridcolor="#30363D", zerolinecolor="#30363D",
                   tickfont=dict(color="#888888")),
        yaxis=dict(gridcolor="#30363D", tickfont=dict(color="#D1D1D1")),
        xaxis_title="M-Score Médio",
        **_BBG_LAYOUT,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Dados sintéticos (demo sem rede)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_DATA = {
    "SafeCo (Saudável)": (
        FinancialData(1_200_000, 720_000, 120_000, 100_000, 900_000, 300_000,
                      400_000, 50_000, 180_000, 90_000, 40_000, 120_000, 150_000),
        FinancialData(1_000_000, 600_000, 100_000,  80_000, 800_000, 260_000,
                      360_000, 40_000, 160_000, 80_000, 36_000, 100_000, 130_000),
    ),
    "RiskyInc (Manipuladora)": (
        FinancialData(1_500_000, 1_200_000, 300_000, 400_000, 1_000_000, 200_000,
                      200_000, 20_000, 500_000, 200_000, 10_000, 200_000, 20_000),
        FinancialData(1_000_000,   700_000, 150_000, 100_000,   800_000, 250_000,
                      280_000, 30_000, 200_000, 100_000, 30_000,  80_000, 100_000),
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Decomposição detalhada dos índices (com explicações)
# ─────────────────────────────────────────────────────────────────────────────

def _render_index_detail(mscore) -> None:
    """8 expanders — um por índice Beneish — com fórmula, explicação e valor."""
    st.markdown("### 📐 Decomposição dos Índices Beneish")
    st.caption("Clique em cada índice para ver o que ele significa, como é calculado e o que sugere.")

    index_values = [
        ("DSRI", mscore.dsri),
        ("GMI",  mscore.gmi),
        ("AQI",  mscore.aqi),
        ("SGI",  mscore.sgi),
        ("DEPI", mscore.depi),
        ("SGAI", mscore.sgai),
        ("LVGI", mscore.lvgi),
        ("TATA", mscore.tata),
    ]

    for name, value in index_values:
        info      = _INDEX_INFO[name]
        threshold = _THRESHOLDS[name]
        above     = value > threshold
        icon      = "⚠️" if above else "✅"
        status    = "Acima do limiar — sinal de alerta" if above else "Dentro do esperado"

        with st.expander(
            f"{icon} **{name}** ({info['full']}) — valor: `{value:.4f}`  |  limiar: `{threshold}`",
            expanded=above,
        ):
            c1, c2 = st.columns([1, 2])
            with c1:
                color = "#FF3E3E" if above else "#00FF00"
                st.markdown(
                    f"<div style='font-size:2.2rem;font-weight:900;color:{color}'>{value:+.4f}</div>"
                    f"<div style='font-size:.8rem;color:#888888'>Limiar Beneish 1999: {threshold}</div>"
                    f"<div style='margin-top:6px'><b>Status:</b> {icon} {status}</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(f"**📌 O que é?**\n\n{info['what']}")
            st.markdown(f"**🔢 Cálculo:** `{info['formula']}`")
            st.markdown(f"**💡 O que sugere?** {info['suggests']}")


# ─────────────────────────────────────────────────────────────────────────────
# Bloco de resultado: shared entre abas
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(ticker: str, sector: str, sector_risk: SectorRiskResult) -> None:
    """
    Renderiza o painel de resultado de uma empresa.
    Adapta automaticamente ao tipo de scorer: beneish | banking | insurance.
    """
    alert  = sector_risk.alert_level.value
    flags  = sector_risk.red_flags
    ms     = sector_risk.mscore_result
    cfq    = sector_risk.cfq_result
    stype  = sector_risk.scorer_type

    # ── Cabeçalho ──────────────────────────────────────────────────────────
    if stype == "beneish" and ms and cfq:
        grade, grade_label = compute_grade(ms.m_score, cfq.accrual_ratio)
        col_grade, col_score, col_alert = st.columns([1, 2, 2])
        with col_grade:
            st.markdown(_grade_html(grade), unsafe_allow_html=True)
            st.caption(grade_label)
        with col_score:
            st.markdown("**M-Score**")
            st.markdown(_fmt_mscore(ms.m_score), unsafe_allow_html=True)
            st.caption(f"Limiar: −1.78 → **{ms.classification}**")
        with col_alert:
            st.markdown("**Nível de Alerta**")
            st.markdown(_pill_html(alert), unsafe_allow_html=True)
            st.caption(f"Accrual Ratio: {cfq.accrual_ratio:+.4f} | Qualidade: {cfq.earnings_quality}")
    else:
        # Banking / Insurance header
        risk_color = "#FF3E3E" if sector_risk.risk_score >= 5 else (
                     "#FFA500" if sector_risk.risk_score >= 3 else "#00FF00")
        col_score, col_alert, col_class = st.columns([1, 1, 2])
        with col_score:
            st.markdown("**Score de Risco**")
            st.markdown(
                f'<span style="color:{risk_color};font-size:2rem;font-weight:900">'
                f'{sector_risk.risk_score:.1f}<span style="font-size:1rem">/10</span></span>',
                unsafe_allow_html=True,
            )
            scorer_label = {"banking": "Modelo Bancário", "insurance": "Índice Combinado"}
            st.caption(scorer_label.get(stype, stype))
        with col_alert:
            st.markdown("**Nível de Alerta**")
            st.markdown(_pill_html(alert), unsafe_allow_html=True)
        with col_class:
            st.markdown("**Classificação**")
            st.info(sector_risk.classification)

    st.divider()

    # ── Visualizações ───────────────────────────────────────────────────────
    if stype == "beneish" and ms:
        col_gauge, col_radar = st.columns(2)
        with col_gauge:
            st.plotly_chart(_gauge_chart(ms.m_score), use_container_width=True)
        with col_radar:
            st.plotly_chart(_radar_chart(ms), use_container_width=True)
        st.divider()
        _render_index_detail(ms)
    else:
        # Sector-specific metric cards
        _render_sector_metrics(stype, sector_risk.metrics)

    st.divider()

    # ── Red Flags ───────────────────────────────────────────────────────────
    st.markdown("#### 🚩 Red Flags Detectados")
    if flags:
        for f in flags:
            st.markdown(f'<div class="flag-item">{f}</div>', unsafe_allow_html=True)
    else:
        st.success("Nenhum red flag acima do limiar detectado.")

    # ── Tese de Risco (IA — todos os scorers) ──────────────────────────────
    st.divider()
    st.markdown("#### 🤖 Tese de Risco — Narrativa Gemini")

    api_key = _get_api_key()
    if not api_key:
        st.info("Configure **GOOGLE_API_KEY** em `.streamlit/secrets.toml` para gerar a narrativa.", icon="🔑")
        return

    if st.button("Gerar Tese de Risco com Gemini", type="primary", key=f"ai_{ticker}"):
        analyst = GeminiAnalyst(api_key=api_key)
        with st.spinner("Gemini está analisando…"):
            placeholder = st.empty()
            full_text: list[str] = []
            try:
                for chunk in analyst.analyze_streaming(
                    ticker=ticker, sector=sector, year=year_t,
                    mscore_result=ms, cfq_result=cfq, red_flags=flags,
                    sector_risk=sector_risk,
                ):
                    full_text.append(chunk)
                    placeholder.markdown("".join(full_text))
            except Exception as exc:
                st.error(f"Erro na API Gemini: {exc}")
                return

        st.download_button(
            "⬇️ Baixar relatório .md",
            data="".join(full_text),
            file_name=f"tese_risco_{ticker}_{year_t}.md",
            mime="text/markdown",
        )


def _metric_card_html(label: str, value: str, delta: str = "", status: str = "ok") -> str:
    """
    Tarefa 4 — Dark equity-research metric card with monospaced value.
    status: 'ok' | 'warn' | 'crit'
    """
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {status}">{value}</div>'
        + (f'<div class="metric-delta">{delta}</div>' if delta else "")
        + "</div>"
    )


def _render_sector_metrics(scorer_type: str, metrics: dict) -> None:
    """Dark-card metric panels for banking and insurance scorers (Tarefa 4)."""

    def _fmt(v, fmt=".2%", fallback="—"):
        return (format(v, fmt) if not pd.isna(v) else fallback)

    def _status(v, ok_thresh, warn_thresh, invert=False):
        """Return 'ok'|'warn'|'crit' relative to thresholds."""
        if pd.isna(v):
            return "ok"
        good = v > ok_thresh if not invert else v < ok_thresh
        mid  = v > warn_thresh if not invert else v < warn_thresh
        if good:
            return "ok"
        if mid:
            return "warn"
        return "crit"

    if scorer_type == "banking":
        st.markdown("### 🏦 Indicadores Bancários — BACEN / Basileia III")
        roa = metrics.get("roa",         float("nan"))
        ci  = metrics.get("cost_income", float("nan"))
        cq  = metrics.get("cfo_quality", float("nan"))
        sp  = metrics.get("spread",      float("nan"))
        rg  = metrics.get("rev_growth",  float("nan"))

        cards = [
            _metric_card_html("ROA",
                _fmt(roa, ".2%"),
                "Limiar saudável: > 1.0%  |  Crítico: < 0.3%",
                _status(roa, 0.008, 0.003)),
            _metric_card_html("Cost / Income",
                _fmt(ci, ".1%"),
                "Eficiente: < 50%  |  Crítico: > 75%",
                _status(ci, 0.60, 0.75, invert=True)),
            _metric_card_html("CFO / Lucro Líquido",
                _fmt(cq, ".2f") + "×",
                "> 0.5× indica qualidade de caixa adequada",
                _status(cq, 0.50, 0.20)),
            _metric_card_html("Spread Financeiro",
                _fmt(sp, ".1%"),
                "Margem líquida de intermediação"),
            _metric_card_html("Crescimento Receita",
                _fmt(rg, ".1%"),
                "Crescimento YoY de receita de intermediação",
                "ok" if (not pd.isna(rg) and rg > 0) else "warn"),
        ]
        c1, c2, c3 = st.columns(3)
        c1.markdown(cards[0] + cards[3], unsafe_allow_html=True)
        c2.markdown(cards[1] + cards[4], unsafe_allow_html=True)
        c3.markdown(cards[2], unsafe_allow_html=True)

    elif scorer_type == "insurance":
        st.markdown("### 🛡️ Indicadores de Seguros — SUSEP / IFRS 17")
        lr  = metrics.get("loss_ratio",     float("nan"))
        er  = metrics.get("expense_ratio",  float("nan"))
        cr  = metrics.get("combined_ratio", float("nan"))
        rg  = metrics.get("rev_growth",     float("nan"))
        roa = metrics.get("roa",            float("nan"))

        cards = [
            _metric_card_html("Sinistralidade (Loss Ratio)",
                _fmt(lr, ".1%"),
                "Saudável: < 65%  |  Crítico: > 80%",
                _status(lr, 0.65, 0.80, invert=True)),
            _metric_card_html("Índice de Despesas",
                _fmt(er, ".1%"),
                "Proporção de despesas comerciais sobre prêmios retidos"),
            _metric_card_html("Índice Combinado",
                _fmt(cr, ".1%"),
                "< 100% = lucro técnico de subscrição",
                _status(cr, 0.95, 1.00, invert=True)),
            _metric_card_html("Crescimento de Prêmios",
                _fmt(rg, ".1%"),
                "Crescimento YoY de prêmios retidos",
                "ok" if (not pd.isna(rg) and rg > 0) else "warn"),
            _metric_card_html("ROA",
                _fmt(roa, ".2%"),
                "Retorno sobre ativos totais"),
        ]
        c1, c2, c3 = st.columns(3)
        c1.markdown(cards[0] + cards[3], unsafe_allow_html=True)
        c2.markdown(cards[1] + cards[4], unsafe_allow_html=True)
        c3.markdown(cards[2], unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tab 0 — Home: Dashboard Setorial (3 colunas por scorer)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=7200)
def _auto_market_df(year_t: int) -> pd.DataFrame:
    """
    Tarefa 2 — Carrega automaticamente o ranking assim que o app abre.
    Cacheado por 2h por st.cache_data; nenhum botão necessário.
    Não chama nenhuma função do Streamlit (sem st.progress / st.spinner aqui).
    """
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    from advisor_brain_fsa.sector_scorer import get_scorer
    import time as _t

    fetcher = CVMDataFetcher()
    results = []
    for ticker in DEFAULT_WATCHLIST:
        sector = get_sector(ticker)
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(
                ticker, year_t=year_t, year_t1=year_t - 1
            )
            sr = get_scorer(sector).score(fd_t, fd_t1)
            results.append(CompanyResult(
                ticker=ticker, sector=sector, year_t=year_t, sector_risk=sr,
            ))
        except Exception as exc:
            results.append(CompanyResult(
                ticker=ticker, sector=sector, year_t=year_t, error=str(exc),
            ))
        _t.sleep(0.1)

    _apply_sector_stats(results)
    return _to_dataframe(results, top_flags=1)


def _load_ranking(tickers, year_t):
    """Carrega resultados usando o scorer adequado por setor (Strategy Pattern)."""
    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    from advisor_brain_fsa.sector_scorer import get_scorer
    fetcher = CVMDataFetcher()
    results = []
    prog = st.progress(0, text="Carregando dados da CVM…")
    for i, ticker in enumerate(tickers):
        prog.progress((i + 1) / len(tickers), text=f"Processando {ticker}…")
        sector = get_sector(ticker)
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t - 1)
            scorer = get_scorer(sector)
            sr = scorer.score(fd_t, fd_t1)
            results.append(CompanyResult(
                ticker=ticker, sector=sector, year_t=year_t, sector_risk=sr,
            ))
        except Exception as exc:
            results.append(CompanyResult(
                ticker=ticker, sector=sector, year_t=year_t, error=str(exc),
            ))
        time.sleep(0.2)
    prog.empty()
    _apply_sector_stats(results)
    return results


def tab_home():
    # ── Tarefa 2: auto-load sem botão ──────────────────────────────────────
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:700;margin-bottom:2px;color:#e2e8f0">'
        '🏠 Dashboard Setorial B3</div>'
        '<div style="color:#888888;font-size:.85rem;margin-bottom:10px">'
        'Top 5 por risco · Industriais &nbsp;·&nbsp; Bancos &amp; Financeiro &nbsp;·&nbsp; Seguros'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Analisando watchlist B3…"):
        df = _auto_market_df(year_t)

    ok = df[df["Score de Risco"].notna()].copy()

    # Botão de refresh manual (não é de "carga" — apenas invalida cache)
    if st.button("↺ Atualizar dados", help="Invalida o cache e recarrega da CVM"):
        st.cache_data.clear()
        st.session_state.pop("home_selected", None)
        st.rerun()

    if ok.empty:
        st.warning("Nenhuma empresa com dados disponíveis.")
        return

    # ── Split por scorer_type e seleciona Top 5 por risco (decrescente) ─────
    _ICON_MAP = {"Crítico": "🔴", "Alto Risco": "🟠", "Atenção": "🟡", "Normal": "🟢"}

    def _top5(scorer_type: str) -> pd.DataFrame:
        sub = ok[ok["Scorer"] == scorer_type]
        return sub.nlargest(5, "Score de Risco") if not sub.empty else sub

    def _score_str(row) -> str:
        if row["Scorer"] == "beneish":
            ms = row.get("M-Score", float("nan"))
            return f"{ms:+.3f}" if pd.notna(ms) else "—"
        return f"{row['Score de Risco']:.1f}/10"

    def _scorer_column(col_container, scorer_type: str, title: str, icon: str,
                       border_color: str) -> None:
        """Painel Top-5 com cards #1C1C1C e botões interativos (Tarefas 2 + 3)."""
        top5 = _top5(scorer_type)
        with col_container:
            st.markdown(
                f'<div style="background:#1C1C1C;border:1px solid {border_color};'
                f'border-radius:8px 8px 0 0;padding:10px 14px;">'
                f'<span style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:{border_color}">{icon} {title}</span>'
                f'<span style="float:right;font-size:.7rem;color:#888888">Top 5 Risco</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if top5.empty:
                st.markdown(
                    '<div style="background:#000000;border:1px solid #30363D;border-top:none;'
                    'padding:12px 14px;color:#888888;font-size:.83rem;">Sem dados disponíveis</div>',
                    unsafe_allow_html=True,
                )
                return
            for _, row in top5.iterrows():
                alert  = row.get("Nível de Alerta", "—")
                a_icon = _ICON_MAP.get(alert, "⚪")
                score  = _score_str(row)
                setor  = row.get("Setor", "")
                ticker = row["Ticker"]
                btn_key = f"btn_{scorer_type}_{ticker}"
                is_sel  = st.session_state.get("home_selected") == ticker
                bg = "#2A1A00" if is_sel else "#000000"
                st.markdown(
                    f'<div style="background:{bg};border:1px solid #30363D;border-top:none;'
                    f'padding:1px 6px;">', unsafe_allow_html=True,
                )
                if st.button(
                    f"{a_icon}  **{ticker}** — `{score}`   ·  _{setor}_",
                    key=btn_key,
                    use_container_width=True,
                ):
                    if st.session_state.get("home_selected") == ticker:
                        st.session_state.pop("home_selected", None)
                    else:
                        st.session_state["home_selected"] = ticker
                        st.session_state["navigate_ticker"] = ticker
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    col_b, col_bk, col_i = st.columns(3)
    _scorer_column(col_b,  "beneish",   "Industriais (Beneish)", "🏭", "#3b82f6")
    _scorer_column(col_bk, "banking",   "Bancos & Financeiro",   "🏦", "#f59e0b")
    _scorer_column(col_i,  "insurance", "Seguros",               "🛡️", "#10b981")

    # ── Gráfico setorial M-Score ─────────────────────────────────────────────
    beneish_ok = ok[ok["Scorer"] == "beneish"]
    if len(beneish_ok) > 1:
        st.divider()
        st.markdown(
            '<span style="font-size:.8rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.08em;color:#888888">M-Score Médio por Setor (Industriais)</span>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_sector_bar(beneish_ok), use_container_width=True)

    # ── Análise Detalhada inline — re-fetch single ticker (disk-cached) ────
    selected = st.session_state.get("home_selected")
    if selected:
        row_df = ok[ok["Ticker"] == selected]
        sector = row_df["Setor"].iloc[0] if not row_df.empty else get_sector(selected)
        scorer_lbl = row_df["Scorer"].iloc[0] if not row_df.empty else "—"
        st.divider()
        st.markdown(
            f'<div style="background:#1C1C1C;border:1px solid #30363D;'
            f'border-radius:8px;padding:12px 18px;margin-bottom:12px;">'
            f'<span style="font-family:monospace;font-size:1.1rem;font-weight:700;'
            f'color:#FFA500">{selected}</span>'
            f'<span style="color:#888888;font-size:.85rem"> · {sector} · {year_t}</span>'
            f'<span style="float:right;font-size:.75rem;color:#888888">'
            f'scorer: {scorer_lbl}</span></div>',
            unsafe_allow_html=True,
        )
        st.info("Para narrativa IA completa, acesse a aba **🔍 Análise Individual**.", icon="💡")
        from advisor_brain_fsa.data_fetcher import CVMDataFetcher
        from advisor_brain_fsa.sector_scorer import get_scorer as _gs
        try:
            with st.spinner(f"Carregando {selected}…"):
                _fd_t, _fd_t1 = CVMDataFetcher().get_financial_data(
                    selected, year_t=year_t, year_t1=year_t - 1
                )
            _render_result(selected, sector, _gs(sector).score(_fd_t, _fd_t1))
        except Exception as _e:
            st.error(f"Erro ao carregar {selected}: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Análise Individual (CVM real)
# ─────────────────────────────────────────────────────────────────────────────

def tab_analyze():
    st.markdown(
        '<div style="font-size:1.2rem;font-weight:700;margin-bottom:2px;color:#D1D1D1">'
        '🔍 Análise Individual</div>'
        '<div style="color:#888888;font-size:.8rem;margin-bottom:10px">'
        'Dados reais do Portal CVM · M-Score + Accruals + Narrativa IA</div>',
        unsafe_allow_html=True,
    )

    # ── Pre-fill ticker navegando da Home ───────────────────────────────────
    _nav = st.session_state.pop("navigate_ticker", None)
    all_tickers = sorted(TICKER_TO_KEYWORD.keys())
    _default_idx = (all_tickers.index(_nav) + 1) if (_nav and _nav in all_tickers) else 0

    col_in, col_year_lbl, col_btn = st.columns([4, 1, 1])
    with col_in:
        ticker_sel = st.selectbox(
            "Ticker B3",
            options=[""] + all_tickers,
            index=_default_idx,
            format_func=lambda t: t if not t else f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
        )
        custom = st.text_input(
            "ou CNPJ / nome livre",
            placeholder="33.000.167/0001-01 ou Petrobras",
            label_visibility="collapsed",
        )
    with col_year_lbl:
        st.markdown(
            f'''<div style="margin-top:28px;font-size:.78rem;color:#888888;
            font-family:\'JetBrains Mono\',monospace">Ano: <b style="color:#FFA500">{year_t}</b></div>''',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Calcular", type="primary", use_container_width=True) or bool(_nav)

    query = custom.strip() if custom.strip() else ticker_sel

    # ── Verificar cache de análise anterior (persiste o resultado entre reruns) ─
    # Isso garante que clicar em "Gerar Tese de Risco" não reseta a busca.
    _cache = st.session_state.get("_analyze_cache")
    _cache_valid = (
        _cache is not None
        and _cache.get("query") == query
        and _cache.get("year_t") == year_t
        and not run
    )

    if not run and not _cache_valid:
        if query:
            st.markdown(
                '<div style="color:#888888;font-size:.83rem;padding:8px 0">' +
                f'Ticker selecionado: <b style="color:#FFA500">{query}</b> — clique em <b>Calcular</b> para analisar.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Selecione ou digite um ticker acima e clique em **Calcular**.", icon="🔍")
        return

    if _cache_valid:
        # Re-renderiza resultado em cache sem chamar CVM (persiste resultado para Gemini)
        sector = _cache["sector"]
        sr = _cache["sr"]
        st.markdown(
            f'<div style="background:#1C1C1C;border:1px solid #30363D;border-radius:6px;' +
            f'padding:8px 16px;margin-bottom:8px;font-size:.82rem;">' +
            f'<span style="color:#FFA500;font-weight:700">{query}</span>' +
            f' <span style="color:#888888">· {sector} · {year_t}</span></div>',
            unsafe_allow_html=True,
        )
        _render_result(query, sector, sr)
        return

    # ── Nova análise ─────────────────────────────────────────────────────────
    sector = get_sector(query)
    st.markdown(
        f'<div style="background:#1C1C1C;border:1px solid #30363D;border-radius:6px;' +
        f'padding:8px 16px;margin-bottom:8px;font-size:.82rem;">' +
        f'<span style="color:#FFA500;font-weight:700">{query}</span>' +
        f' <span style="color:#888888">· {sector} · Ano: {year_t} vs {year_t - 1}</span></div>',
        unsafe_allow_html=True,
    )

    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    with st.spinner(f"Baixando dados da CVM para {query}…"):
        try:
            fetcher = CVMDataFetcher()
            fd_t, fd_t1 = fetcher.get_financial_data(query, year_t=year_t, year_t1=year_t - 1)
        except Exception as exc:
            st.error(f"**Erro ao buscar dados:** {exc}")
            st.markdown("> 💡 Use a aba **Demo** para testar com dados sintéticos.")
            return

    from advisor_brain_fsa.sector_scorer import get_scorer
    scorer = get_scorer(sector)
    sr = scorer.score(fd_t, fd_t1)

    # Salva resultado em cache de sessão para persistir entre reruns (ex: clique no Gemini)
    st.session_state["_analyze_cache"] = {
        "query": query, "sector": sector, "sr": sr, "year_t": year_t
    }

    _render_result(query, sector, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Ranking em Lote
# ─────────────────────────────────────────────────────────────────────────────

def tab_rank():
    st.header("📊 Ranking de Mercado")
    st.caption("Processa múltiplos tickers e ordena por nível de risco.")

    selected = st.multiselect(
        "Tickers",
        options=sorted(TICKER_TO_KEYWORD.keys()),
        default=["PETR4", "VALE3", "ITUB4", "ABEV3", "ELET3"],
        format_func=lambda t: f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
    )
    use_all = st.checkbox("Usar watchlist padrão completo (24 tickers)", value=False)
    tickers = DEFAULT_WATCHLIST if use_all else selected

    if not tickers:
        st.warning("Selecione ao menos um ticker.")
        return
    if not st.button("Calcular Ranking", type="primary"):
        return

    results = _load_ranking(tickers, year_t)
    df = _to_dataframe(results, top_flags=3)

    ok  = df[df["Nível de Alerta"] != "N/D"]
    err = df[df["Nível de Alerta"] == "N/D"]

    # Métricas
    counts = ok["Nível de Alerta"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Crítico",    counts.get("Crítico",    0))
    c2.metric("🟠 Alto Risco", counts.get("Alto Risco", 0))
    c3.metric("🟡 Atenção",    counts.get("Atenção",    0))
    c4.metric("🟢 Normal",     counts.get("Normal",     0))

    st.divider()

    # Tabela — colunas adaptadas ao tipo de scorer presente
    want_cols = ["Ticker","Setor","Scorer","Score de Risco","M-Score",
                 "Nível de Alerta","Accrual Ratio","Δ vs Grupo","Red Flag 1"]
    cols = [c for c in want_cols if c in ok.columns]

    def _ca(val):
        return {"Crítico":"background-color:#2a0000;color:#FF3E3E;font-family:monospace",
                "Alto Risco":"background-color:#2a1a00;color:#FFA500;font-family:monospace",
                "Atenção":"background-color:#2a2000;color:#FFD700;font-family:monospace",
                "Normal":"background-color:#002200;color:#00FF00;font-family:monospace"}.get(val, "")

    def _cm(val):
        if pd.isna(val): return ""
        return "color:#FF3E3E;font-weight:700;font-family:monospace" if val > -1.78 else "color:#00FF00;font-weight:700;font-family:monospace"

    fmt = {"Score de Risco": "{:.2f}", "Accrual Ratio": "{:+.4f}", "Δ vs Grupo": "{:+.4f}"}
    if "M-Score" in cols:
        fmt["M-Score"] = "{:+.4f}"

    styled = (ok[cols].style
              .applymap(_ca, subset=["Nível de Alerta"])
              .applymap(_cm, subset=["M-Score"])
              .format(fmt, na_rep="—"))
    st.dataframe(styled, use_container_width=True, height=400)

    if len(ok) > 1:
        beneish_ok = ok[ok["Scorer"] == "beneish"]
        if not beneish_ok.empty:
            st.markdown("#### M-Score Médio por Setor (empresas não-financeiras)")
            st.plotly_chart(_sector_bar(beneish_ok), use_container_width=True)

    if not err.empty:
        with st.expander(f"⚠️ {len(err)} ticker(s) sem dados"):
            st.dataframe(err[["Ticker","Setor","Erro"]], hide_index=True)

    # Download CSV
    import io as _io
    buf = _io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("⬇️ Exportar CSV", data=buf.getvalue(),
                       file_name=f"ranking_{year_t}.csv", mime="text/csv")

    # ── Tarefa 3 — Acesso Rápido + Drill-down ────────────────────────────────
    st.divider()
    ok_tickers = list(ok["Ticker"])

    qcol, ncol = st.columns([3, 1])
    with qcol:
        st.markdown(
            '<span style="font-size:.8rem;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.08em;color:#888888">⚡ Acesso Rápido</span>',
            unsafe_allow_html=True,
        )
        quick = st.selectbox(
            "Selecione para análise inline ou navegue para Análise Individual:",
            options=["— selecione —"] + ok_tickers,
            key="rank_quick_nav",
            label_visibility="collapsed",
        )
    with ncol:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(
            "Abrir em Análise Individual →",
            disabled=(quick == "— selecione —"),
            key="rank_nav_btn",
        ):
            if quick != "— selecione —":
                st.session_state["navigate_ticker"] = quick
                st.session_state["home_selected"] = quick

    if quick and quick != "— selecione —":
        match = [r for r in results if r.ticker == quick and r.ok]
        if match:
            r = match[0]
            st.markdown(
                f'<div style="background:#1C1C1C;border:1px solid #30363D;border-radius:8px;'
                f'padding:10px 16px;margin:8px 0;">'
                f'<span style="font-family:monospace;font-size:1rem;font-weight:700;color:#FFA500">'
                f'{quick}</span>'
                f'<span style="color:#888888;font-size:.83rem"> · {r.sector} · scorer: {r.sector_risk.scorer_type}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _render_result(quick, r.sector, r.sector_risk)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Demo (dados sintéticos, sem rede)
# ─────────────────────────────────────────────────────────────────────────────

def tab_demo():
    st.header("🧪 Demo — Dados Sintéticos")
    st.caption("Funciona sem conexão com a CVM. Ideal para testar o pipeline e a IA.")

    from advisor_brain_fsa.sector_scorer import BeneishSectorScorer
    empresa = st.radio("Empresa", list(_DEMO_DATA.keys()), horizontal=True)
    fd_t, fd_t1 = _DEMO_DATA[empresa]
    sr = BeneishSectorScorer().score(fd_t, fd_t1)
    ticker_demo = "SAFE3" if "Safe" in empresa else "RISKY4"
    sector_demo = "Energia" if "Safe" in empresa else "Consumo"
    _render_result(ticker_demo, sector_demo, sr)


# ─────────────────────────────────────────────────────────────────────────────
# Tarefa 3 — Rodapé legal fixo (CFA Institute disclaimer)
# ─────────────────────────────────────────────────────────────────────────────

def _render_footer() -> None:
    """
    Tarefa 4 — Disclaimer institucional fixo: CFA Institute + isenção de IA
    e de recomendação de investimentos.
    """
    st.markdown(
        """
        <div class="legal-footer">
        <b>CFA Institute:</b>
        Indicadores quantitativos (M-Score, Accruals, índices Beneish) são sinais de alerta,
        não prova de fraude. Anomalias podem ter origens operacionais legítimas e
        <b>exigem análise qualitativa profunda</b> conforme os padrões do CFA Institute.
        &nbsp;·&nbsp;
        <b>IA &amp; Dados:</b>
        Ferramenta baseada em IA (Gemini 2.5 Flash) — narrativas geradas automaticamente e dados da CVM
        podem conter erros de cálculo ou interpretação.
        &nbsp;·&nbsp;
        <b>Não constitui recomendação de compra ou venda de ativos.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Roteamento por abas
# ─────────────────────────────────────────────────────────────────────────────

tab0, tab1, tab2, tab3 = st.tabs([
    "🏠 Início",
    "🔍 Análise Individual",
    "📊 Ranking de Mercado",
    "🧪 Demo",
])

with tab0: tab_home()
with tab1: tab_analyze()
with tab2: tab_rank()
with tab3: tab_demo()

# Footer fixo aparece em todas as abas (injetado uma única vez no DOM)
_render_footer()
