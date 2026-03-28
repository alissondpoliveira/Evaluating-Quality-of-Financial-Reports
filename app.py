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
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Grade badge ─────────────────────────────────────────────────── */
    .grade-badge {
        display:inline-block; font-size:3rem; font-weight:900;
        width:90px; height:90px; line-height:90px;
        text-align:center; border-radius:50%; color:white;
    }
    .grade-A{background:#22c55e;} .grade-B{background:#84cc16;}
    .grade-C{background:#eab308;color:#1a1a1a;} .grade-D{background:#f97316;}
    .grade-F{background:#ef4444;}

    /* ── Alert pills ─────────────────────────────────────────────────── */
    .pill{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:600;font-size:.85rem;}
    .pill-critico{background:#fee2e2;color:#b91c1c;}
    .pill-alto{background:#ffedd5;color:#c2410c;}
    .pill-atencao{background:#fef9c3;color:#854d0e;}
    .pill-normal{background:#dcfce7;color:#15803d;}

    /* ── Red flag ────────────────────────────────────────────────────── */
    .flag-item{background:#1e1b4b;border-left:4px solid #f97316;
               padding:8px 14px;margin:6px 0;border-radius:0 8px 8px 0;
               font-size:.9rem;color:#fde68a;}

    /* ── Index explanation cards ─────────────────────────────────────── */
    .idx-explain{background:#0f172a;border:1px solid #1e293b;
                 border-radius:10px;padding:14px 16px;margin:6px 0;color:#cbd5e1;}

    /* ── Tarefa 4 — Equity Research dark metric cards ─────────────────── */
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-family: 'JetBrains Mono','Fira Code','Cascadia Code','Consolas',monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -0.02em;
    }
    .metric-value.ok  { color: #4ade80; }
    .metric-value.warn{ color: #fb923c; }
    .metric-value.crit{ color: #f87171; }
    .metric-delta {
        font-family: 'JetBrains Mono','Fira Code',monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 2px;
    }

    /* ── Scorer column header (Home dashboard) ───────────────────────── */
    .scorer-header {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
        margin-bottom: 0;
    }
    .scorer-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
    }

    /* ── Ticker row in top-5 panel ───────────────────────────────────── */
    .ticker-row {
        background: #111827;
        border: 1px solid #1e293b;
        border-top: none;
        padding: 8px 14px;
        font-family: 'JetBrains Mono','Fira Code',monospace;
        font-size: 0.88rem;
        color: #cbd5e1;
        cursor: pointer;
        transition: background 0.15s;
    }
    .ticker-row:hover { background: #1e293b; }

    /* ── Tarefa 3 — Fixed legal footer ───────────────────────────────── */
    .legal-footer {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: rgba(2, 6, 23, 0.96);
        border-top: 1px solid #1e293b;
        padding: 7px 24px;
        font-size: 0.68rem;
        color: #475569;
        z-index: 9999;
        backdrop-filter: blur(6px);
        line-height: 1.5;
    }
    .legal-footer b { color: #64748b; }

    /* Compensate for fixed footer height */
    .main .block-container { padding-bottom: 80px !important; }

    /* ── Monospace numbers in all st.metric widgets ──────────────────── */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono','Fira Code','Consolas',monospace !important;
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
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Advisor-Brain-FSA")
    st.caption("Qualidade de Relatórios Financeiros")
    st.divider()

    current_year = date.today().year
    year_t = st.selectbox(
        "Ano de Referência",
        options=list(range(current_year - 1, current_year - 6, -1)),
        index=0,
    )

    st.divider()
    _key = _get_api_key()
    if _key:
        st.success("Gemini API configurada", icon="✅")
    else:
        st.warning(
            "GOOGLE_API_KEY não encontrada.\n\n"
            "Configure em `.streamlit/secrets.toml` ou via variável de ambiente "
            "para habilitar a Tese de Risco com IA.",
            icon="🔑",
        )

    st.divider()
    st.markdown("""
**Modelo:** Beneish M-Score (1999)
**Accruals:** CFA Level 2 — Sloan (1996)
**IA:** Gemini 2.0 Flash
    """)
    st.caption("v0.5.0 — dados: Portal CVM Dados Abertos")

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
    color = "#ef4444" if val > -1.78 else "#16a34a"
    return f'<span style="color:{color};font-size:2rem;font-weight:900">{val:+.4f}</span>'

# ─────────────────────────────────────────────────────────────────────────────
# Gráficos Plotly
# ─────────────────────────────────────────────────────────────────────────────

def _gauge_chart(m_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=m_score,
        delta={"reference": -1.78, "valueformat": ".3f"},
        number={"valueformat": "+.4f", "font": {"size": 28}},
        gauge={
            "axis": {"range": [-5, 2], "tickwidth": 1},
            "bar": {"color": "#ef4444" if m_score > -1.78 else "#22c55e", "thickness": 0.25},
            "steps": [
                {"range": [-5, -1.78], "color": "#dcfce7"},
                {"range": [-1.78, 2],  "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#b91c1c", "width": 3}, "thickness": 0.8, "value": -1.78},
        },
        title={"text": "M-Score  (limiar −1.78)", "font": {"size": 14}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def _radar_chart(mscore) -> go.Figure:
    idx = ["DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA"]
    vals = [mscore.dsri, mscore.gmi, mscore.aqi, mscore.sgi,
            mscore.depi, mscore.sgai, mscore.lvgi, mscore.tata]
    thresh = [_THRESHOLDS[i] for i in idx]
    colors = ["#ef4444" if v > t else "#22c55e" for v, t in zip(vals, thresh)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=thresh+[thresh[0]], theta=idx+[idx[0]], fill="toself",
        fillcolor="rgba(59,130,246,0.08)", line=dict(color="rgba(59,130,246,0.4)", dash="dash"),
        name="Limiar não-manipulador",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=idx+[idx[0]], fill="toself",
        fillcolor="rgba(239,68,68,0.08)", line=dict(color="#ef4444", width=2),
        marker=dict(color=colors, size=8), name="Empresa",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                      showlegend=True, height=360, margin=dict(l=30,r=30,t=40,b=30),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def _sector_bar(df: pd.DataFrame) -> go.Figure:
    ok = df[df["M-Score"].notna()]
    if ok.empty:
        return go.Figure()
    avg = ok.groupby("Setor")["M-Score"].mean().sort_values()
    colors = ["#ef4444" if v > -1.78 else "#22c55e" for v in avg.values]
    fig = go.Figure(go.Bar(
        x=avg.values, y=avg.index, orientation="h",
        marker_color=colors, text=[f"{v:+.3f}" for v in avg.values], textposition="outside",
    ))
    fig.add_vline(x=-1.78, line_dash="dot", line_color="#b91c1c",
                  annotation_text="−1.78", annotation_position="top")
    fig.update_layout(height=max(250, len(avg)*40), margin=dict(l=10,r=60,t=20,b=30),
                      paper_bgcolor="rgba(0,0,0,0)", xaxis_title="M-Score Médio")
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
                color = "#ef4444" if above else "#16a34a"
                st.markdown(
                    f"<div style='font-size:2.2rem;font-weight:900;color:{color}'>{value:+.4f}</div>"
                    f"<div style='font-size:.8rem;color:#64748b'>Limiar Beneish 1999: {threshold}</div>"
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
        risk_color = "#ef4444" if sector_risk.risk_score >= 5 else (
                     "#f97316" if sector_risk.risk_score >= 3 else "#22c55e")
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

    # ── Tese de Risco (IA — apenas para Beneish por ora) ───────────────────
    st.divider()
    st.markdown("#### 🤖 Tese de Risco — Narrativa Gemini")

    if stype != "beneish" or not ms or not cfq:
        st.info(
            f"A narrativa IA está disponível para empresas industriais (Beneish). "
            f"Para o scorer **{stype}**, os indicadores estão na seção acima.",
            icon="ℹ️",
        )
        return

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
    # ── Tarefa 1: Dashboard Setorial em 3 colunas ──────────────────────────
    st.markdown(
        '<div style="font-size:1.5rem;font-weight:700;margin-bottom:2px">'
        '🏠 Dashboard Setorial B3</div>'
        '<div style="color:#64748b;font-size:.85rem;margin-bottom:16px">'
        'Top 5 por risco — Industriais · Bancos & Financeiro · Seguros. '
        'Selecione um ativo para ver análise detalhada inline.</div>',
        unsafe_allow_html=True,
    )

    key_res = f"home_results_{year_t}"

    # ── Controles de carga ──────────────────────────────────────────────────
    col_btn, col_reset = st.columns([2, 1])
    with col_btn:
        load = st.button("🔄 Carregar / Atualizar Ranking", type="primary",
                         disabled=(key_res in st.session_state))
    with col_reset:
        if st.button("Limpar cache", disabled=(key_res not in st.session_state)):
            del st.session_state[key_res]
            if "home_selected" in st.session_state:
                del st.session_state["home_selected"]
            st.rerun()

    if load:
        st.session_state[key_res] = _load_ranking(DEFAULT_WATCHLIST, year_t)
        st.session_state.pop("home_selected", None)
        st.rerun()

    if key_res not in st.session_state:
        st.info("Clique em **Carregar / Atualizar Ranking** para buscar os dados da CVM.")
        return

    results = st.session_state[key_res]
    df = _to_dataframe(results, top_flags=1)
    ok = df[df["Score de Risco"].notna()].copy()

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
        """Renderiza um painel Top-5 com botões interativos (Tarefa 1)."""
        top5 = _top5(scorer_type)
        with col_container:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid {border_color};'
                f'border-radius:8px 8px 0 0;padding:10px 14px;">'
                f'<span style="font-size:.75rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.1em;color:{border_color}">{icon} {title}</span></div>',
                unsafe_allow_html=True,
            )
            if top5.empty:
                st.markdown(
                    '<div style="background:#111827;border:1px solid #1e293b;border-top:none;'
                    'padding:12px 14px;color:#475569;font-size:.83rem;">Sem dados</div>',
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
                bg = "#1e3a5f" if is_sel else "#111827"
                st.markdown(
                    f'<div style="background:{bg};border:1px solid #1e293b;border-top:none;'
                    f'padding:1px 6px;">', unsafe_allow_html=True,
                )
                if st.button(
                    f"{a_icon}  **{ticker}** — {score}   ·  _{setor}_",
                    key=btn_key,
                    use_container_width=True,
                ):
                    if st.session_state.get("home_selected") == ticker:
                        st.session_state.pop("home_selected", None)
                    else:
                        st.session_state["home_selected"] = ticker
                        # Signal tab_analyze to pre-fill (Tarefa 1 — cross-tab nav)
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
            'letter-spacing:.08em;color:#64748b">M-Score Médio por Setor (Industriais)</span>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_sector_bar(beneish_ok), use_container_width=True)

    # ── Análise Detalhada inline (Tarefa 1 — interatividade) ────────────────
    selected = st.session_state.get("home_selected")
    if selected:
        match = [r for r in results if r.ticker == selected and r.ok]
        if match:
            r = match[0]
            st.divider()
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e3a5f;'
                f'border-radius:8px;padding:12px 18px;margin-bottom:12px;">'
                f'<span style="font-family:monospace;font-size:1.1rem;font-weight:700;'
                f'color:#60a5fa">{selected}</span>'
                f'<span style="color:#64748b;font-size:.85rem"> · {r.sector} · {year_t}</span>'
                f'<span style="float:right;font-size:.75rem;color:#475569">'
                f'scorer: {r.sector_risk.scorer_type}</span></div>',
                unsafe_allow_html=True,
            )
            st.info(
                "Para análise completa com IA (Gemini), acesse a aba **🔍 Análise Individual**.",
                icon="💡",
            )
            _render_result(selected, r.sector, r.sector_risk)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Análise Individual (CVM real)
# ─────────────────────────────────────────────────────────────────────────────

def tab_analyze():
    st.header("🔍 Análise Individual")
    st.caption("Busca dados reais no Portal CVM e calcula M-Score + qualidade de accruals.")

    # ── Tarefa 1: pre-fill ticker quando navegando da Home ──────────────────
    _nav = st.session_state.pop("navigate_ticker", None)
    all_tickers = sorted(TICKER_TO_KEYWORD.keys())
    _default_idx = (all_tickers.index(_nav) + 1) if (_nav and _nav in all_tickers) else 0

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker_sel = st.selectbox(
            "Ticker B3",
            options=[""] + all_tickers,
            index=_default_idx,
            format_func=lambda t: t if not t else f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
        )
        custom = st.text_input("…ou CNPJ / nome livre", placeholder="33.000.167/0001-01 ou Petrobras")
    query = custom.strip() if custom.strip() else ticker_sel

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        # Auto-run if navigated from Home
        run = st.button("Calcular", type="primary", disabled=not query) or bool(_nav and query)

    if not run or not query:
        st.info("Selecione ou digite um ticker acima e clique em **Calcular**.")
        return

    sector = get_sector(query)
    st.markdown(f"**{query}** · Setor: {sector} · Ano: {year_t} vs {year_t - 1}")

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
        return {"Crítico":"background-color:#fee2e2;color:#b91c1c",
                "Alto Risco":"background-color:#ffedd5;color:#c2410c",
                "Atenção":"background-color:#fef9c3;color:#854d0e",
                "Normal":"background-color:#dcfce7;color:#15803d"}.get(val, "")

    def _cm(val):
        if pd.isna(val): return ""
        return "color:#ef4444;font-weight:700" if val > -1.78 else "color:#16a34a;font-weight:700"

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

    # Drill-down
    st.divider()
    st.markdown("#### 🔎 Explorar empresa em detalhe")
    ok_tickers = list(ok["Ticker"])
    chosen = st.selectbox("Selecione o ticker:", ["—"] + ok_tickers, key="rank_detail")
    if chosen and chosen != "—":
        match = [r for r in results if r.ticker == chosen and r.ok]
        if match:
            r = match[0]
            st.markdown(f"**{chosen}** · {r.sector}")
            _render_result(chosen, r.sector, r.sector_risk)


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
    st.markdown(
        """
        <div class="legal-footer">
        <b>⚠️ Aviso Legal:</b> Os indicadores apresentados (como M-Score e Accruals) são sinais
        de alerta e não garantem a existência de fraude ou erro. Conforme as diretrizes do
        <b>CFA Institute</b>, anomalias quantitativas podem ter origens operacionais legítimas
        e exigem análise qualitativa profunda. &nbsp;|&nbsp;
        <b>⚠️ Isenção de Responsabilidade:</b> Esta é uma ferramenta experimental baseada em IA
        e processamento automático de dados da CVM; pode conter erros de cálculo ou interpretação.
        Este conteúdo <b>não constitui recomendação de compra ou venda de ativos</b>.
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
