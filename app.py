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
    .grade-badge {
        display:inline-block; font-size:3rem; font-weight:900;
        width:90px; height:90px; line-height:90px;
        text-align:center; border-radius:50%; color:white;
    }
    .grade-A{background:#22c55e;} .grade-B{background:#84cc16;}
    .grade-C{background:#eab308;color:#1a1a1a;} .grade-D{background:#f97316;}
    .grade-F{background:#ef4444;}

    .pill{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:600;font-size:.85rem;}
    .pill-critico{background:#fee2e2;color:#b91c1c;}
    .pill-alto{background:#ffedd5;color:#c2410c;}
    .pill-atencao{background:#fef9c3;color:#854d0e;}
    .pill-normal{background:#dcfce7;color:#15803d;}

    .flag-item{background:#fff7ed;border-left:4px solid #f97316;
               padding:8px 14px;margin:6px 0;border-radius:0 8px 8px 0;font-size:.9rem;}

    .idx-explain{background:#f8fafc;border:1px solid #e2e8f0;
                 border-radius:10px;padding:14px 16px;margin:6px 0;}
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


def _render_sector_metrics(scorer_type: str, metrics: dict) -> None:
    """Cards de métricas específicas para banking e insurance."""
    if scorer_type == "banking":
        st.markdown("### 🏦 Indicadores Bancários (BACEN / Basileia)")
        c1, c2, c3 = st.columns(3)
        roa = metrics.get("roa", float("nan"))
        ci  = metrics.get("cost_income", float("nan"))
        cq  = metrics.get("cfo_quality", float("nan"))
        sp  = metrics.get("spread", float("nan"))
        rg  = metrics.get("rev_growth", float("nan"))
        c1.metric("ROA", f"{roa:.2%}" if not pd.isna(roa) else "—",
                  delta="OK" if roa > 0.008 else "Baixo", delta_color="normal" if roa > 0.008 else "inverse")
        c2.metric("Cost/Income", f"{ci:.1%}" if not pd.isna(ci) else "—",
                  delta="OK" if ci < 0.60 else "Alto", delta_color="normal" if ci < 0.60 else "inverse")
        c3.metric("CFO / Lucro", f"{cq:.2f}x" if not pd.isna(cq) else "—",
                  delta="OK" if cq > 0.50 else "Baixo", delta_color="normal" if cq > 0.50 else "inverse")
        c4, c5, _ = st.columns(3)
        c4.metric("Spread Financeiro", f"{sp:.1%}" if not pd.isna(sp) else "—")
        c5.metric("Crescimento Receita", f"{rg:.1%}" if not pd.isna(rg) else "—",
                  delta="Crescendo" if not pd.isna(rg) and rg > 1 else "Contraindo",
                  delta_color="normal" if not pd.isna(rg) and rg > 1 else "inverse")

    elif scorer_type == "insurance":
        st.markdown("### 🛡️ Indicadores de Seguros (SUSEP)")
        c1, c2, c3 = st.columns(3)
        lr = metrics.get("loss_ratio", float("nan"))
        er = metrics.get("expense_ratio", float("nan"))
        cr = metrics.get("combined_ratio", float("nan"))
        rg = metrics.get("rev_growth", float("nan"))
        roa = metrics.get("roa", float("nan"))
        c1.metric("Sinistralidade", f"{lr:.1%}" if not pd.isna(lr) else "—",
                  delta="OK" if lr < 0.65 else "Alta", delta_color="normal" if lr < 0.65 else "inverse")
        c2.metric("Índice de Despesas", f"{er:.1%}" if not pd.isna(er) else "—")
        c3.metric("Índice Combinado", f"{cr:.1%}" if not pd.isna(cr) else "—",
                  delta="Lucro técnico" if cr < 1.0 else "Prejuízo técnico",
                  delta_color="normal" if cr < 1.0 else "inverse")
        c4, c5, _ = st.columns(3)
        c4.metric("Crescimento Prêmios", f"{rg:.1%}" if not pd.isna(rg) else "—",
                  delta="Crescendo" if not pd.isna(rg) and rg > 1 else "Contraindo",
                  delta_color="normal" if not pd.isna(rg) and rg > 1 else "inverse")
        c5.metric("ROA", f"{roa:.2%}" if not pd.isna(roa) else "—")

# ─────────────────────────────────────────────────────────────────────────────
# Tab 0 — Home: Top 10 Melhores e Piores M-Score
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
    st.header("🏠 Visão Geral do Mercado")
    st.caption("Top 5 melhores e piores por segmento — watchlist padrão B3.")

    key_res = f"home_results_{year_t}"

    # ── Controles de carga ──────────────────────────────────────────────────
    col_btn, col_reset = st.columns([2, 1])
    with col_btn:
        load = st.button("🔄 Carregar / Atualizar Ranking", type="primary",
                         disabled=(key_res in st.session_state))
    with col_reset:
        if st.button("Limpar cache", disabled=(key_res not in st.session_state)):
            del st.session_state[key_res]
            st.rerun()

    if load:
        st.session_state[key_res] = _load_ranking(DEFAULT_WATCHLIST, year_t)
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

    # ── Filtro de setor ──────────────────────────────────────────────────────
    available_sectors = sorted(ok["Setor"].unique())
    all_option = "Todos os setores"
    sector_filter = st.selectbox(
        "🔍 Filtrar por segmento:",
        options=[all_option] + available_sectors,
        index=0,
        key="home_sector_filter",
    )

    filtered = ok if sector_filter == all_option else ok[ok["Setor"] == sector_filter]

    if filtered.empty:
        st.warning(f"Nenhuma empresa disponível para o setor **{sector_filter}**.")
        return

    st.divider()

    # ── Top 5 Melhores / Piores ──────────────────────────────────────────────
    # For Beneish companies rank by M-Score; for financial by Score de Risco
    is_fin_filter = sector_filter in FINANCIAL_GROUP

    def _primary_col(row):
        if row["Scorer"] in ("banking", "insurance"):
            return row["Score de Risco"]
        return row["M-Score"] if pd.notna(row["M-Score"]) else row["Score de Risco"]

    filtered = filtered.copy()
    filtered["_primary"] = filtered.apply(_primary_col, axis=1)

    # Best = lowest primary score (low M-Score or low risk_score)
    top5_best  = filtered.nsmallest(5, "_primary")
    top5_worst = filtered.nlargest(5,  "_primary")

    def _mini_table(sub: pd.DataFrame, label_col: str):
        rows = []
        for _, r in sub.iterrows():
            alerta = r.get("Nível de Alerta", "—")
            icon_map = {"Crítico": "🔴", "Alto Risco": "🟠", "Atenção": "🟡", "Normal": "🟢"}
            icon = icon_map.get(alerta, "⚪")
            score_val = r["_primary"]
            scorer = r.get("Scorer", "beneish")
            score_label = f"{score_val:+.4f}" if scorer == "beneish" else f"{score_val:.2f}/10"
            rows.append({
                "": icon,
                "Ticker": r["Ticker"],
                "Setor": r["Setor"],
                label_col: score_label,
                "Alerta": alerta,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    score_label = "Score de Risco" if is_fin_filter else "M-Score"
    col_best, col_worst = st.columns(2)
    with col_best:
        st.markdown("#### 🟢 Top 5 Melhores")
        st.caption("Menor risco no segmento selecionado")
        _mini_table(top5_best, score_label)
    with col_worst:
        st.markdown("#### 🔴 Top 5 Piores")
        st.caption("Maior risco no segmento selecionado")
        _mini_table(top5_worst, score_label)

    # ── Gráfico setorial (apenas quando "Todos") ──────────────────────────────
    if sector_filter == all_option and len(ok) > 1:
        st.divider()
        st.markdown("#### M-Score Médio por Setor")
        beneish_ok = ok[ok["Scorer"] == "beneish"]
        if not beneish_ok.empty:
            st.plotly_chart(_sector_bar(beneish_ok), use_container_width=True)

    # ── Drill-down por empresa ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔎 Explorar empresa em detalhe")
    ok_tickers = list(filtered["Ticker"])
    chosen = st.selectbox("Selecione o ticker:", ["—"] + ok_tickers, key="home_detail")

    if chosen and chosen != "—":
        match = [r for r in results if r.ticker == chosen and r.ok]
        if match:
            r = match[0]
            st.markdown(f"**{chosen}** · {r.sector} · Ano {year_t}")
            _render_result(chosen, r.sector, r.sector_risk)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Análise Individual (CVM real)
# ─────────────────────────────────────────────────────────────────────────────

def tab_analyze():
    st.header("🔍 Análise Individual")
    st.caption("Busca dados reais no Portal CVM e calcula M-Score + qualidade de accruals.")

    all_tickers = sorted(TICKER_TO_KEYWORD.keys())
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        ticker_sel = st.selectbox(
            "Ticker B3",
            options=[""] + all_tickers,
            format_func=lambda t: t if not t else f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
        )
        custom = st.text_input("…ou CNPJ / nome livre", placeholder="33.000.167/0001-01 ou Petrobras")
    query = custom.strip() if custom.strip() else ticker_sel

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Calcular", type="primary", disabled=not query)

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
