"""
Advisor-Brain-FSA — Streamlit App
===================================
Interface web para análise de qualidade de relatórios financeiros.

Rodar:
    streamlit run app.py
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

# ── módulos do projeto ────────────────────────────────────────────────────────
from advisor_brain_fsa import BeneishMScore, CashFlowQuality
from advisor_brain_fsa.beneish_mscore import FinancialData
from advisor_brain_fsa.mda_analyst import MDAnalyst, compute_grade
from advisor_brain_fsa.rank_market import (
    DEFAULT_WATCHLIST,
    _apply_sector_stats,
    _to_dataframe,
    CompanyResult,
    detect_red_flags,
)
from advisor_brain_fsa.report_generator import generate_report
from advisor_brain_fsa.ticker_map import TICKER_TO_KEYWORD, get_sector

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
# CSS customizado
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Grade badge */
    .grade-badge {
        display: inline-block;
        font-size: 3rem;
        font-weight: 900;
        width: 90px; height: 90px;
        line-height: 90px;
        text-align: center;
        border-radius: 50%;
        color: white;
    }
    .grade-A { background: #22c55e; }
    .grade-B { background: #84cc16; }
    .grade-C { background: #eab308; color: #1a1a1a; }
    .grade-D { background: #f97316; }
    .grade-F { background: #ef4444; }

    /* Alert pills */
    .pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .pill-critico   { background:#fee2e2; color:#b91c1c; }
    .pill-alto      { background:#ffedd5; color:#c2410c; }
    .pill-atencao   { background:#fef9c3; color:#854d0e; }
    .pill-normal    { background:#dcfce7; color:#15803d; }

    /* Red flag items */
    .flag-item {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 8px 14px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }

    /* Index card */
    .idx-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .idx-val { font-size: 1.5rem; font-weight: 700; }
    .idx-ok  { color: #16a34a; }
    .idx-warn{ color: #ea580c; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/CFA-Level%202-003087?style=for-the-badge", width=160)
    st.title("Advisor-Brain-FSA")
    st.caption("Análise de Qualidade de Relatórios Financeiros")
    st.divider()

    st.subheader("⚙️ Configuração")

    api_key = st.text_input(
        "ANTHROPIC_API_KEY",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Necessária apenas para o módulo de Tese de Risco (IA).",
        placeholder="sk-ant-...",
    )

    current_year = date.today().year
    year_t = st.selectbox(
        "Ano de Referência",
        options=list(range(current_year - 1, current_year - 6, -1)),
        index=0,
    )

    st.divider()
    st.markdown("""
**Modelo:** Beneish M-Score (1999)
**Accruals:** CFA Level 2 — Sloan (1996)
**IA:** claude-opus-4-6 + adaptive thinking
    """)
    st.caption("v0.4.0 — dados: Portal CVM Dados Abertos")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers de UI
# ─────────────────────────────────────────────────────────────────────────────

_ALERT_PILL = {
    "Crítico":    ("pill-critico",  "🔴 Crítico"),
    "Alto Risco": ("pill-alto",     "🟠 Alto Risco"),
    "Atenção":    ("pill-atencao",  "🟡 Atenção"),
    "Normal":     ("pill-normal",   "🟢 Normal"),
}

_GRADE_COLOR = {"A": "#22c55e", "B": "#84cc16", "C": "#eab308", "D": "#f97316", "F": "#ef4444"}
_INDEX_THRESHOLDS = {
    "DSRI": 1.031, "GMI": 1.014, "AQI": 1.039, "SGI": 1.134,
    "DEPI": 1.017, "SGAI": 1.054, "LVGI": 1.000, "TATA": -0.012,
}
_INDEX_DESCRIPTIONS = {
    "DSRI": "Recebíveis vs Receita",
    "GMI":  "Margem Bruta",
    "AQI":  "Qualidade de Ativos",
    "SGI":  "Crescimento de Vendas",
    "DEPI": "Taxa de Depreciação",
    "SGAI": "Despesas G&A",
    "LVGI": "Alavancagem",
    "TATA": "Accruals / Ativos",
}


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

def _radar_chart(mscore) -> go.Figure:
    """Radar dos 8 índices vs limiares não-manipuladores."""
    indices = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
    values  = [mscore.dsri, mscore.gmi, mscore.aqi, mscore.sgi,
                mscore.depi, mscore.sgai, mscore.lvgi, mscore.tata]
    threshs = [_INDEX_THRESHOLDS[i] for i in indices]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=threshs + [threshs[0]],
        theta=indices + [indices[0]],
        fill="toself",
        fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="rgba(59,130,246,0.4)", dash="dash"),
        name="Limiar não-manipulador",
    ))
    colors = ["#ef4444" if v > t else "#22c55e" for v, t in zip(values, threshs)]
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=indices + [indices[0]],
        fill="toself",
        fillcolor="rgba(239,68,68,0.08)",
        line=dict(color="#ef4444", width=2),
        marker=dict(color=colors, size=8),
        name="Empresa",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        showlegend=True,
        height=360,
        margin=dict(l=30, r=30, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _gauge_chart(m_score: float) -> go.Figure:
    """Gauge do M-Score com zona de risco destacada."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=m_score,
        delta={"reference": -1.78, "valueformat": ".3f"},
        number={"valueformat": "+.4f", "font": {"size": 28}},
        gauge={
            "axis": {"range": [-5, 2], "tickwidth": 1},
            "bar":  {"color": "#ef4444" if m_score > -1.78 else "#22c55e", "thickness": 0.25},
            "steps": [
                {"range": [-5, -1.78], "color": "#dcfce7"},
                {"range": [-1.78, 2],  "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": "#b91c1c", "width": 3},
                "thickness": 0.8,
                "value": -1.78,
            },
        },
        title={"text": "M-Score  (limiar −1.78)", "font": {"size": 14}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _bar_indices(mscore) -> go.Figure:
    """Barras horizontais dos índices vs limiar."""
    indices = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
    values  = [mscore.dsri, mscore.gmi, mscore.aqi, mscore.sgi,
                mscore.depi, mscore.sgai, mscore.lvgi, mscore.tata]
    threshs = [_INDEX_THRESHOLDS[i] for i in indices]
    colors  = ["#ef4444" if v > t else "#22c55e" for v, t in zip(values, threshs)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values, y=indices, orientation="h",
        marker_color=colors, text=[f"{v:.3f}" for v in values],
        textposition="outside", name="Valor",
    ))
    for i, (idx, t) in enumerate(zip(indices, threshs)):
        fig.add_shape(type="line",
            x0=t, x1=t, y0=i - 0.4, y1=i + 0.4,
            line=dict(color="#1e40af", width=2, dash="dot"))
    fig.update_layout(
        showlegend=False, height=320,
        margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Valor do Índice",
    )
    return fig


def _sector_bar(df: pd.DataFrame) -> go.Figure:
    """M-Score médio por setor."""
    ok = df[df["M-Score"].notna()]
    if ok.empty:
        return go.Figure()
    sector_avg = ok.groupby("Setor")["M-Score"].mean().sort_values()
    colors = ["#ef4444" if v > -1.78 else "#22c55e" for v in sector_avg.values]
    fig = go.Figure(go.Bar(
        x=sector_avg.values, y=sector_avg.index, orientation="h",
        marker_color=colors, text=[f"{v:+.3f}" for v in sector_avg.values],
        textposition="outside",
    ))
    fig.add_vline(x=-1.78, line_dash="dot", line_color="#b91c1c",
                  annotation_text="−1.78", annotation_position="top")
    fig.update_layout(
        height=max(250, len(sector_avg) * 40),
        margin=dict(l=10, r=60, t=20, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="M-Score Médio",
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
# Bloco de resultado: compartilhado entre abas
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(ticker: str, sector: str, mscore, cfq, flags, api_key: str):
    grade, grade_label = compute_grade(mscore.m_score, cfq.accrual_ratio)
    alert = cfq.alert_level.value

    # ── Cabeçalho ─────────────────────────────────────────────────────────────
    col_grade, col_score, col_alert = st.columns([1, 2, 2])

    with col_grade:
        st.markdown(_grade_html(grade), unsafe_allow_html=True)
        st.caption(grade_label)

    with col_score:
        st.markdown("**M-Score**")
        st.markdown(_fmt_mscore(mscore.m_score), unsafe_allow_html=True)
        st.caption(f"Limiar: −1.78 → **{mscore.classification}**")

    with col_alert:
        st.markdown("**Nível de Alerta**")
        st.markdown(_pill_html(alert), unsafe_allow_html=True)
        st.caption(
            f"Accrual Ratio: {cfq.accrual_ratio:+.4f} | "
            f"Qualidade: {cfq.earnings_quality}"
        )

    st.divider()

    # ── Gráficos ───────────────────────────────────────────────────────────────
    col_gauge, col_radar = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(_gauge_chart(mscore.m_score), use_container_width=True)
    with col_radar:
        st.plotly_chart(_radar_chart(mscore), use_container_width=True)

    # ── Índices detalhados ─────────────────────────────────────────────────────
    with st.expander("📊 Índices Beneish detalhados", expanded=False):
        st.plotly_chart(_bar_indices(mscore), use_container_width=True)

        idx_names = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
        idx_vals  = [mscore.dsri, mscore.gmi, mscore.aqi, mscore.sgi,
                     mscore.depi, mscore.sgai, mscore.lvgi, mscore.tata]
        rows = []
        for n, v in zip(idx_names, idx_vals):
            t = _INDEX_THRESHOLDS[n]
            status = "⚠️ Acima" if v > t else "✅ OK"
            rows.append({"Índice": n, "Descrição": _INDEX_DESCRIPTIONS[n],
                         "Valor": f"{v:.4f}", "Limiar (B99)": f"{t:.3f}", "Status": status})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Red Flags ──────────────────────────────────────────────────────────────
    st.markdown("#### 🚩 Red Flags Detectados")
    if flags:
        for f in flags:
            st.markdown(f'<div class="flag-item">{f}</div>', unsafe_allow_html=True)
    else:
        st.success("Nenhum red flag acima do limiar detectado.")

    # ── Tese de Risco (IA) ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🤖 Tese de Risco — Narrativa Claude")

    if not api_key:
        st.info("Configure sua **ANTHROPIC_API_KEY** na barra lateral para gerar a narrativa de IA.", icon="🔑")
        return

    if st.button("Gerar Tese de Risco com Claude", type="primary", key=f"ai_{ticker}"):
        analyst = MDAnalyst(api_key=api_key)
        with st.spinner("Claude está analisando (adaptive thinking)…"):
            placeholder = st.empty()
            full_text = []
            try:
                for chunk in analyst.analyze_streaming(
                    ticker=ticker, sector=sector, year=year_t,
                    mscore_result=mscore, cfq_result=cfq, red_flags=flags,
                ):
                    full_text.append(chunk)
                    placeholder.markdown("".join(full_text))
            except Exception as exc:
                st.error(f"Erro na API: {exc}")
                return

        # Download button
        st.download_button(
            "⬇️ Baixar relatório .md",
            data="".join(full_text),
            file_name=f"tese_risco_{ticker}_{year_t}.md",
            mime="text/markdown",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Análise Individual (CVM real)
# ─────────────────────────────────────────────────────────────────────────────

def tab_analyze():
    st.header("🔍 Análise Individual")
    st.caption("Busca dados reais no Portal CVM Dados Abertos e calcula o M-Score.")

    all_tickers = sorted(TICKER_TO_KEYWORD.keys())
    col_input, col_btn = st.columns([3, 1])

    with col_input:
        ticker_input = st.selectbox(
            "Ticker B3 / CNPJ / Nome da empresa",
            options=[""] + all_tickers,
            index=0,
            format_func=lambda t: t if not t else f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
        )
        custom_input = st.text_input(
            "…ou digite um CNPJ / nome livre",
            placeholder="33.000.167/0001-01  ou  Petrobras",
        )

    query = custom_input.strip() if custom_input.strip() else ticker_input

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
            st.markdown(
                "> 💡 O Portal CVM pode estar offline ou o DFP para este ano ainda não foi publicado. "
                "Use a aba **Demo** para testar com dados sintéticos."
            )
            return

    mscore = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
    cfq    = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore)
    flags  = detect_red_flags(mscore)

    _render_result(query, sector, mscore, cfq, flags, api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Ranking em Lote (CVM real)
# ─────────────────────────────────────────────────────────────────────────────

def tab_rank():
    st.header("📊 Ranking de Mercado")
    st.caption("Processa múltiplos tickers e ordena por nível de risco.")

    selected = st.multiselect(
        "Selecione os tickers",
        options=sorted(TICKER_TO_KEYWORD.keys()),
        default=["PETR4", "VALE3", "ITUB4", "ABEV3", "ELET3"],
        format_func=lambda t: f"{t} — {TICKER_TO_KEYWORD.get(t, t)}",
    )
    use_watchlist = st.checkbox("Usar watchlist padrão completo (24 tickers)", value=False)
    tickers = DEFAULT_WATCHLIST if use_watchlist else selected

    if not tickers:
        st.warning("Selecione ao menos um ticker.")
        return

    if not st.button("Calcular Ranking", type="primary"):
        return

    progress = st.progress(0, text="Iniciando…")
    results  = []

    from advisor_brain_fsa.data_fetcher import CVMDataFetcher
    fetcher = CVMDataFetcher()

    for i, ticker in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Processando {ticker}…")
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t - 1)
            ms   = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
            cfq  = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(ms)
            flags = detect_red_flags(ms)
            results.append(CompanyResult(
                ticker=ticker, sector=get_sector(ticker), year_t=year_t,
                mscore=ms, cfq=cfq, red_flags=flags,
            ))
        except Exception as exc:
            results.append(CompanyResult(
                ticker=ticker, sector=get_sector(ticker), year_t=year_t,
                mscore=None, cfq=None, red_flags=[], error=str(exc),
            ))
        time.sleep(0.3)

    progress.empty()
    _apply_sector_stats(results)
    df = _to_dataframe(results, top_flags=3)

    _render_ranking(df)


def _render_ranking(df: pd.DataFrame):
    ok  = df[df["Nível de Alerta"] != "N/D"]
    err = df[df["Nível de Alerta"] == "N/D"]

    # ── Métricas resumo ────────────────────────────────────────────────────────
    counts = ok["Nível de Alerta"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Crítico",    counts.get("Crítico",    0))
    c2.metric("🟠 Alto Risco", counts.get("Alto Risco", 0))
    c3.metric("🟡 Atenção",    counts.get("Atenção",    0))
    c4.metric("🟢 Normal",     counts.get("Normal",     0))

    st.divider()

    # ── Tabela principal ───────────────────────────────────────────────────────
    display_cols = ["Ticker", "Setor", "M-Score", "Nível de Alerta",
                    "Accrual Ratio", "Qualidade Earnings", "Δ vs Setor",
                    "Red Flag 1"]
    available = [c for c in display_cols if c in df.columns]

    def _color_alert(val):
        colors = {
            "Crítico":    "background-color:#fee2e2;color:#b91c1c",
            "Alto Risco": "background-color:#ffedd5;color:#c2410c",
            "Atenção":    "background-color:#fef9c3;color:#854d0e",
            "Normal":     "background-color:#dcfce7;color:#15803d",
        }
        return colors.get(val, "")

    def _color_mscore(val):
        if pd.isna(val):
            return ""
        return "color:#ef4444;font-weight:700" if val > -1.78 else "color:#16a34a;font-weight:700"

    styled = (
        ok[available]
        .style
        .applymap(_color_alert, subset=["Nível de Alerta"])
        .applymap(_color_mscore, subset=["M-Score"])
        .format({"M-Score": "{:+.4f}", "Accrual Ratio": "{:+.4f}", "Δ vs Setor": "{:+.4f}"}, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=400)

    # ── Gráfico setorial ───────────────────────────────────────────────────────
    if len(ok) > 1:
        st.markdown("#### M-Score Médio por Setor")
        st.plotly_chart(_sector_bar(ok), use_container_width=True)

    # ── Erros ─────────────────────────────────────────────────────────────────
    if not err.empty:
        with st.expander(f"⚠️ {len(err)} ticker(s) sem dados"):
            st.dataframe(err[["Ticker", "Setor", "Erro"]], hide_index=True)

    # ── Download ───────────────────────────────────────────────────────────────
    import io as _io
    buf = _io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Exportar CSV",
        data=buf.getvalue(),
        file_name=f"ranking_{year_t}.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Demo (dados sintéticos, sem rede)
# ─────────────────────────────────────────────────────────────────────────────

def tab_demo():
    st.header("🧪 Demo — Dados Sintéticos")
    st.caption("Funciona sem conexão com a CVM. Útil para testar o pipeline e o módulo de IA.")

    empresa = st.radio(
        "Empresa",
        options=list(_DEMO_DATA.keys()),
        horizontal=True,
    )

    fd_t, fd_t1 = _DEMO_DATA[empresa]
    mscore = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
    cfq    = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore)
    flags  = detect_red_flags(mscore)

    ticker_demo = "SAFE3" if "Safe" in empresa else "RISKY4"
    sector_demo = "Energia" if "Safe" in empresa else "Consumo"

    _render_result(ticker_demo, sector_demo, mscore, cfq, flags, api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Roteamento por abas
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 Análise Individual",
    "📊 Ranking de Mercado",
    "🧪 Demo",
])

with tab1:
    tab_analyze()

with tab2:
    tab_rank()

with tab3:
    tab_demo()
