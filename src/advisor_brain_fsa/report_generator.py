"""
Report Generator
----------------
Converts the ranked DataFrame produced by rank_market() into:
  1. A Markdown table with risk-level sections and highlighted red flags.
  2. A CSV file for further analysis.

Usage
-----
    from advisor_brain_fsa.rank_market import rank_market
    from advisor_brain_fsa.report_generator import generate_report

    df = rank_market(["PETR4", "VALE3", "ITUB4"], year_t=2023)
    md_path, csv_path = generate_report(df, output_dir="reports/")
    print(open(md_path).read())
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Alert level ordering and emoji badges
# ---------------------------------------------------------------------------

_ALERT_ORDER   = ["Crítico", "Alto Risco", "Atenção", "Normal", "N/D"]
_ALERT_BADGES  = {
    "Crítico":    "🔴 Crítico",
    "Alto Risco": "🟠 Alto Risco",
    "Atenção":    "🟡 Atenção",
    "Normal":     "🟢 Normal",
    "N/D":        "⚪ N/D",
}

# Columns shown in the summary Markdown table
_SUMMARY_COLS = [
    "Ticker", "Setor", "M-Score", "Nível de Alerta",
    "Accrual Ratio", "Qualidade Earnings",
    "M-Score Médio Setor", "Δ vs Setor",
]

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    output_dir: str | Path = "reports",
    year_t: Optional[int] = None,
    prefix: str = "ranking",
) -> tuple[Path, Path]:
    """
    Write Markdown and CSV reports from a ranked DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of rank_market().
    output_dir : str | Path
        Directory where files are saved (created if absent).
    year_t : int | None
        Fiscal year label used in the report header.
    prefix : str
        File name prefix (default "ranking").

    Returns
    -------
    (md_path, csv_path) : tuple[Path, Path]
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    year  = year_t or (df["Ano"].iloc[0] if not df.empty else "")
    stem  = f"{prefix}_{year}_{ts}"

    md_path  = out / f"{stem}.md"
    csv_path = out / f"{stem}.csv"

    md_path.write_text(build_markdown(df, year_t=year), encoding="utf-8")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return md_path, csv_path


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def build_markdown(df: pd.DataFrame, year_t: Optional[int] = None) -> str:
    """Return the full Markdown report as a string."""
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    year_label = str(year_t) if year_t else (str(df["Ano"].iloc[0]) if not df.empty else "")
    lines += [
        f"# Ranking de Risco de Manipulação Contábil — {year_label}",
        "",
        f"> Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}  ",
        f"> Modelo: **Beneish M-Score** (Beneish, 1999) + **CFA Accruals Quality**  ",
        f"> Limiar M-Score: **−1.78** (acima → Potential Manipulator)  ",
        "",
    ]

    # ── Legend ───────────────────────────────────────────────────────────────
    lines += [
        "## Legenda de Níveis de Alerta",
        "",
        "| Badge | Nível | Critério |",
        "|---|---|---|",
        "| 🔴 | Crítico    | M-Score > −1.78 **E** Accrual Ratio > 5% |",
        "| 🟠 | Alto Risco | M-Score > −1.78 (accruals aceitáveis)     |",
        "| 🟡 | Atenção    | M-Score ≤ −1.78, mas Accruals altos       |",
        "| 🟢 | Normal     | M-Score ≤ −1.78 **E** Accruals baixos     |",
        "",
    ]

    # ── Executive summary ───────────────────────────────────────────────────
    if not df.empty:
        lines += _executive_summary(df)

    # ── Sector summary ──────────────────────────────────────────────────────
    if not df.empty:
        lines += _sector_table(df)

    # ── Per-level detail sections ───────────────────────────────────────────
    for level in _ALERT_ORDER:
        subset = df[df["Nível de Alerta"] == level]
        if subset.empty:
            continue
        badge = _ALERT_BADGES.get(level, level)
        lines += [f"## {badge}", ""]
        lines += _summary_table(subset)
        lines += _detail_cards(subset)

    # ── Failed tickers ───────────────────────────────────────────────────────
    errors = df[df["Erro"] != ""]
    if not errors.empty:
        lines += ["## ⚪ Dados Indisponíveis", ""]
        lines += [
            "| Ticker | Setor | Motivo |",
            "|---|---|---|",
        ]
        for _, row in errors.iterrows():
            reason = _truncate(str(row["Erro"]), 80)
            lines.append(f"| {row['Ticker']} | {row['Setor']} | {reason} |")
        lines.append("")

    # ── Beneish indices reference ─────────────────────────────────────────────
    lines += _index_reference()

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _executive_summary(df: pd.DataFrame) -> list[str]:
    ok = df[df["Nível de Alerta"] != "N/D"]
    counts = ok["Nível de Alerta"].value_counts()

    total      = len(ok)
    criticos   = counts.get("Crítico", 0)
    alto_risco = counts.get("Alto Risco", 0)
    atencao    = counts.get("Atenção", 0)
    normais    = counts.get("Normal", 0)
    erros      = len(df[df["Nível de Alerta"] == "N/D"])

    worst_ticker = ""
    if not ok.empty:
        worst = ok.loc[ok["M-Score"].idxmax()]
        worst_ticker = f" | Maior M-Score: **{worst['Ticker']}** ({worst['M-Score']:.4f})"

    return [
        "## Resumo Executivo",
        "",
        f"Empresas analisadas: **{total + erros}** "
        f"({erros} sem dados disponíveis)",
        "",
        "| 🔴 Crítico | 🟠 Alto Risco | 🟡 Atenção | 🟢 Normal |",
        "|---|---|---|---|",
        f"| **{criticos}** | **{alto_risco}** | **{atencao}** | **{normais}** |",
        "",
        f"> {worst_ticker.lstrip(' | ')}",
        "",
    ]


def _sector_table(df: pd.DataFrame) -> list[str]:
    ok = df[df["M-Score"].notna()]
    if ok.empty:
        return []

    sector_stats = (
        ok.groupby("Setor")
          .agg(
              Empresas=("Ticker", "count"),
              M_Score_Medio=("M-Score", "mean"),
              M_Score_Max=("M-Score", "max"),
              Criticos=("Nível de Alerta",
                        lambda x: (x == "Crítico").sum()),
          )
          .sort_values("M_Score_Medio", ascending=False)
          .reset_index()
    )

    lines = [
        "## Normalização Setorial",
        "",
        "| Setor | Empresas | M-Score Médio | M-Score Máx | Críticos |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for _, row in sector_stats.iterrows():
        indicator = " 🔴" if row["Criticos"] > 0 else ""
        lines.append(
            f"| {row['Setor']} | {int(row['Empresas'])} "
            f"| {row['M_Score_Medio']:+.4f} "
            f"| {row['M_Score_Max']:+.4f} "
            f"| {int(row['Criticos'])}{indicator} |"
        )
    lines.append("")
    return lines


def _summary_table(df: pd.DataFrame) -> list[str]:
    lines = [
        "| Ticker | Setor | M-Score | Accrual Ratio | Qualidade | Δ vs Setor |",
        "|---|---|:---:|:---:|:---:|:---:|",
    ]
    for _, row in df.iterrows():
        if row["Nível de Alerta"] == "N/D":
            continue
        delta = f"{row['Δ vs Setor']:+.4f}" if pd.notna(row["Δ vs Setor"]) else "—"
        lines.append(
            f"| **{row['Ticker']}** | {row['Setor']} "
            f"| `{row['M-Score']:+.4f}` "
            f"| {row['Accrual Ratio']:+.4f} "
            f"| {row['Qualidade Earnings']} "
            f"| {delta} |"
        )
    lines.append("")
    return lines


def _detail_cards(df: pd.DataFrame) -> list[str]:
    """Collapsible detail block per company with Beneish indices + red flags."""
    lines = []
    flag_cols = [c for c in df.columns if c.startswith("Red Flag")]

    for _, row in df.iterrows():
        if row["Nível de Alerta"] == "N/D":
            continue

        lines += [
            f"<details>",
            f"<summary><strong>{row['Ticker']}</strong> — "
            f"M-Score: <code>{row['M-Score']:+.4f}</code> | "
            f"Classificação: {row['Classificação']}</summary>",
            "",
            "**Índices Beneish:**",
            "",
            "| DSRI | GMI | AQI | SGI | DEPI | SGAI | LVGI | TATA |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
            (f"| {row['DSRI']:.4f} | {row['GMI']:.4f} | {row['AQI']:.4f} "
             f"| {row['SGI']:.4f} | {row['DEPI']:.4f} | {row['SGAI']:.4f} "
             f"| {row['LVGI']:.4f} | {row['TATA']:.4f} |"),
            "",
        ]

        flags = [str(row[c]) for c in flag_cols if str(row[c]).strip()]
        if flags:
            lines.append("**🚩 Red Flags:**")
            lines.append("")
            for flag in flags:
                lines.append(f"- {flag}")
            lines.append("")

        # Sector comparison
        if pd.notna(row["Δ vs Setor"]):
            direction = "acima" if row["Δ vs Setor"] > 0 else "abaixo"
            lines += [
                f"**Normalização Setorial ({row['Setor']}):** "
                f"M-Score médio do setor = `{row['M-Score Médio Setor']:+.4f}` — "
                f"esta empresa está **{abs(row['Δ vs Setor']):.4f} pontos {direction}** da média.",
                "",
            ]

        lines += ["</details>", ""]

    return lines


def _index_reference() -> list[str]:
    return [
        "---",
        "",
        "## Referência — Índices Beneish",
        "",
        "| Índice | Fórmula Resumida | Sinal de Alerta |",
        "|---|---|---|",
        "| **DSRI** | (Rec/Rev)ₜ ÷ (Rec/Rev)ₜ₋₁ | > 1.05 — recebíveis inflados |",
        "| **GMI**  | Margem Brutaₜ₋₁ ÷ Margem Brutaₜ | > 1.05 — margem em queda |",
        "| **AQI**  | (1 − Ativos Líquidos/Total)ₜ ÷ … | > 1.05 — ativos ocultos |",
        "| **SGI**  | Receitaₜ ÷ Receitaₜ₋₁ | > 1.10 — crescimento excessivo |",
        "| **DEPI** | Taxa Depreciação ₜ₋₁ ÷ Taxa ₜ | > 1.03 — D&A desacelerando |",
        "| **SGAI** | (SGA/Rev)ₜ ÷ (SGA/Rev)ₜ₋₁ | > 1.05 — custos G&A crescendo |",
        "| **LVGI** | Alavancagemₜ ÷ Alavancagemₜ₋₁ | > 1.05 — dívida aumentando |",
        "| **TATA** | (LucroLíquido − CFO) ÷ AtivoTotal | > 0.031 — accruals altos |",
        "",
        "> Modelo: M = −4.84 + 0.920·DSRI + 0.528·GMI + 0.404·AQI + 0.892·SGI "
        "+ 0.115·DEPI − 0.172·SGAI + 4.679·TATA − 0.327·LVGI",
        "> **Threshold: M > −1.78 → Potential Manipulator**",
        "",
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."
