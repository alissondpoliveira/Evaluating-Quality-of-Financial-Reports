"""
Advisor-Brain-FSA — CLI Entry Point
=====================================
Ferramenta de análise de qualidade de relatórios financeiros baseada em:
  · Beneish M-Score (Beneish, 1999)
  · CFA Level 2 — Evaluating Quality of Financial Reports

Uso
---
  # Demo com dados sintéticos (comportamento original)
  python main.py

  # Análise de um ticker via dados reais da CVM (sem IA)
  python main.py --analyze PETR4

  # Análise completa com narrativa gerada pelo Claude (requer ANTHROPIC_API_KEY)
  python main.py --analyze PETR4 --ai

  # Ano específico
  python main.py --analyze PETR4 --year 2022 --ai

  # Salvar relatório em diretório
  python main.py --analyze PETR4 --ai --output reports/

  # Ranking em lote de múltiplos tickers
  python main.py --rank PETR4 VALE3 ITUB4 ABEV3

  # Ranking do watchlist padrão
  python main.py --rank --all --output reports/
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd

from advisor_brain_fsa import BeneishMScore
from advisor_brain_fsa.accruals import CashFlowQuality
from advisor_brain_fsa.beneish_mscore import FinancialData
from advisor_brain_fsa.mda_analyst import MDAnalyst, compute_grade
from advisor_brain_fsa.rank_market import DEFAULT_WATCHLIST, detect_red_flags, rank_market
from advisor_brain_fsa.report_generator import build_markdown, generate_report
from advisor_brain_fsa.ticker_map import get_sector


# ============================================================================
# Demo data (kept from v0.1)
# ============================================================================

_SAFECO_T = FinancialData(
    revenues=1_200_000, cost_of_goods_sold=720_000,
    sales_general_admin_expenses=120_000, receivables=100_000,
    total_assets=900_000, current_assets=300_000, pp_and_e=400_000,
    securities=50_000, total_long_term_debt=180_000, current_liabilities=90_000,
    depreciation=40_000, net_income=120_000, cash_from_operations=150_000,
)
_SAFECO_T1 = FinancialData(
    revenues=1_000_000, cost_of_goods_sold=600_000,
    sales_general_admin_expenses=100_000, receivables=80_000,
    total_assets=800_000, current_assets=260_000, pp_and_e=360_000,
    securities=40_000, total_long_term_debt=160_000, current_liabilities=80_000,
    depreciation=36_000, net_income=100_000, cash_from_operations=130_000,
)

_RISKY_T = FinancialData(
    revenues=1_500_000, cost_of_goods_sold=1_200_000,
    sales_general_admin_expenses=300_000, receivables=400_000,
    total_assets=1_000_000, current_assets=200_000, pp_and_e=200_000,
    securities=20_000, total_long_term_debt=500_000, current_liabilities=200_000,
    depreciation=10_000, net_income=200_000, cash_from_operations=20_000,
)
_RISKY_T1 = FinancialData(
    revenues=1_000_000, cost_of_goods_sold=700_000,
    sales_general_admin_expenses=150_000, receivables=100_000,
    total_assets=800_000, current_assets=250_000, pp_and_e=280_000,
    securities=30_000, total_long_term_debt=200_000, current_liabilities=100_000,
    depreciation=30_000, net_income=80_000, cash_from_operations=100_000,
)


# ============================================================================
# Command implementations
# ============================================================================

def cmd_demo() -> None:
    """Original demo with synthetic SafeCo vs RiskyInc scenarios."""
    print("\n" + "=" * 60)
    print("     ADVISOR-BRAIN-FSA | Beneish M-Score Demo")
    print("=" * 60)

    scenarios = [
        ("SafeCo (Non-Manipulator expected)",     _SAFECO_T, _SAFECO_T1),
        ("RiskyInc (Potential Manipulator expected)", _RISKY_T, _RISKY_T1),
    ]

    results = []
    for name, t, t1 in scenarios:
        print(f"\n{'#' * 60}\n  Company: {name}\n{'#' * 60}")
        mscore = BeneishMScore(current=t, prior=t1).calculate()
        cfq    = CashFlowQuality(current=t, prior=t1).calculate(mscore)
        flags  = detect_red_flags(mscore)

        print(mscore)
        print(f"\n  CashFlowQuality: {cfq}")
        if flags:
            print("\n  Red Flags:")
            for f in flags:
                print(f"    • {f}")

        grade, label = compute_grade(mscore.m_score, cfq.accrual_ratio)
        print(f"\n  Nota Qualitativa: {grade} — {label}")

        row = mscore.to_dict()
        row["Company"] = name
        row["Accrual Ratio"] = round(cfq.accrual_ratio, 4)
        row["Grade"] = grade
        results.append(row)

    df = pd.DataFrame(results).set_index("Company")
    print("\n\n  Comparative Summary Table")
    print("-" * 60)
    print(df[["M-Score", "Classification", "Accrual Ratio", "Grade"]].to_string())
    print()


def cmd_analyze(
    ticker: str,
    year_t: int,
    year_t1: int,
    use_ai: bool,
    output_dir: str | None,
    api_key: str | None,
) -> None:
    """Single-ticker analysis: fetch CVM data → M-Score → AI report."""
    from advisor_brain_fsa.data_fetcher import fetch_data

    sector = get_sector(ticker)
    print(f"\n{'=' * 60}")
    print(f"  ADVISOR-BRAIN-FSA | Análise: {ticker}")
    print(f"  Setor: {sector} | Ano: {year_t} vs {year_t1}")
    print(f"{'=' * 60}\n")

    # ── 1. Fetch data ────────────────────────────────────────────────
    print(f"[1/4] Baixando dados da CVM para {ticker} ({year_t} e {year_t1})…")
    try:
        fd_t, fd_t1 = fetch_data(ticker, year_t=year_t, year_t1=year_t1)
    except Exception as exc:
        print(f"\n✗ Erro ao buscar dados: {exc}")
        print("  Dica: verifique se o ticker está no ticker_map ou use --cnpj / nome da empresa.")
        sys.exit(1)
    print("  ✓ Dados carregados com sucesso.\n")

    # ── 2. Beneish M-Score ──────────────────────────────────────────
    print("[2/4] Calculando Beneish M-Score…")
    mscore = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
    print(mscore)

    # ── 3. Cash Flow Quality ─────────────────────────────────────────
    print("\n[3/4] Calculando CFA Accruals Quality…")
    cfq   = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore)
    flags = detect_red_flags(mscore)
    grade, grade_label = compute_grade(mscore.m_score, cfq.accrual_ratio)
    print(f"  {cfq}")
    print(f"  Nota Qualitativa: {grade} — {grade_label}")
    if flags:
        print("\n  Red Flags Detectados:")
        for f in flags:
            print(f"    🚩 {f}")

    # ── 4. AI narrative ──────────────────────────────────────────────
    if use_ai:
        _cmd_ai_narrative(
            ticker=ticker, sector=sector, year=year_t,
            mscore=mscore, cfq=cfq, flags=flags,
            api_key=api_key, output_dir=output_dir,
        )
    elif output_dir:
        # Local Markdown report without AI
        _save_local_report(ticker, year_t, mscore, cfq, flags, output_dir)


def _cmd_ai_narrative(
    ticker, sector, year, mscore, cfq, flags,
    api_key, output_dir,
):
    print("\n[4/4] Gerando Tese de Risco com Claude (streaming)…\n")
    print("─" * 60)

    analyst = MDAnalyst(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""))

    full_report = []
    try:
        for chunk in analyst.analyze_streaming(
            ticker=ticker, sector=sector, year=year,
            mscore_result=mscore, cfq_result=cfq, red_flags=flags,
        ):
            print(chunk, end="", flush=True)
            full_report.append(chunk)
    except ValueError as exc:
        print(f"\n✗ {exc}")
        print("  Configure ANTHROPIC_API_KEY=<sua_chave> e tente novamente.")
        sys.exit(1)
    except ImportError as exc:
        print(f"\n✗ {exc}")
        sys.exit(1)

    print("\n" + "─" * 60)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        fname = out / f"tese_risco_{ticker}_{year}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        fname.write_text("".join(full_report), encoding="utf-8")
        print(f"\n  ✓ Relatório salvo em: {fname}")


def _save_local_report(ticker, year, mscore, cfq, flags, output_dir):
    """Save a local Markdown report using report_generator (no AI)."""
    import numpy as np
    from advisor_brain_fsa.rank_market import (
        CompanyResult, _apply_sector_stats, _to_dataframe
    )
    from advisor_brain_fsa.accruals import AlertLevel, CashFlowQualityResult

    sector = get_sector(ticker)
    result = CompanyResult(
        ticker=ticker, sector=sector, year_t=year,
        mscore=mscore, cfq=cfq, red_flags=flags,
    )
    _apply_sector_stats([result])
    df = _to_dataframe([result], top_flags=3)
    md_path, csv_path = generate_report(df, output_dir=output_dir, year_t=year)
    print(f"\n  ✓ Relatório Markdown: {md_path}")
    print(f"  ✓ CSV: {csv_path}")


def cmd_rank(
    tickers: list[str],
    year_t: int,
    year_t1: int,
    output_dir: str | None,
) -> None:
    """Batch ranking of multiple tickers."""
    print(f"\n{'=' * 60}")
    print(f"  ADVISOR-BRAIN-FSA | Ranking de Mercado")
    print(f"  Tickers: {', '.join(tickers)} | Ano: {year_t}")
    print(f"{'=' * 60}\n")
    print(f"  Processando {len(tickers)} empresa(s)… (isso pode levar alguns minutos)\n")

    df = rank_market(
        tickers=tickers,
        year_t=year_t,
        year_t1=year_t1,
        retry_delay=0.5,
    )

    # Print summary table
    display_cols = [
        "Ticker", "Setor", "M-Score", "Nível de Alerta",
        "Accrual Ratio", "Δ vs Setor",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))
    print()

    if output_dir:
        md_path, csv_path = generate_report(df, output_dir=output_dir, year_t=year_t)
        print(f"\n  ✓ Relatório Markdown: {md_path}")
        print(f"  ✓ CSV exportado: {csv_path}")


# ============================================================================
# CLI argument parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Advisor-Brain-FSA — Análise de Qualidade de Relatórios Financeiros\n"
            "Modelo: Beneish M-Score + CFA Level 2 Accruals Quality"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Exemplos:
              python main.py                           # demo sintético
              python main.py --analyze PETR4           # análise local (sem IA)
              python main.py --analyze PETR4 --ai      # análise com Claude
              python main.py --analyze PETR4 --year 2022 --ai --output reports/
              python main.py --rank PETR4 VALE3 ITUB4
              python main.py --rank --all --output reports/
        """),
    )

    # ── Analyze mode ───────────────────────────────────────────────────────
    parser.add_argument(
        "--analyze", metavar="TICKER",
        help="Ticker, CNPJ ou nome da empresa para análise individual.",
    )
    parser.add_argument(
        "--ai", action="store_true",
        help="Gerar narrativa de Tese de Risco via Claude API (requer ANTHROPIC_API_KEY).",
    )
    parser.add_argument(
        "--api-key", metavar="KEY",
        help="Chave da API Anthropic (alternativa à variável ANTHROPIC_API_KEY).",
    )

    # ── Rank mode ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--rank", nargs="*", metavar="TICKER",
        help=(
            "Ranking em lote. Passe tickers explícitos ou use --all "
            "para o watchlist padrão (%(default)s tickers)."
        ),
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Usar o watchlist padrão completo no modo --rank.",
    )

    # ── Shared options ─────────────────────────────────────────────────────
    parser.add_argument(
        "--year", type=int, default=None,
        help="Ano fiscal corrente (padrão: ano anterior).",
    )
    parser.add_argument(
        "--output", metavar="DIR",
        help="Diretório para salvar relatório .md e .csv.",
    )

    return parser


# Lazy import to avoid breaking demo when argparse not needed
try:
    import textwrap as textwrap   # already in stdlib
except ImportError:
    pass


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    import textwrap

    parser = build_parser()
    args = parser.parse_args()

    current_year = date.today().year
    year_t  = args.year or (current_year - 1)
    year_t1 = year_t - 1

    if args.analyze:
        cmd_analyze(
            ticker=args.analyze,
            year_t=year_t,
            year_t1=year_t1,
            use_ai=args.ai,
            output_dir=args.output,
            api_key=args.api_key,
        )

    elif args.rank is not None or args.all:
        if args.all or (args.rank is not None and len(args.rank) == 0):
            tickers = DEFAULT_WATCHLIST
        else:
            tickers = args.rank
        cmd_rank(
            tickers=tickers,
            year_t=year_t,
            year_t1=year_t1,
            output_dir=args.output,
        )

    else:
        # No arguments → run original demo
        cmd_demo()


if __name__ == "__main__":
    main()
