"""
Market Ranking Orchestrator
-----------------------------
Runs the full Beneish M-Score + Cash Flow Quality pipeline across a list
of B3 tickers, groups results by sector, and produces a ranked DataFrame.

Usage
-----
    from advisor_brain_fsa.rank_market import rank_market

    # Rank the default watchlist (top 25 liquid tickers)
    df = rank_market(year_t=2023)

    # Rank a custom list
    df = rank_market(["PETR4", "VALE3", "ITUB4"], year_t=2023)

    # Save reports
    from advisor_brain_fsa.report_generator import generate_report
    generate_report(df, output_dir="reports/")

Red Flag Detection
------------------
Each Beneish index is evaluated against its empirical threshold derived from
Beneish (1999). The top-3 flags are selected by weighted M-Score contribution:

    contribution_i = |coefficient_i| × max(0, index_i − threshold_i)

This ensures the most impactful deviations appear first.

Beneish (1999) manipulator / non-manipulator mean values (Table 3):
  DSRI  1.465 / 1.031 │ GMI  1.193 / 1.014 │ AQI  1.254 / 1.039
  SGI   1.607 / 1.134 │ DEPI 1.077 / 1.017 │ TATA 0.031 / −0.012
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .accruals import AlertLevel, CashFlowQuality, CashFlowQualityResult
from .beneish_mscore import BeneishMScore, FinancialData, MScoreResult
from .data_fetcher import CVMDataFetcher
from .ticker_map import TICKER_SECTOR, get_sector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default watchlist — curated set of high-liquidity B3 names
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: List[str] = [
    # Energia
    "PETR4", "CSAN3", "PRIO3",
    # Bancos
    "ITUB4", "BBDC4", "BBAS3", "BPAC11",
    # Mineração
    "VALE3", "GGBR4", "CSNA3",
    # Utilidades
    "ELET3", "CMIG4", "SBSP3", "EGIE3",
    # Consumo & Alimentos
    "ABEV3", "JBSS3", "BRFS3", "LREN3",
    # Outros
    "EMBR3", "SUZB3", "TOTVS3", "RDOR3", "RAIL3",
]

# ---------------------------------------------------------------------------
# Red-flag index specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _IndexSpec:
    attr: str           # attribute on MScoreResult
    coeff: float        # Beneish coefficient (absolute value used for weight)
    threshold: float    # "non-manipulator mean" from Beneish (1999)
    label_tmpl: str     # human-readable description template (receives value)
    alarming_if: str    # "high" → flag when value > threshold (all positive-coeff indices)


_INDEX_SPECS: List[_IndexSpec] = [
    _IndexSpec("dsri",  0.920,  1.031,
               "DSRI elevado ({v:.2f}) — recebíveis crescem mais rápido que receita", "high"),
    _IndexSpec("gmi",   0.528,  1.014,
               "GMI deteriorado ({v:.2f}) — margem bruta em queda", "high"),
    _IndexSpec("aqi",   0.404,  1.039,
               "AQI divergente ({v:.2f}) — ativos intangíveis se expandindo", "high"),
    _IndexSpec("sgi",   0.892,  1.134,
               "SGI acelerado ({v:.2f}) — crescimento de vendas agressivo", "high"),
    _IndexSpec("depi",  0.115,  1.017,
               "DEPI elevado ({v:.2f}) — depreciação sendo desacelerada", "high"),
    _IndexSpec("tata",  4.679,  -0.012,
               "TATA alto ({v:.4f}) — accruals elevados vs. ativos totais", "high"),
    # SGAI and LVGI have negative coefficients in the model; operationally
    # still flagged if materially above 1.0 (deteriorating cost structure).
    _IndexSpec("sgai",  0.172,  1.054,
               "SGAI expandido ({v:.2f}) — despesas G&A desproporcionais", "high"),
    _IndexSpec("lvgi",  0.327,  1.000,
               "LVGI crescente ({v:.2f}) — alavancagem financeira em alta", "high"),
]


def detect_red_flags(result: MScoreResult, top_n: int = 3) -> List[str]:
    """
    Identify the top-N most impactful Beneish index deviations.

    Score per flag = |coefficient| × max(0, value − non_manipulator_mean).
    Sorted descending so the most M-Score-relevant flag appears first.

    Parameters
    ----------
    result : MScoreResult
    top_n : int
        Number of flags to return (default 3).

    Returns
    -------
    list[str]
        Human-readable flag descriptions, e.g.
        ["DSRI elevado (1.82) — recebíveis crescem mais rápido que receita", ...]
    """
    scored: list[tuple[float, str]] = []
    for spec in _INDEX_SPECS:
        value = getattr(result, spec.attr)
        deviation = max(0.0, value - spec.threshold)
        score = spec.coeff * deviation
        if score > 0:
            label = spec.label_tmpl.format(v=value)
            scored.append((score, label))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in scored[:top_n]]


# ---------------------------------------------------------------------------
# Per-company result
# ---------------------------------------------------------------------------

@dataclass
class CompanyResult:
    ticker: str
    sector: str
    year_t: int
    mscore: MScoreResult
    cfq: CashFlowQualityResult
    red_flags: List[str]
    error: Optional[str] = None     # set if fetch/calculation failed

    # Filled after sector grouping
    sector_avg_mscore: float = float("nan")
    mscore_vs_sector:  float = float("nan")

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def rank_market(
    tickers: Optional[List[str]] = None,
    year_t: Optional[int] = None,
    year_t1: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    force_download: bool = False,
    top_flags: int = 3,
    retry_delay: float = 1.0,
) -> pd.DataFrame:
    """
    Run the full pipeline for a list of tickers and return a ranked DataFrame.

    Parameters
    ----------
    tickers : list[str] | None
        B3 tickers to analyse. Defaults to DEFAULT_WATCHLIST.
    year_t : int | None
        Current fiscal year. Defaults to last completed calendar year.
    year_t1 : int | None
        Prior fiscal year. Defaults to year_t − 1.
    cache_dir : Path | str | None
        Local directory for CVM ZIP cache.
    force_download : bool
        Skip cache and re-download ZIPs.
    top_flags : int
        Number of red flags to report per company (default 3).
    retry_delay : float
        Seconds to wait between requests to avoid rate limiting.

    Returns
    -------
    pd.DataFrame
        Columns: Ticker, Setor, Ano, M-Score, Classificação, Nível de Alerta,
        Accrual Ratio, Qualidade, M-Score Médio Setor, Δ vs Setor,
        DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA,
        Red Flag 1, Red Flag 2, Red Flag 3, Erro
        Sorted by: Nível de Alerta (desc) → M-Score (desc).
    """
    current_year = date.today().year
    year_t  = year_t  or (current_year - 1)
    year_t1 = year_t1 or (year_t - 1)
    tickers = tickers or DEFAULT_WATCHLIST

    fetcher = CVMDataFetcher(cache_dir=cache_dir, force_download=force_download)

    results: List[CompanyResult] = []

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] Processing %s …", i + 1, len(tickers), ticker)
        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t1)
            mscore = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
            cfq    = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore)
            flags  = detect_red_flags(mscore, top_n=top_flags)

            results.append(CompanyResult(
                ticker=ticker,
                sector=get_sector(ticker),
                year_t=year_t,
                mscore=mscore,
                cfq=cfq,
                red_flags=flags,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ✗ %s — %s", ticker, exc)
            results.append(CompanyResult(
                ticker=ticker,
                sector=get_sector(ticker),
                year_t=year_t,
                mscore=None,   # type: ignore[arg-type]
                cfq=None,      # type: ignore[arg-type]
                red_flags=[],
                error=str(exc),
            ))

        if i < len(tickers) - 1:
            time.sleep(retry_delay)

    # Sector normalisation — compute mean M-Score per sector (successes only)
    _apply_sector_stats(results)

    return _to_dataframe(results, top_flags)


# ---------------------------------------------------------------------------
# Sector statistics
# ---------------------------------------------------------------------------

def _apply_sector_stats(results: List[CompanyResult]) -> None:
    """Mutate results in-place, adding sector_avg_mscore and mscore_vs_sector."""
    sector_scores: Dict[str, list[float]] = {}
    for r in results:
        if r.ok:
            sector_scores.setdefault(r.sector, []).append(r.mscore.m_score)

    sector_avg: Dict[str, float] = {
        s: float(np.mean(scores)) for s, scores in sector_scores.items()
    }

    for r in results:
        if r.ok:
            avg = sector_avg.get(r.sector, float("nan"))
            r.sector_avg_mscore = avg
            r.mscore_vs_sector  = r.mscore.m_score - avg


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _to_dataframe(results: List[CompanyResult], top_flags: int) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.ok:
            flags = r.red_flags + [""] * top_flags          # pad to top_flags cols
            row = {
                "Ticker":             r.ticker,
                "Setor":              r.sector,
                "Ano":                r.year_t,
                "M-Score":            round(r.mscore.m_score, 4),
                "Classificação":      r.mscore.classification,
                "Nível de Alerta":    r.cfq.alert_level.value,
                "_alerta_rank":       r.cfq.alert_level.rank,
                "Accrual Ratio":      round(r.cfq.accrual_ratio, 4),
                "Qualidade Earnings": r.cfq.earnings_quality,
                "M-Score Médio Setor":round(r.sector_avg_mscore, 4),
                "Δ vs Setor":         round(r.mscore_vs_sector, 4),
                "DSRI":  round(r.mscore.dsri, 4),
                "GMI":   round(r.mscore.gmi,  4),
                "AQI":   round(r.mscore.aqi,  4),
                "SGI":   round(r.mscore.sgi,  4),
                "DEPI":  round(r.mscore.depi, 4),
                "SGAI":  round(r.mscore.sgai, 4),
                "LVGI":  round(r.mscore.lvgi, 4),
                "TATA":  round(r.mscore.tata, 4),
                "Erro":  "",
            }
        else:
            row = {
                "Ticker":              r.ticker,
                "Setor":               r.sector,
                "Ano":                 r.year_t,
                "M-Score":             float("nan"),
                "Classificação":       "N/D",
                "Nível de Alerta":     "N/D",
                "_alerta_rank":        -1,
                "Accrual Ratio":       float("nan"),
                "Qualidade Earnings":  "N/D",
                "M-Score Médio Setor": float("nan"),
                "Δ vs Setor":          float("nan"),
                "DSRI": float("nan"), "GMI":  float("nan"),
                "AQI":  float("nan"), "SGI":  float("nan"),
                "DEPI": float("nan"), "SGAI": float("nan"),
                "LVGI": float("nan"), "TATA": float("nan"),
                "Erro": r.error or "",
            }
            flags = [""] * top_flags

        for idx in range(top_flags):
            row[f"Red Flag {idx + 1}"] = flags[idx] if idx < len(flags) else ""

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort: alert severity desc → M-Score desc (more positive = worse)
    df = (
        df.sort_values(["_alerta_rank", "M-Score"], ascending=[False, False])
          .drop(columns=["_alerta_rank"])
          .reset_index(drop=True)
    )
    return df
