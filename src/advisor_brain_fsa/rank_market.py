"""
Market Ranking Orchestrator
-----------------------------
Runs the full risk-scoring pipeline across a list of B3 tickers using a
Strategy Pattern: each sector uses the appropriate risk scorer.

  Non-financial sectors -> BeneishSectorScorer (M-Score + CFA Accruals)
  Bancos / Financeiro   -> BankingScorer (ROA, cost-to-income, CFO quality)
  Seguros               -> InsuranceScorer (sinistralidade, indice combinado)

Delta vs Setor respects group boundaries (Tarefa 4):
  Financial companies are compared only against other financials,
  using the normalised risk_score (0-10).
  All other sectors compare M-Score within their own sector peer group.
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
from .sector_scorer import SectorRiskResult, get_scorer
from .ticker_map import FINANCIAL_GROUP, TICKER_SECTOR, get_sector, get_sector_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default watchlist
# ---------------------------------------------------------------------------

DEFAULT_WATCHLIST: List[str] = [
    "PETR4", "CSAN3", "PRIO3",
    "ITUB4", "BBDC4", "BBAS3", "BPAC11",
    "BBSE3", "PSSA3", "IRBR3",
    "VALE3", "GGBR4", "CSNA3",
    "ELET3", "CMIG4", "SBSP3", "EGIE3",
    "ABEV3", "JBSS3", "BRFS3", "LREN3",
    "WEGE3", "EMBR3", "SUZB3", "TOTVS3",
    "RDOR3", "RAIL3",
]

# ---------------------------------------------------------------------------
# Red-flag detection (Beneish indices only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _IndexSpec:
    attr: str
    coeff: float
    threshold: float
    label_tmpl: str


_INDEX_SPECS: List[_IndexSpec] = [
    _IndexSpec("dsri",  0.920,  1.031,
               "DSRI elevado ({v:.2f}) -- recebiveis crescem mais rapido que receita"),
    _IndexSpec("gmi",   0.528,  1.014,
               "GMI deteriorado ({v:.2f}) -- margem bruta em queda"),
    _IndexSpec("aqi",   0.404,  1.039,
               "AQI divergente ({v:.2f}) -- ativos intangiveis se expandindo"),
    _IndexSpec("sgi",   0.892,  1.134,
               "SGI acelerado ({v:.2f}) -- crescimento de vendas agressivo"),
    _IndexSpec("depi",  0.115,  1.017,
               "DEPI elevado ({v:.2f}) -- depreciacao sendo desacelerada"),
    _IndexSpec("tata",  4.679, -0.012,
               "TATA alto ({v:.4f}) -- accruals elevados vs. ativos totais"),
    _IndexSpec("sgai",  0.172,  1.054,
               "SGAI expandido ({v:.2f}) -- despesas G&A desproporcionais"),
    _IndexSpec("lvgi",  0.327,  1.000,
               "LVGI crescente ({v:.2f}) -- alavancagem financeira em alta"),
]


def detect_red_flags(result: MScoreResult, top_n: int = 3) -> List[str]:
    scored: list[tuple[float, str]] = []
    for spec in _INDEX_SPECS:
        value = getattr(result, spec.attr)
        deviation = max(0.0, value - spec.threshold)
        sv = spec.coeff * deviation
        if sv > 0:
            scored.append((sv, spec.label_tmpl.format(v=value)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in scored[:top_n]]


# ---------------------------------------------------------------------------
# Per-company result
# ---------------------------------------------------------------------------

@dataclass
class CompanyResult:
    ticker:      str
    sector:      str
    year_t:      int
    sector_risk: Optional[SectorRiskResult] = None
    error:       Optional[str] = None

    sector_avg_score: float = float("nan")
    score_vs_sector:  float = float("nan")

    @property
    def ok(self) -> bool:
        return self.error is None and self.sector_risk is not None

    @property
    def mscore(self) -> Optional[MScoreResult]:
        return self.sector_risk.mscore_result if self.sector_risk else None

    @property
    def cfq(self) -> Optional[CashFlowQualityResult]:
        return self.sector_risk.cfq_result if self.sector_risk else None

    @property
    def red_flags(self) -> List[str]:
        return self.sector_risk.red_flags if self.sector_risk else []


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def rank_market(
    tickers: Optional[List[str]] = None,
    year_t:  Optional[int] = None,
    year_t1: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    force_download: bool = False,
    top_flags: int = 3,
    retry_delay: float = 1.0,
    use_registry: bool = True,
) -> pd.DataFrame:
    """
    Run the full risk-scoring pipeline across a list of B3 tickers.

    Parameters
    ----------
    use_registry : bool
        When *True* (default), tickers absent from the static ticker_map are
        resolved via the CVM cadastral registry (Tarefa 4).  A network
        download of ``cad_cia_aberta.csv`` is triggered only if needed and not
        already cached.  Set to *False* to skip registry lookup (faster, but
        unknown tickers always get sector "Outros" / BeneishSectorScorer).
    """
    current_year = date.today().year
    year_t  = year_t  or (current_year - 1)
    year_t1 = year_t1 or (year_t - 1)
    tickers = tickers or DEFAULT_WATCHLIST

    # Tarefa 4 — lazy-load registry only when there is at least one unknown ticker
    _unknown = [t for t in tickers if t.upper() not in TICKER_SECTOR]
    _registry = None
    if use_registry and _unknown:
        try:
            from .cvm_registry import CVMRegistry  # noqa: PLC0415
            _registry = CVMRegistry.get_instance(cache_dir=cache_dir)
            logger.info(
                "CVMRegistry loaded (%d active companies). "
                "Resolving %d unknown ticker(s): %s",
                len(_registry.df),
                len(_unknown),
                ", ".join(_unknown),
            )
        except Exception as exc:
            logger.warning(
                "CVMRegistry unavailable (%s). Unknown tickers default to Outros.", exc
            )

    fetcher = CVMDataFetcher(cache_dir=cache_dir, force_download=force_download)
    results: List[CompanyResult] = []

    for i, ticker in enumerate(tickers):
        logger.info("[%d/%d] %s ...", i + 1, len(tickers), ticker)

        # --- Tarefa 4: resolve sector, falling back to registry ---
        if ticker.upper() in TICKER_SECTOR:
            sector = get_sector(ticker)
        elif _registry is not None:
            sector, _ = _registry.resolve_ticker_sector(ticker)
            logger.info("  Registry resolved %s → sector '%s'", ticker, sector)
        else:
            sector = "Outros"

        try:
            fd_t, fd_t1 = fetcher.get_financial_data(ticker, year_t=year_t, year_t1=year_t1)
            scorer = get_scorer(sector)
            sr = scorer.score(fd_t, fd_t1)
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, sector_risk=sr))
        except Exception as exc:
            logger.warning("  X %s -- %s", ticker, exc)
            results.append(CompanyResult(ticker=ticker, sector=sector,
                                         year_t=year_t, error=str(exc)))
        if i < len(tickers) - 1:
            time.sleep(retry_delay)

    _apply_sector_stats(results)
    return _to_dataframe(results, top_flags)


# ---------------------------------------------------------------------------
# Sector statistics -- group-aware (Tarefa 4)
# ---------------------------------------------------------------------------

def _apply_sector_stats(results: List[CompanyResult]) -> None:
    sector_mscores: Dict[str, List[float]] = {}
    fin_scores: List[float] = []

    for r in results:
        if not r.ok:
            continue
        if r.sector in FINANCIAL_GROUP:
            fin_scores.append(r.sector_risk.risk_score)
        elif r.sector_risk.mscore_result is not None:
            sector_mscores.setdefault(r.sector, []).append(
                r.sector_risk.mscore_result.m_score
            )

    sector_avg: Dict[str, float] = {
        s: float(np.mean(v)) for s, v in sector_mscores.items()
    }
    fin_avg = float(np.mean(fin_scores)) if fin_scores else float("nan")

    for r in results:
        if not r.ok:
            continue
        if r.sector in FINANCIAL_GROUP:
            r.sector_avg_score = fin_avg
            r.score_vs_sector  = r.sector_risk.risk_score - fin_avg
        elif r.sector_risk.mscore_result is not None:
            avg = sector_avg.get(r.sector, float("nan"))
            r.sector_avg_score = avg
            r.score_vs_sector  = r.sector_risk.mscore_result.m_score - avg


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def _to_dataframe(results: List[CompanyResult], top_flags: int) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.ok:
            sr = r.sector_risk
            ms = sr.mscore_result
            row = {
                "Ticker":             r.ticker,
                "Setor":              r.sector,
                "Ano":                r.year_t,
                "Scorer":             sr.scorer_type,
                "Score de Risco":     round(sr.risk_score, 4),
                "M-Score":            round(ms.m_score, 4) if ms else float("nan"),
                "Classificação":      sr.classification,
                "Nível de Alerta":    sr.alert_level.value,
                "_alerta_rank":       sr.alert_level.rank,
                "Accrual Ratio":      round(sr.cfq_result.accrual_ratio, 4) if sr.cfq_result else float("nan"),
                "Qualidade Earnings": sr.cfq_result.earnings_quality if sr.cfq_result else "N/A",
                "Score Médio Grupo":  round(r.sector_avg_score, 4),
                "Δ vs Grupo":     round(r.score_vs_sector, 4),
                "DSRI":  round(ms.dsri,  4) if ms else float("nan"),
                "GMI":   round(ms.gmi,   4) if ms else float("nan"),
                "AQI":   round(ms.aqi,   4) if ms else float("nan"),
                "SGI":   round(ms.sgi,   4) if ms else float("nan"),
                "DEPI":  round(ms.depi,  4) if ms else float("nan"),
                "SGAI":  round(ms.sgai,  4) if ms else float("nan"),
                "LVGI":  round(ms.lvgi,  4) if ms else float("nan"),
                "TATA":  round(ms.tata,  4) if ms else float("nan"),
                "ROA":              round(sr.metrics.get("roa",            float("nan")), 4),
                "Cost_Income":      round(sr.metrics.get("cost_income",    float("nan")), 4),
                "Indice_Combinado": round(sr.metrics.get("combined_ratio", float("nan")), 4),
                "Sinistralidade":   round(sr.metrics.get("loss_ratio",     float("nan")), 4),
                "Erro":             "",
            }
            flags = r.red_flags + [""] * top_flags
        else:
            row = {
                "Ticker": r.ticker, "Setor": r.sector, "Ano": r.year_t,
                "Scorer": "N/D", "Score de Risco": float("nan"),
                "M-Score": float("nan"), "Classificação": "N/D",
                "Nível de Alerta": "N/D", "_alerta_rank": -1,
                "Accrual Ratio": float("nan"), "Qualidade Earnings": "N/D",
                "Score Médio Grupo": float("nan"), "Δ vs Grupo": float("nan"),
                "DSRI": float("nan"), "GMI": float("nan"),
                "AQI": float("nan"),  "SGI": float("nan"),
                "DEPI": float("nan"), "SGAI": float("nan"),
                "LVGI": float("nan"), "TATA": float("nan"),
                "ROA": float("nan"), "Cost_Income": float("nan"),
                "Indice_Combinado": float("nan"), "Sinistralidade": float("nan"),
                "Erro": r.error or "",
            }
            flags = [""] * top_flags

        for idx in range(top_flags):
            row[f"Red Flag {idx + 1}"] = flags[idx] if idx < len(flags) else ""
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return (
        df.sort_values(["_alerta_rank", "Score de Risco"], ascending=[False, False])
          .drop(columns=["_alerta_rank"])
          .reset_index(drop=True)
    )
