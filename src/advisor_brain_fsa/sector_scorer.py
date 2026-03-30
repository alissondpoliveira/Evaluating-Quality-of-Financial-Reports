"""
Sector Risk Scorer — Beneish M-Score (non-financial companies only)
--------------------------------------------------------------------
Single strategy: BeneishSectorScorer applies the Beneish M-Score (1999)
combined with the CFA Level 2 Accrual Ratio quality metric to all
non-financial B3 companies (Indústria, Comércio, Serviços, Energia, etc.).

Banking and insurance scorers were removed because the Beneish M-Score
does not apply to financial institutions (different accounting structure).
The Home Dashboard excludes financial companies at the CVM registry level.

Usage
-----
    from advisor_brain_fsa.sector_scorer import get_scorer

    scorer = get_scorer("Energia")   # always returns BeneishSectorScorer
    result = scorer.score(current=fd_t, prior=fd_t1)
    print(result.alert_level, result.red_flags)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .accruals import AlertLevel, CashFlowQuality, CashFlowQualityResult
from .beneish_mscore import BeneishMScore, FinancialData, MScoreResult


# ---------------------------------------------------------------------------
# Unified result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SectorRiskResult:
    """
    Risk result returned by BeneishSectorScorer.

    Attributes
    ----------
    risk_score : float
        Normalised 0–10 score (higher = more risk).
    classification : str
        Human-readable Beneish classification.
    alert_level : AlertLevel
        Alert level enum (Normal / Atenção / Alto Risco / Crítico).
    metrics : dict
        Key metrics (m_score, accrual_ratio, dsri, gmi, aqi, …).
    red_flags : list[str]
        Top-3 human-readable risk flags.
    scorer_type : str
        Always "beneish".
    mscore_result : MScoreResult | None
    cfq_result : CashFlowQualityResult | None
    """
    risk_score:     float
    classification: str
    alert_level:    AlertLevel
    metrics:        Dict[str, float]
    red_flags:      List[str]
    scorer_type:    str
    mscore_result:  Optional[MScoreResult]          = None
    cfq_result:     Optional[CashFlowQualityResult] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SectorScorer(ABC):
    """Strategy interface — all scorers must implement score()."""

    @abstractmethod
    def score(self, current: FinancialData, prior: FinancialData) -> SectorRiskResult:
        ...

    @property
    @abstractmethod
    def scorer_type(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Single strategy: Beneish M-Score + CFA Accruals
# ---------------------------------------------------------------------------

class BeneishSectorScorer(SectorScorer):
    """
    Scorer for all non-financial companies.
    Uses the Beneish M-Score (1999) + CFA Level 2 Accrual Ratio.
    """

    @property
    def scorer_type(self) -> str:
        return "beneish"

    def score(self, current: FinancialData, prior: FinancialData) -> SectorRiskResult:
        from .rank_market import detect_red_flags  # avoid circular at module load

        ms    = BeneishMScore(current=current, prior=prior).calculate()
        cfq   = CashFlowQuality(current=current, prior=prior).calculate(ms)
        flags = detect_red_flags(ms)

        # Normalise M-Score [-5, +2] → risk_score [0, 10]
        risk_score = float(np.clip((ms.m_score + 5) / 7 * 10, 0, 10))

        return SectorRiskResult(
            risk_score=risk_score,
            classification=ms.classification,
            alert_level=cfq.alert_level,
            metrics={
                "m_score":       round(ms.m_score, 4),
                "accrual_ratio": round(cfq.accrual_ratio, 4),
                "dsri":  round(ms.dsri,  4), "gmi":  round(ms.gmi,  4),
                "aqi":   round(ms.aqi,   4), "sgi":  round(ms.sgi,  4),
                "depi":  round(ms.depi,  4), "sgai": round(ms.sgai, 4),
                "lvgi":  round(ms.lvgi,  4), "tata": round(ms.tata, 4),
            },
            red_flags=flags,
            scorer_type=self.scorer_type,
            mscore_result=ms,
            cfq_result=cfq,
        )


# ---------------------------------------------------------------------------
# Factory — single scorer for all non-financial sectors
# ---------------------------------------------------------------------------

def get_scorer(sector: str) -> SectorScorer:  # noqa: ARG001
    """
    Return BeneishSectorScorer for any sector.
    Financial companies are excluded upstream at the CVM registry level.
    """
    return BeneishSectorScorer()
