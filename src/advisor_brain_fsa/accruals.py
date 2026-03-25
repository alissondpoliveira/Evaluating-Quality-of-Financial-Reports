"""
Cash Flow Quality — Accruals Analysis
--------------------------------------
Implements the CFA Level 2 accruals-based earnings quality metric and
combines it with the Beneish M-Score to produce a composite alert level.

Theory (CFA Level 2 — Financial Reporting Quality)
----------------------------------------------------
High-quality earnings are primarily driven by cash flows, not accruals.
The Cash Flow Accruals Ratio isolates the non-cash component of earnings:

    Accrual Ratio = (Net Income - Cash From Operations) / Avg Total Assets

Interpretation:
  ·  < 0.01  → Alta qualidade: lucro é basicamente caixa
  ·  0.01–0.05 → Moderada: accruals presentes mas aceitáveis
  ·  > 0.05  → Baixa: componente accrual dominante; risco de reversão

Combined Alert Level (M-Score × Accruals)
------------------------------------------
  M-Score > -1.78  AND  Accruals > 5%  → "Crítico"   ★★★
  M-Score > -1.78  AND  Accruals ≤ 5%  → "Alto Risco" ★★
  M-Score ≤ -1.78  AND  Accruals > 5%  → "Atenção"   ★
  M-Score ≤ -1.78  AND  Accruals ≤ 5%  → "Normal"    ✓

References
----------
- CFA Program Curriculum, Level 2 — Financial Reporting Quality
- Sloan, R. G. (1996). Do stock prices fully reflect information in accruals
  and cash flows about future earnings? The Accounting Review, 71(3), 289–315.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .beneish_mscore import FinancialData, MScoreResult, _safe_div


# ---------------------------------------------------------------------------
# Alert level
# ---------------------------------------------------------------------------

class AlertLevel(str, Enum):
    """Composite risk alert combining M-Score and accruals quality."""
    NORMAL    = "Normal"      # Non-manipulator + high earnings quality
    WATCH     = "Atenção"     # Non-manipulator BUT high accruals
    HIGH_RISK = "Alto Risco"  # Potential manipulator, quality acceptable
    CRITICAL  = "Crítico"     # Potential manipulator + high accruals

    @property
    def rank(self) -> int:
        """Numeric sort rank: higher = more severe."""
        return {
            AlertLevel.NORMAL:    0,
            AlertLevel.WATCH:     1,
            AlertLevel.HIGH_RISK: 2,
            AlertLevel.CRITICAL:  3,
        }[self]

    @property
    def stars(self) -> str:
        return {
            AlertLevel.NORMAL:    "✓",
            AlertLevel.WATCH:     "★",
            AlertLevel.HIGH_RISK: "★★",
            AlertLevel.CRITICAL:  "★★★",
        }[self]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CashFlowQualityResult:
    """Output of CashFlowQuality.calculate()."""

    accrual_ratio: float
    """(Net Income − CFO) / Average Total Assets."""

    earnings_quality: str
    """'Alta' | 'Moderada' | 'Baixa'"""

    alert_level: AlertLevel
    """Composite alert combining M-Score and accruals."""

    def to_dict(self) -> dict:
        return {
            "Accrual Ratio":    round(self.accrual_ratio, 6),
            "Earnings Quality": self.earnings_quality,
            "Alert Level":      self.alert_level.value,
        }

    def __str__(self) -> str:
        return (
            f"CashFlowQuality | Accrual Ratio: {self.accrual_ratio:+.4f} "
            f"| Qualidade: {self.earnings_quality} "
            f"| Alerta: {self.alert_level.value} {self.alert_level.stars}"
        )


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_ACCRUAL_WARN  = 0.01   # above this → "Moderada"
_ACCRUAL_HIGH  = 0.05   # above this → "Baixa" (high-accruals zone)


def _earnings_quality_label(ratio: float) -> str:
    if ratio < _ACCRUAL_WARN:
        return "Alta"
    if ratio < _ACCRUAL_HIGH:
        return "Moderada"
    return "Baixa"


def _combine_alert(mscore_result: MScoreResult, earnings_quality: str) -> AlertLevel:
    is_manipulator = mscore_result.is_manipulator
    low_quality    = earnings_quality == "Baixa"

    if is_manipulator and low_quality:
        return AlertLevel.CRITICAL
    if is_manipulator:
        return AlertLevel.HIGH_RISK
    if low_quality:
        return AlertLevel.WATCH
    return AlertLevel.NORMAL


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class CashFlowQuality:
    """
    Computes the CFA accruals-based earnings quality metric and combines
    it with a pre-computed Beneish M-Score result.

    Parameters
    ----------
    current : FinancialData
        Current period (T) accounting data.
    prior : FinancialData
        Prior period (T-1) accounting data.

    Example
    -------
    >>> from advisor_brain_fsa import BeneishMScore
    >>> from advisor_brain_fsa.accruals import CashFlowQuality
    >>> mscore = BeneishMScore(current=data_t, prior=data_t1).calculate()
    >>> cfq    = CashFlowQuality(current=data_t, prior=data_t1).calculate(mscore)
    >>> print(cfq)
    """

    def __init__(self, current: FinancialData, prior: FinancialData) -> None:
        self._t  = current
        self._t1 = prior

    def calculate(self, mscore_result: MScoreResult) -> CashFlowQualityResult:
        """
        Compute the accrual ratio and derive the composite alert level.

        Parameters
        ----------
        mscore_result : MScoreResult
            Already-calculated Beneish M-Score for the same period pair.

        Returns
        -------
        CashFlowQualityResult
        """
        avg_assets = (self._t.total_assets + self._t1.total_assets) / 2.0
        accruals   = self._t.net_income - self._t.cash_from_operations
        ratio      = _safe_div(accruals, avg_assets, fallback=0.0)

        quality = _earnings_quality_label(ratio)
        alert   = _combine_alert(mscore_result, quality)

        return CashFlowQualityResult(
            accrual_ratio   = ratio,
            earnings_quality= quality,
            alert_level     = alert,
        )
