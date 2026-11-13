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

A régua acima, com sinal, é a do cálculo de uma empresa isolada. Com o mercado inteiro
(site estático), a qualidade usa o desvio em módulo em relação à mediana do setor:
ver medianas_por_setor / qualidade_por_desvio.

References
----------
- Dechow, P. M., Sloan, R. G. & Sweeney, A. P. (1995). Detecting earnings management.
  The Accounting Review, 70(2), 193–225 (Industry Model).
- Becker, C. L., DeFond, M. L., Jiambalvo, J. & Subramanyam, K. R. (1998). The effect of
  audit quality on earnings management. Contemporary Accounting Research, 15(1), 1–24.
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
    return combinar_alerta(mscore_result.is_manipulator, earnings_quality)


# ---------------------------------------------------------------------------
# Accrual relativo ao setor (modelo de setor: Dechow e Sloan, 1991;
# Dechow, Sloan e Sweeney, 1995)
# ---------------------------------------------------------------------------
# O accrual "normal" de uma empresa não é zero: a depreciação reduz o lucro sem sair do
# caixa, e o capital de giro de cada setor tem a sua dinâmica. O modelo de setor estima o
# accrual normal pela mediana dos accruals totais (sobre o ativo) das empresas do mesmo
# setor no mesmo período; o que sobra é o accrual discricionário. A qualidade se lê pelo
# módulo dessa sobra (Warfield, Wild e Wild, 1995; Becker et al., 1998): lucro muito acima
# do caixa (agressivo) e muito abaixo (big bath, constituição de reservas) contam igual.

DESVIO_MODERADO = 0.05   # a partir daqui → "Moderada"
DESVIO_ALTO     = 0.10   # a partir daqui → "Baixa"
MIN_SETOR       = 10     # setor menor usa a mediana da amostra toda


def medianas_por_setor(pares, minimo: int = MIN_SETOR) -> tuple[dict, float, dict]:
    """pares: (setor, accrual_ratio) das empresas de referência (dado completo).
    Devolve ({setor: mediana} só dos setores com pelo menos `minimo` empresas,
    mediana da amostra toda, {setor: n})."""
    import statistics

    grupos: dict[str, list[float]] = {}
    for setor, r in pares:
        if r is not None:
            grupos.setdefault(setor, []).append(float(r))
    todos = [r for v in grupos.values() for r in v]
    geral = statistics.median(todos) if todos else 0.0
    return ({s: statistics.median(v) for s, v in grupos.items() if len(v) >= minimo}, geral,
            {s: len(v) for s, v in grupos.items()})


def qualidade_por_desvio(desvio: float) -> str:
    d = abs(desvio)
    if d < DESVIO_MODERADO:
        return "Alta"
    if d < DESVIO_ALTO:
        return "Moderada"
    return "Baixa"


def combinar_alerta(is_manipulator: bool, earnings_quality: str) -> AlertLevel:
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
