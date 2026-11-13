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

Interpretation (desvio = accrual ratio − mediana do setor, em módulo):
  ·  |desvio| < 0.05       → Alta: accrual perto do normal do setor
  ·  0.05 ≤ |desvio| < 0.10 → Moderada
  ·  |desvio| ≥ 0.10       → Baixa: lucro muito acima do caixa (agressivo) ou muito
                              abaixo (big bath, constituição de reservas)

O accrual normal não é zero (a depreciação reduz o lucro sem sair do caixa); a mediana do
setor vem do último lote do site (site/public/dados.json, "medianas_accrual"). Sem esse
arquivo, ou sem o setor nele, usa a mediana da amostra toda, e por último MEDIANA_PADRAO.

Combined Alert Level (M-Score × Accruals)
------------------------------------------
  M-Score > -1.78  AND  qualidade Baixa  → "Crítico"   ★★★
  M-Score > -1.78  AND  outra            → "Alto Risco" ★★
  M-Score ≤ -1.78  AND  qualidade Baixa  → "Atenção"   ★
  M-Score ≤ -1.78  AND  outra            → "Normal"    ✓

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

import json
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

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

    mediana_setor: float = 0.0
    """Accrual ratio normal (mediana do setor) usado como referência."""

    desvio: float = 0.0
    """accrual_ratio − mediana_setor; a qualidade se lê pelo módulo."""

    referencia: str = ""
    """De onde veio a mediana: nome do setor, 'amostra toda' ou 'padrão'."""

    def to_dict(self) -> dict:
        return {
            "Accrual Ratio":    round(self.accrual_ratio, 6),
            "Mediana do Setor": round(self.mediana_setor, 6),
            "Desvio":           round(self.desvio, 6),
            "Earnings Quality": self.earnings_quality,
            "Alert Level":      self.alert_level.value,
        }

    def __str__(self) -> str:
        return (
            f"CashFlowQuality | Accrual Ratio: {self.accrual_ratio:+.4f} "
            f"| Mediana ({self.referencia or 'setor'}): {self.mediana_setor:+.4f} "
            f"| Desvio: {self.desvio:+.4f} "
            f"| Qualidade: {self.earnings_quality} "
            f"| Alerta: {self.alert_level.value} {self.alert_level.stars}"
        )


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
MEDIANA_PADRAO  = -0.047 # mediana do mercado (não financeiras, 12 meses até jun/2026): último recurso

# medianas publicadas pelo último lote do site; ACCRUAL_MEDIANAS aponta outro arquivo
_ARQUIVO_MEDIANAS = Path(os.getenv("ACCRUAL_MEDIANAS") or
                         Path(__file__).resolve().parents[2] / "site" / "public" / "dados.json")


@lru_cache(maxsize=4)
def _medianas_publicadas(caminho: str) -> tuple[dict, dict]:
    """(medianas_accrual, {ticker: (mediana, grupo)}) do último lote publicado."""
    try:
        d = json.loads(Path(caminho).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}
    por_ticker = {}
    for e in d.get("empresas") or []:
        a = e.get("accrual_setor")
        if a and e.get("ticker"):
            por_ticker[str(e["ticker"]).upper()] = (float(a["mediana"]), a.get("grupo") or "amostra toda")
    return d.get("medianas_accrual") or {}, por_ticker


def referencia_accrual(setor: str | None = None, ticker: str | None = None) -> tuple[float, str]:
    """(accrual ratio normal, origem) para uma empresa isolada, do último lote do mercado:
    1) a empresa está no lote → a mesma mediana usada no site (setor do cadastro da CVM);
    2) o setor informado tem mediana publicada → ela; 3) mediana da amostra toda;
    4) sem arquivo → MEDIANA_PADRAO."""
    m, por_ticker = _medianas_publicadas(str(_ARQUIVO_MEDIANAS))
    if ticker:
        t = ticker.upper().strip()
        if t in por_ticker:
            return por_ticker[t]
        raiz = [v for k, v in por_ticker.items() if k[:4] == t[:4]]
        if len(t) >= 4 and raiz and len({v for v in raiz}) == 1:
            return raiz[0]
    setores = m.get("setores") or {}
    if setor and setor in setores:
        return float(setores[setor]), setor
    if m.get("geral") is not None:
        return float(m["geral"]), "amostra toda"
    return MEDIANA_PADRAO, "padrão"


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
    setor, ticker : str, optional
        Para buscar a mediana publicada (referencia_accrual); o ticker tem precedência.
    mediana : float, optional
        Accrual ratio normal explícito; tem precedência sobre o setor.

    Example
    -------
    >>> from advisor_brain_fsa import BeneishMScore
    >>> from advisor_brain_fsa.accruals import CashFlowQuality
    >>> mscore = BeneishMScore(current=data_t, prior=data_t1).calculate()
    >>> cfq    = CashFlowQuality(current=data_t, prior=data_t1).calculate(mscore)
    >>> print(cfq)
    """

    def __init__(self, current: FinancialData, prior: FinancialData,
                 setor: str | None = None, mediana: float | None = None,
                 ticker: str | None = None) -> None:
        self._t  = current
        self._t1 = prior
        self._setor = setor
        self._mediana = mediana
        self._ticker = ticker

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

        if self._mediana is not None:
            mediana, origem = float(self._mediana), "informada"
        else:
            mediana, origem = referencia_accrual(self._setor, self._ticker)
        desvio  = ratio - mediana
        quality = qualidade_por_desvio(desvio)
        alert   = _combine_alert(mscore_result, quality)

        return CashFlowQualityResult(
            accrual_ratio   = ratio,
            earnings_quality= quality,
            alert_level     = alert,
            mediana_setor   = mediana,
            desvio          = desvio,
            referencia      = origem,
        )
