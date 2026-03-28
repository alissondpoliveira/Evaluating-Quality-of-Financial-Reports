"""
Sector-specific Risk Scorers — Strategy Pattern
------------------------------------------------
Provides pluggable risk-scoring strategies for different B3 industry groups.

  Standard (non-financial) → BeneishSectorScorer
      Uses Beneish M-Score (1999) + CFA Level 2 Accruals Quality.

  Banks / Financeiro       → BankingScorer
      Uses proxies for ROA, cost-to-income ratio, CFO quality
      and interest-spread margin (BACEN prudential benchmarks).

  Insurance / Seguros      → InsuranceScorer
      Uses Loss Ratio, Expense Ratio and Combined Ratio
      (SUSEP regulatory thresholds for Brazilian insurance market).

Usage
-----
    from advisor_brain_fsa.sector_scorer import get_scorer

    scorer = get_scorer(sector="Bancos")
    result = scorer.score(current=fd_t, prior=fd_t1)
    print(result.alert_level, result.red_flags)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .accruals import AlertLevel, CashFlowQuality, CashFlowQualityResult
from .beneish_mscore import BeneishMScore, FinancialData, MScoreResult, _safe_div
from .ticker_map import FINANCIAL_GROUP

# ---------------------------------------------------------------------------
# Sector group helpers
# ---------------------------------------------------------------------------

INSURANCE_SECTORS = frozenset({"Seguros"})
BANKING_SECTORS   = frozenset({"Bancos", "Financeiro"})


def is_financial(sector: str) -> bool:
    return sector in FINANCIAL_GROUP


# ---------------------------------------------------------------------------
# Unified result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SectorRiskResult:
    """
    Unified risk result returned by any SectorScorer implementation.

    Attributes
    ----------
    risk_score : float
        Normalised 0–10 score (higher = more risk). Used for cross-sector
        ranking within the same group.
    classification : str
        Human-readable risk classification.
    alert_level : AlertLevel
        Alert level enum (Normal / Atenção / Alto Risco / Crítico).
    metrics : dict
        Scorer-specific key metrics (e.g. m_score, roa, combined_ratio).
    red_flags : list[str]
        Top-3 human-readable risk flags.
    scorer_type : str
        "beneish" | "banking" | "insurance"
    mscore_result : MScoreResult | None
        Set only for BeneishSectorScorer.
    cfq_result : CashFlowQualityResult | None
        Set only for BeneishSectorScorer.
    """
    risk_score:     float
    classification: str
    alert_level:    AlertLevel
    metrics:        Dict[str, float]
    red_flags:      List[str]
    scorer_type:    str
    mscore_result:  Optional[MScoreResult]         = None
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
# Strategy 1: Standard Beneish + CFA Accruals (all non-financial sectors)
# ---------------------------------------------------------------------------

class BeneishSectorScorer(SectorScorer):
    """
    Default scorer for industrial, consumer, utility, and other
    non-financial companies.  Uses the Beneish M-Score model (1999)
    combined with the CFA Level 2 Accrual Ratio quality metric.
    """

    @property
    def scorer_type(self) -> str:
        return "beneish"

    def score(self, current: FinancialData, prior: FinancialData) -> SectorRiskResult:
        from .rank_market import detect_red_flags

        ms  = BeneishMScore(current=current, prior=prior).calculate()
        cfq = CashFlowQuality(current=current, prior=prior).calculate(ms)
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
# Strategy 2: Banking / Financial Institutions
# ---------------------------------------------------------------------------

class BankingScorer(SectorScorer):
    """
    Risk scorer for banks (Bancos) and financial services (Financeiro).

    FinancialData fields are re-interpreted for financial institutions:
      revenues              → Receita de Intermediação Financeira
      cost_of_goods_sold    → Despesa de Intermediação Financeira
      sales_general_admin   → Despesas Administrativas (overhead)
      net_income            → Lucro Líquido
      total_assets          → Ativo Total
      cash_from_operations  → CFO
      total_long_term_debt  → Passivo LP (capital proxy)

    Key metrics (BACEN / Basel III references):
      ROA              = net_income / total_assets (>1% healthy, <0.5% risky)
      Cost-to-Income   = sga / revenues (<50% efficient, >75% critical)
      CFO Quality      = cfo / net_income (>0.8 good, <0.5 poor)
      Spread Margin    = (revenues − cogs) / revenues
      Revenue Growth   = revenues_t / revenues_t1
    """

    _ROA_WARN          = 0.008   # < 0.8 % → watch
    _ROA_CRIT          = 0.003   # < 0.3 % → high risk
    _COST_INCOME_WARN  = 0.60    # > 60 % → watch
    _COST_INCOME_CRIT  = 0.75    # > 75 % → critical
    _CFO_QUALITY_WARN  = 0.50    # CFO/NI < 50 % → low cash quality
    _SPREAD_WARN       = 0.20    # interest spread < 20 % → compressed
    _REV_DECLINE       = 0.97    # revenue shrinkage > 3 % → red flag

    @property
    def scorer_type(self) -> str:
        return "banking"

    def score(self, current: FinancialData, prior: FinancialData) -> SectorRiskResult:
        t, t1 = current, prior

        roa         = _safe_div(t.net_income, t.total_assets, fallback=0.0)
        cost_income = _safe_div(t.sales_general_admin_expenses, t.revenues, fallback=0.0)
        cfo_quality = _safe_div(t.cash_from_operations, t.net_income, fallback=1.0)
        rev_growth  = _safe_div(t.revenues, t1.revenues, fallback=1.0)
        spread      = _safe_div(t.revenues - t.cost_of_goods_sold, t.revenues, fallback=0.0)
        leverage    = _safe_div(t.total_long_term_debt, t.total_assets, fallback=0.0)

        flags: List[str] = []
        pts = 0.0

        if roa < self._ROA_CRIT:
            flags.append(
                f"ROA crítico ({roa:.2%}) — rentabilidade muito abaixo do mínimo "
                f"BACEN ({self._ROA_WARN:.1%})"
            )
            pts += 4.0
        elif roa < self._ROA_WARN:
            flags.append(
                f"ROA deteriorado ({roa:.2%}) — abaixo do nível prudencial recomendado"
            )
            pts += 2.0

        if cost_income > self._COST_INCOME_CRIT:
            flags.append(
                f"Índice de eficiência crítico ({cost_income:.1%}) — despesas consomem "
                f"{cost_income:.0%} da receita financeira"
            )
            pts += 3.0
        elif cost_income > self._COST_INCOME_WARN:
            flags.append(
                f"Índice de eficiência elevado ({cost_income:.1%}) — overhead acima de 60%"
            )
            pts += 1.5

        if t.net_income > 0 and cfo_quality < self._CFO_QUALITY_WARN:
            flags.append(
                f"Baixa qualidade de caixa: CFO/LL = {cfo_quality:.2f} "
                f"— lucro não se converte em caixa operacional"
            )
            pts += 2.0

        if rev_growth < self._REV_DECLINE:
            flags.append(
                f"Contração de receita financeira: {rev_growth:.1%} "
                f"— intermediação financeira encolhendo"
            )
            pts += 1.5

        if spread < self._SPREAD_WARN:
            flags.append(
                f"Spread financeiro comprimido ({spread:.1%}) "
                f"— margem entre captação e aplicação reduzida"
            )
            pts += 1.0

        risk_score = float(np.clip(pts, 0, 10))

        if risk_score >= 7:
            alert          = AlertLevel.CRITICAL
            classification = "Crítico — Indicadores prudenciais bancários de alto risco"
        elif risk_score >= 4:
            alert          = AlertLevel.HIGH_RISK
            classification = "Alto Risco — Métricas de eficiência e rentabilidade deterioradas"
        elif risk_score >= 2:
            alert          = AlertLevel.WATCH
            classification = "Atenção — Monitorar eficiência operacional e qualidade de crédito"
        else:
            alert          = AlertLevel.NORMAL
            classification = "Normal — Indicadores bancários dentro do esperado"

        return SectorRiskResult(
            risk_score=risk_score,
            classification=classification,
            alert_level=alert,
            metrics={
                "roa":          round(roa, 4),
                "cost_income":  round(cost_income, 4),
                "cfo_quality":  round(cfo_quality, 4),
                "rev_growth":   round(rev_growth, 4),
                "spread":       round(spread, 4),
                "leverage":     round(leverage, 4),
            },
            red_flags=flags[:3],
            scorer_type=self.scorer_type,
        )


# ---------------------------------------------------------------------------
# Strategy 3: Insurance Companies
# ---------------------------------------------------------------------------

class InsuranceScorer(SectorScorer):
    """
    Risk scorer for insurance companies (Seguros).

    FinancialData fields are re-interpreted for insurers:
      revenues              → Prêmios Retidos
      cost_of_goods_sold    → Sinistros Retidos
      sales_general_admin   → Despesas de Comercialização + Admin
      net_income            → Lucro Líquido
      total_assets          → Ativo Total
      cash_from_operations  → CFO

    Key metrics (SUSEP / ANS references):
      Loss Ratio (Sinistralidade) = claims / premiums (<65% good, >80% critical)
      Expense Ratio               = sga / premiums (<30% good)
      Combined Ratio (IC)         = loss + expense (<100% = underwriting profit)
      Revenue Growth              = premiums_t / premiums_t1
    """

    _LOSS_WARN     = 0.65   # > 65 % → watch
    _LOSS_CRIT     = 0.80   # > 80 % → critical
    _COMBINED_WARN = 0.95   # > 95 % → watch
    _COMBINED_CRIT = 1.00   # > 100 % → underwriting loss
    _REV_DECLINE   = 0.97

    @property
    def scorer_type(self) -> str:
        return "insurance"

    def score(self, current: FinancialData, prior: FinancialData) -> SectorRiskResult:
        t, t1 = current, prior

        loss_ratio     = _safe_div(t.cost_of_goods_sold, t.revenues, fallback=0.0)
        expense_ratio  = _safe_div(t.sales_general_admin_expenses, t.revenues, fallback=0.0)
        combined_ratio = loss_ratio + expense_ratio
        rev_growth     = _safe_div(t.revenues, t1.revenues, fallback=1.0)
        roa            = _safe_div(t.net_income, t.total_assets, fallback=0.0)
        cfo_quality    = _safe_div(t.cash_from_operations, t.net_income, fallback=1.0)

        flags: List[str] = []
        pts = 0.0

        if combined_ratio > self._COMBINED_CRIT:
            flags.append(
                f"Índice Combinado > 100% ({combined_ratio:.1%}) — resultado de subscrição "
                f"negativo: sinistros + despesas superam os prêmios arrecadados"
            )
            pts += 5.0
        elif combined_ratio > self._COMBINED_WARN:
            flags.append(
                f"Índice Combinado elevado ({combined_ratio:.1%}) — "
                f"operações de subscrição sob pressão"
            )
            pts += 2.5

        if loss_ratio > self._LOSS_CRIT:
            flags.append(
                f"Sinistralidade crítica ({loss_ratio:.1%}) — sinistros consomem "
                f"{loss_ratio:.0%} dos prêmios (limiar SUSEP: 65%)"
            )
            pts += 3.0
        elif loss_ratio > self._LOSS_WARN:
            flags.append(
                f"Sinistralidade elevada ({loss_ratio:.1%}) — acima do nível de atenção"
            )
            pts += 1.5

        if rev_growth < self._REV_DECLINE:
            flags.append(
                f"Retração de prêmios ({rev_growth:.1%}) — carteira encolhendo"
            )
            pts += 1.5

        if t.net_income > 0 and cfo_quality < 0.50:
            flags.append(
                f"Baixa conversão de caixa: CFO/LL = {cfo_quality:.2f}"
            )
            pts += 1.0

        risk_score = float(np.clip(pts, 0, 10))

        if risk_score >= 7:
            alert          = AlertLevel.CRITICAL
            classification = "Crítico — Resultado técnico (subscrição) deficitário"
        elif risk_score >= 4:
            alert          = AlertLevel.HIGH_RISK
            classification = "Alto Risco — Sinistralidade e custos operacionais pressionados"
        elif risk_score >= 2:
            alert          = AlertLevel.WATCH
            classification = "Atenção — Monitorar índice combinado e sinistralidade"
        else:
            alert          = AlertLevel.NORMAL
            classification = "Normal — Indicadores de seguro dentro do esperado"

        return SectorRiskResult(
            risk_score=risk_score,
            classification=classification,
            alert_level=alert,
            metrics={
                "loss_ratio":     round(loss_ratio, 4),
                "expense_ratio":  round(expense_ratio, 4),
                "combined_ratio": round(combined_ratio, 4),
                "rev_growth":     round(rev_growth, 4),
                "roa":            round(roa, 4),
                "cfo_quality":    round(cfo_quality, 4),
            },
            red_flags=flags[:3],
            scorer_type=self.scorer_type,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def get_scorer(sector: str) -> SectorScorer:
    """
    Return the appropriate SectorScorer for the given sector string.

    Financial institutions (Bancos, Financeiro) → BankingScorer
    Insurance companies (Seguros)               → InsuranceScorer
    All other sectors                           → BeneishSectorScorer
    """
    if sector in INSURANCE_SECTORS:
        return InsuranceScorer()
    if sector in BANKING_SECTORS:
        return BankingScorer()
    return BeneishSectorScorer()
