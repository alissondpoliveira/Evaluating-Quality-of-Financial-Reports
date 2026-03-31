"""
Beneish M-Score Model
---------------------
Detects earnings manipulation probability using 8 financial ratios,
as described in Messod D. Beneish (1999) and covered in CFA Level 2 curriculum.

M-Score = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
          + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI

Threshold: M-Score > -1.78 → Potential Manipulator
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FinancialData:
    """
    Container for one period's accounting data required by the Beneish model.

    All monetary values should be in the same currency unit (e.g. thousands USD).
    """

    # Income Statement
    revenues: float
    cost_of_goods_sold: float
    sales_general_admin_expenses: float  # SGA

    # Balance Sheet
    receivables: float          # Net accounts receivable
    total_assets: float
    current_assets: float
    pp_and_e: float             # Property, Plant & Equipment (net)
    securities: float           # Marketable securities + short-term investments
    total_long_term_debt: float
    current_liabilities: float

    # Cash Flow / Accruals
    depreciation: float         # Depreciation & amortisation expense
    net_income: float
    cash_from_operations: float

    # Fields that represent costs/expenses and must always be stored as
    # positive absolute values for the Beneish formulas to be valid.
    _MUST_BE_POSITIVE = frozenset({
        "cost_of_goods_sold",
        "sales_general_admin_expenses",
        "depreciation",
    })

    def __post_init__(self) -> None:
        all_fields = [
            "revenues", "cost_of_goods_sold", "sales_general_admin_expenses",
            "receivables", "total_assets", "current_assets", "pp_and_e",
            "securities", "total_long_term_debt", "current_liabilities",
            "depreciation", "net_income", "cash_from_operations",
        ]
        for f in all_fields:
            v = getattr(self, f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raise ValueError(f"Field '{f}' must not be None or NaN.")
            v = float(v)
            # Defensive abs() for cost/expense fields — guards against
            # upstream sign inconsistencies in CVM DRE/DFC files.
            if f in self._MUST_BE_POSITIVE and v < 0:
                logger.warning(
                    "FinancialData: field '%s' arrived as negative (%.4f); "
                    "taking abs() — check data_fetcher sign correction.",
                    f, v,
                )
                v = abs(v)
            setattr(self, f, v)


@dataclass
class MScoreResult:
    """Stores all intermediate ratios and the final M-Score."""

    # Eight indices
    dsri: float   # Days Sales in Receivables Index
    gmi: float    # Gross Margin Index
    aqi: float    # Asset Quality Index
    sgi: float    # Sales Growth Index
    depi: float   # Depreciation Index
    sgai: float   # SGA Expense Index
    lvgi: float   # Leverage Index
    tata: float   # Total Accruals to Total Assets

    # Final score
    m_score: float

    @property
    def is_manipulator(self) -> bool:
        return self.m_score > -1.78

    @property
    def classification(self) -> str:
        return "Potential Manipulator" if self.is_manipulator else "Non-Manipulator"

    def to_dict(self) -> dict:
        return {
            "DSRI": round(self.dsri, 6),
            "GMI": round(self.gmi, 6),
            "AQI": round(self.aqi, 6),
            "SGI": round(self.sgi, 6),
            "DEPI": round(self.depi, 6),
            "SGAI": round(self.sgai, 6),
            "LVGI": round(self.lvgi, 6),
            "TATA": round(self.tata, 6),
            "M-Score": round(self.m_score, 6),
            "Classification": self.classification,
        }

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())

    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "           BENEISH M-SCORE ANALYSIS",
            "=" * 50,
            f"  DSRI  (Days Sales in Receivables):  {self.dsri:>10.4f}",
            f"  GMI   (Gross Margin Index):          {self.gmi:>10.4f}",
            f"  AQI   (Asset Quality Index):         {self.aqi:>10.4f}",
            f"  SGI   (Sales Growth Index):          {self.sgi:>10.4f}",
            f"  DEPI  (Depreciation Index):          {self.depi:>10.4f}",
            f"  SGAI  (SGA Expense Index):           {self.sgai:>10.4f}",
            f"  LVGI  (Leverage Index):              {self.lvgi:>10.4f}",
            f"  TATA  (Total Accruals/Total Assets): {self.tata:>10.4f}",
            "-" * 50,
            f"  M-Score:                             {self.m_score:>10.4f}",
            f"  Threshold:                                  -1.7800",
            f"  Classification: {self.classification}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model coefficients (Beneish 1999, probit model)
# ---------------------------------------------------------------------------
_COEFFICIENTS = {
    "intercept": -4.84,
    "dsri":       0.920,
    "gmi":        0.528,
    "aqi":        0.404,
    "sgi":        0.892,
    "depi":       0.115,
    "sgai":      -0.172,
    "tata":       4.679,
    "lvgi":      -0.327,
}

_MANIPULATION_THRESHOLD = -1.78


def _safe_div(numerator: float, denominator: float, fallback: float = 1.0) -> float:
    """Division with zero/NaN guard. Returns *fallback* when denominator is near-zero."""
    if abs(denominator) < 1e-6 or np.isnan(denominator) or np.isnan(numerator):
        return fallback
    return numerator / denominator


def _positive_index(name: str, value: float, fallback: float = 1.0) -> float:
    """
    Validate that a Beneish ratio index is positive.

    DSRI, GMI, AQI, SGI, DEPI, SGAI and LVGI are ratios of proportions
    and must be positive for the M-Score formula to be valid.  A negative
    value indicates a sign anomaly in the upstream financial data.
    Log a WARNING and return *fallback* (neutral = 1.0) in that case.
    """
    if value < 0:
        logger.warning(
            "Beneish index %s = %.6f is negative — sign anomaly in input data. "
            "Substituting neutral fallback %.1f to prevent M-Score corruption.",
            name, value, fallback,
        )
        return fallback
    return value


class BeneishMScore:
    """
    Calculates the Beneish M-Score for a given company.

    Parameters
    ----------
    current : FinancialData
        Accounting data for the current period (T).
    prior : FinancialData
        Accounting data for the prior period (T-1).

    Example
    -------
    >>> model = BeneishMScore(current=data_t, prior=data_t1)
    >>> result = model.calculate()
    >>> print(result)
    """

    def __init__(self, current: FinancialData, prior: FinancialData) -> None:
        self._t = current
        self._t1 = prior

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self) -> MScoreResult:
        """Compute all 8 indices and the final M-Score."""
        t, t1 = self._t, self._t1

        dsri = self._dsri(t, t1)
        gmi = self._gmi(t, t1)
        aqi = self._aqi(t, t1)
        sgi = self._sgi(t, t1)
        depi = self._depi(t, t1)
        sgai = self._sgai(t, t1)
        lvgi = self._lvgi(t, t1)
        tata = self._tata(t)

        m_score = (
            _COEFFICIENTS["intercept"]
            + _COEFFICIENTS["dsri"] * dsri
            + _COEFFICIENTS["gmi"] * gmi
            + _COEFFICIENTS["aqi"] * aqi
            + _COEFFICIENTS["sgi"] * sgi
            + _COEFFICIENTS["depi"] * depi
            + _COEFFICIENTS["sgai"] * sgai
            + _COEFFICIENTS["tata"] * tata
            + _COEFFICIENTS["lvgi"] * lvgi
        )

        return MScoreResult(
            dsri=dsri, gmi=gmi, aqi=aqi, sgi=sgi,
            depi=depi, sgai=sgai, lvgi=lvgi, tata=tata,
            m_score=m_score,
        )

    # ------------------------------------------------------------------
    # Index calculators
    # ------------------------------------------------------------------

    @staticmethod
    def _dsri(t: FinancialData, t1: FinancialData) -> float:
        """
        Days Sales in Receivables Index.
        DSRI = (Receivables_T / Revenues_T) / (Receivables_T1 / Revenues_T1)

        A large increase suggests revenue inflation or channel stuffing.
        Must be positive; negative result indicates upstream data sign error.
        """
        ratio_t  = _safe_div(t.receivables,  t.revenues)
        ratio_t1 = _safe_div(t1.receivables, t1.revenues)
        return _positive_index("DSRI", _safe_div(ratio_t, ratio_t1))

    @staticmethod
    def _gmi(t: FinancialData, t1: FinancialData) -> float:
        """
        Gross Margin Index.
        GMI = [(Revenues_T1 - COGS_T1) / Revenues_T1] /
              [(Revenues_T  - COGS_T)  / Revenues_T ]

        GMI > 1 indicates deteriorating gross margins, a red flag.
        Must be positive; negative result means gross margin flipped sign
        (e.g. COGS > Revenue in one period), which is treated as anomalous.
        """
        gm_t  = _safe_div(t.revenues  - t.cost_of_goods_sold,  t.revenues)
        gm_t1 = _safe_div(t1.revenues - t1.cost_of_goods_sold, t1.revenues)
        return _positive_index("GMI", _safe_div(gm_t1, gm_t))

    @staticmethod
    def _aqi(t: FinancialData, t1: FinancialData) -> float:
        """
        Asset Quality Index.
        AQI = [1 - (CurrentAssets_T + PP&E_T) / TotalAssets_T] /
              [1 - (CurrentAssets_T1 + PP&E_T1) / TotalAssets_T1]

        Measures the proportion of non-current, non-PP&E assets
        (i.e., assets more susceptible to manipulation).
        Must be positive; negative result (current + PP&E > total assets)
        indicates a data integrity issue.
        """
        quality_t  = 1.0 - _safe_div(t.current_assets  + t.pp_and_e,  t.total_assets)
        quality_t1 = 1.0 - _safe_div(t1.current_assets + t1.pp_and_e, t1.total_assets)
        return _positive_index("AQI", _safe_div(quality_t, quality_t1))

    @staticmethod
    def _sgi(t: FinancialData, t1: FinancialData) -> float:
        """
        Sales Growth Index.
        SGI = Revenues_T / Revenues_T1

        High growth firms face greater incentives to manipulate.
        Must be positive; both revenue figures must be positive.
        """
        return _positive_index("SGI", _safe_div(t.revenues, t1.revenues))

    @staticmethod
    def _depi(t: FinancialData, t1: FinancialData) -> float:
        """
        Depreciation Index.
        DEPI = [Depreciation_T1 / (PP&E_T1 + Depreciation_T1)] /
               [Depreciation_T  / (PP&E_T  + Depreciation_T )]

        DEPI > 1 may signal assets being depreciated more slowly.
        Must be positive; a negative value (the root cause of the CVCB3/2019
        bug) means D&A arrived with the wrong sign — caught here as last resort.
        """
        rate_t  = _safe_div(t.depreciation,  t.pp_and_e  + t.depreciation)
        rate_t1 = _safe_div(t1.depreciation, t1.pp_and_e + t1.depreciation)
        return _positive_index("DEPI", _safe_div(rate_t1, rate_t))

    @staticmethod
    def _sgai(t: FinancialData, t1: FinancialData) -> float:
        """
        SGA Expense Index.
        SGAI = (SGA_T / Revenues_T) / (SGA_T1 / Revenues_T1)

        Disproportionate SGA growth may indicate future problems.
        Must be positive; SGA and revenues should both be positive.
        """
        ratio_t  = _safe_div(t.sales_general_admin_expenses,  t.revenues)
        ratio_t1 = _safe_div(t1.sales_general_admin_expenses, t1.revenues)
        return _positive_index("SGAI", _safe_div(ratio_t, ratio_t1))

    @staticmethod
    def _lvgi(t: FinancialData, t1: FinancialData) -> float:
        """
        Leverage Index.
        LVGI = [(LongTermDebt_T + CurrentLiabilities_T) / TotalAssets_T] /
               [(LongTermDebt_T1 + CurrentLiabilities_T1) / TotalAssets_T1]

        Increased leverage creates incentive to manipulate earnings.
        Must be positive; all balance sheet items should be non-negative.
        """
        lev_t  = _safe_div(t.total_long_term_debt  + t.current_liabilities,  t.total_assets)
        lev_t1 = _safe_div(t1.total_long_term_debt + t1.current_liabilities, t1.total_assets)
        return _positive_index("LVGI", _safe_div(lev_t, lev_t1))

    @staticmethod
    def _tata(t: FinancialData) -> float:
        """
        Total Accruals to Total Assets (current period only).
        TATA = (NetIncome_T - CashFromOperations_T) / TotalAssets_T

        High accruals relative to assets indicate aggressive accounting.
        """
        accruals = t.net_income - t.cash_from_operations
        return _safe_div(accruals, t.total_assets, fallback=0.0)
