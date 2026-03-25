"""
Unit tests for the BeneishMScore module.
Run with: pytest tests/
"""

import math
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.beneish_mscore import (
    BeneishMScore,
    FinancialData,
    MScoreResult,
    _safe_div,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clean_data_t() -> FinancialData:
    """Healthy company – current period."""
    return FinancialData(
        revenues=1_200_000,
        cost_of_goods_sold=720_000,
        sales_general_admin_expenses=120_000,
        receivables=100_000,
        total_assets=900_000,
        current_assets=300_000,
        pp_and_e=400_000,
        securities=50_000,
        total_long_term_debt=180_000,
        current_liabilities=90_000,
        depreciation=40_000,
        net_income=120_000,
        cash_from_operations=150_000,
    )


def _make_clean_data_t1() -> FinancialData:
    """Healthy company – prior period."""
    return FinancialData(
        revenues=1_000_000,
        cost_of_goods_sold=600_000,
        sales_general_admin_expenses=100_000,
        receivables=80_000,
        total_assets=800_000,
        current_assets=260_000,
        pp_and_e=360_000,
        securities=40_000,
        total_long_term_debt=160_000,
        current_liabilities=80_000,
        depreciation=36_000,
        net_income=100_000,
        cash_from_operations=130_000,
    )


def _make_manipulator_t() -> FinancialData:
    """Aggressive accounting – current period."""
    return FinancialData(
        revenues=1_500_000,
        cost_of_goods_sold=1_200_000,       # very thin margin
        sales_general_admin_expenses=300_000,
        receivables=400_000,                 # bloated receivables
        total_assets=1_000_000,
        current_assets=200_000,
        pp_and_e=200_000,
        securities=20_000,
        total_long_term_debt=500_000,        # heavy debt
        current_liabilities=200_000,
        depreciation=10_000,                 # very low depreciation
        net_income=200_000,
        cash_from_operations=20_000,         # huge accruals
    )


def _make_manipulator_t1() -> FinancialData:
    """Aggressive accounting – prior period."""
    return FinancialData(
        revenues=1_000_000,
        cost_of_goods_sold=700_000,
        sales_general_admin_expenses=150_000,
        receivables=100_000,
        total_assets=800_000,
        current_assets=250_000,
        pp_and_e=280_000,
        securities=30_000,
        total_long_term_debt=200_000,
        current_liabilities=100_000,
        depreciation=30_000,
        net_income=80_000,
        cash_from_operations=100_000,
    )


# ---------------------------------------------------------------------------
# _safe_div
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal_division(self):
        assert _safe_div(10.0, 2.0) == pytest.approx(5.0)

    def test_zero_denominator_returns_fallback(self):
        assert _safe_div(10.0, 0.0) == pytest.approx(1.0)

    def test_custom_fallback(self):
        assert _safe_div(5.0, 0.0, fallback=0.0) == pytest.approx(0.0)

    def test_nan_numerator_returns_fallback(self):
        assert _safe_div(float("nan"), 5.0) == pytest.approx(1.0)

    def test_nan_denominator_returns_fallback(self):
        assert _safe_div(5.0, float("nan")) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FinancialData validation
# ---------------------------------------------------------------------------

class TestFinancialData:
    def test_valid_construction(self):
        fd = _make_clean_data_t()
        assert fd.revenues == 1_200_000.0

    def test_rejects_none_field(self):
        with pytest.raises((TypeError, ValueError)):
            FinancialData(
                revenues=None,  # type: ignore
                cost_of_goods_sold=720_000,
                sales_general_admin_expenses=120_000,
                receivables=100_000,
                total_assets=900_000,
                current_assets=300_000,
                pp_and_e=400_000,
                securities=50_000,
                total_long_term_debt=180_000,
                current_liabilities=90_000,
                depreciation=40_000,
                net_income=120_000,
                cash_from_operations=150_000,
            )

    def test_rejects_nan_field(self):
        with pytest.raises(ValueError):
            FinancialData(
                revenues=float("nan"),
                cost_of_goods_sold=720_000,
                sales_general_admin_expenses=120_000,
                receivables=100_000,
                total_assets=900_000,
                current_assets=300_000,
                pp_and_e=400_000,
                securities=50_000,
                total_long_term_debt=180_000,
                current_liabilities=90_000,
                depreciation=40_000,
                net_income=120_000,
                cash_from_operations=150_000,
            )


# ---------------------------------------------------------------------------
# BeneishMScore – individual indices
# ---------------------------------------------------------------------------

class TestIndices:
    def setup_method(self):
        self.t = _make_clean_data_t()
        self.t1 = _make_clean_data_t1()
        self.model = BeneishMScore(current=self.t, prior=self.t1)

    def test_dsri_positive(self):
        dsri = BeneishMScore._dsri(self.t, self.t1)
        assert dsri > 0

    def test_gmi_positive(self):
        gmi = BeneishMScore._gmi(self.t, self.t1)
        assert gmi > 0

    def test_aqi_positive(self):
        aqi = BeneishMScore._aqi(self.t, self.t1)
        assert aqi > 0

    def test_sgi_equals_revenue_growth(self):
        sgi = BeneishMScore._sgi(self.t, self.t1)
        expected = self.t.revenues / self.t1.revenues
        assert sgi == pytest.approx(expected)

    def test_depi_positive(self):
        depi = BeneishMScore._depi(self.t, self.t1)
        assert depi > 0

    def test_sgai_positive(self):
        sgai = BeneishMScore._sgai(self.t, self.t1)
        assert sgai > 0

    def test_lvgi_positive(self):
        lvgi = BeneishMScore._lvgi(self.t, self.t1)
        assert lvgi > 0

    def test_tata_with_negative_accruals(self):
        # Cash from ops > Net income → negative TATA (good sign)
        tata = BeneishMScore._tata(self.t)
        assert tata < 0

    def test_tata_zero_total_assets_fallback(self):
        t_zero = FinancialData(
            revenues=100, cost_of_goods_sold=60,
            sales_general_admin_expenses=10, receivables=10,
            total_assets=0, current_assets=0, pp_and_e=0,
            securities=0, total_long_term_debt=0, current_liabilities=0,
            depreciation=5, net_income=20, cash_from_operations=5,
        )
        # Should not raise; returns fallback 0.0
        tata = BeneishMScore._tata(t_zero)
        assert tata == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BeneishMScore – full calculation
# ---------------------------------------------------------------------------

class TestFullCalculation:
    def test_healthy_company_is_non_manipulator(self):
        model = BeneishMScore(
            current=_make_clean_data_t(),
            prior=_make_clean_data_t1(),
        )
        result = model.calculate()
        assert isinstance(result, MScoreResult)
        assert not result.is_manipulator
        assert result.classification == "Non-Manipulator"

    def test_aggressive_company_is_manipulator(self):
        model = BeneishMScore(
            current=_make_manipulator_t(),
            prior=_make_manipulator_t1(),
        )
        result = model.calculate()
        assert result.is_manipulator
        assert result.classification == "Potential Manipulator"

    def test_mscore_is_finite(self):
        model = BeneishMScore(
            current=_make_clean_data_t(),
            prior=_make_clean_data_t1(),
        )
        result = model.calculate()
        assert math.isfinite(result.m_score)

    def test_to_dict_has_all_keys(self):
        model = BeneishMScore(
            current=_make_clean_data_t(),
            prior=_make_clean_data_t1(),
        )
        d = model.calculate().to_dict()
        expected_keys = {"DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA",
                         "M-Score", "Classification"}
        assert expected_keys == set(d.keys())

    def test_to_series_returns_pandas_series(self):
        import pandas as pd
        model = BeneishMScore(
            current=_make_clean_data_t(),
            prior=_make_clean_data_t1(),
        )
        s = model.calculate().to_series()
        assert isinstance(s, pd.Series)

    def test_str_contains_mscore(self):
        model = BeneishMScore(
            current=_make_clean_data_t(),
            prior=_make_clean_data_t1(),
        )
        output = str(model.calculate())
        assert "M-Score" in output
        assert "Non-Manipulator" in output
