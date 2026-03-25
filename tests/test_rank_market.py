"""
Tests for rank_market, accruals, and report_generator.

All CVM network calls are mocked — no HTTP requests.
"""

from __future__ import annotations

import io
import math
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.accruals import (
    AlertLevel,
    CashFlowQuality,
    CashFlowQualityResult,
    _earnings_quality_label,
    _combine_alert,
)
from advisor_brain_fsa.beneish_mscore import BeneishMScore, FinancialData, MScoreResult
from advisor_brain_fsa.rank_market import (
    CompanyResult,
    DEFAULT_WATCHLIST,
    _apply_sector_stats,
    _to_dataframe,
    detect_red_flags,
    rank_market,
)
from advisor_brain_fsa.report_generator import build_markdown, generate_report
from advisor_brain_fsa.ticker_map import get_sector, TICKER_SECTOR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_fd(
    revenues=1_200_000, cogs=720_000, sga=120_000,
    receivables=100_000, total_assets=900_000, current_assets=300_000,
    pp_and_e=400_000, securities=50_000, ltd=180_000, cl=90_000,
    depreciation=40_000, net_income=120_000, cfo=150_000,
) -> FinancialData:
    return FinancialData(
        revenues=revenues,
        cost_of_goods_sold=cogs,
        sales_general_admin_expenses=sga,
        receivables=receivables,
        total_assets=total_assets,
        current_assets=current_assets,
        pp_and_e=pp_and_e,
        securities=securities,
        total_long_term_debt=ltd,
        current_liabilities=cl,
        depreciation=depreciation,
        net_income=net_income,
        cash_from_operations=cfo,
    )


@pytest.fixture
def fd_t():
    return _make_fd()


@pytest.fixture
def fd_t1():
    return _make_fd(
        revenues=1_000_000, cogs=600_000, sga=100_000,
        receivables=80_000, total_assets=800_000, current_assets=260_000,
        pp_and_e=360_000, securities=40_000, ltd=160_000, cl=80_000,
        depreciation=36_000, net_income=100_000, cfo=130_000,
    )


@pytest.fixture
def mscore_result(fd_t, fd_t1) -> MScoreResult:
    return BeneishMScore(current=fd_t, prior=fd_t1).calculate()


@pytest.fixture
def cfq_result(fd_t, fd_t1, mscore_result) -> CashFlowQualityResult:
    return CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore_result)


# ---------------------------------------------------------------------------
# ticker_map — sector extension
# ---------------------------------------------------------------------------

class TestGetSector:
    def test_petr4_is_energia(self):
        assert get_sector("PETR4") == "Energia"

    def test_itub4_is_bancos(self):
        assert get_sector("ITUB4") == "Bancos"

    def test_vale3_is_mineracao(self):
        assert get_sector("VALE3") == "Mineração"

    def test_unknown_ticker_returns_outros(self):
        assert get_sector("XXXX99") == "Outros"

    def test_case_insensitive(self):
        assert get_sector("petr4") == "Energia"

    def test_all_watchlist_tickers_have_sector(self):
        for ticker in DEFAULT_WATCHLIST:
            sector = get_sector(ticker)
            assert isinstance(sector, str) and len(sector) > 0


# ---------------------------------------------------------------------------
# accruals — CashFlowQuality
# ---------------------------------------------------------------------------

class TestEarningsQualityLabel:
    def test_below_warn_threshold_is_alta(self):
        assert _earnings_quality_label(0.005) == "Alta"

    def test_between_thresholds_is_moderada(self):
        assert _earnings_quality_label(0.03) == "Moderada"

    def test_above_high_threshold_is_baixa(self):
        assert _earnings_quality_label(0.06) == "Baixa"

    def test_exactly_on_warn_threshold(self):
        assert _earnings_quality_label(0.01) == "Moderada"

    def test_exactly_on_high_threshold(self):
        assert _earnings_quality_label(0.05) == "Baixa"

    def test_negative_accruals_is_alta(self):
        assert _earnings_quality_label(-0.05) == "Alta"


class TestCombineAlert:
    def _ms(self, m_score: float) -> MScoreResult:
        return MScoreResult(
            dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
            depi=1.0, sgai=1.0, lvgi=1.0, tata=0.0,
            m_score=m_score,
        )

    def test_non_manipulator_alta_quality_is_normal(self):
        assert _combine_alert(self._ms(-2.5), "Alta") == AlertLevel.NORMAL

    def test_non_manipulator_baixa_quality_is_watch(self):
        assert _combine_alert(self._ms(-2.5), "Baixa") == AlertLevel.WATCH

    def test_manipulator_alta_quality_is_high_risk(self):
        assert _combine_alert(self._ms(-1.0), "Alta") == AlertLevel.HIGH_RISK

    def test_manipulator_baixa_quality_is_critical(self):
        assert _combine_alert(self._ms(-1.0), "Baixa") == AlertLevel.CRITICAL

    def test_boundary_mscore_is_non_manipulator(self):
        # exactly -1.78 → is_manipulator = False (strictly greater than)
        assert _combine_alert(self._ms(-1.78), "Alta") == AlertLevel.NORMAL


class TestCashFlowQualityCalculate:
    def test_returns_cfq_result(self, fd_t, fd_t1, mscore_result):
        cfq = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore_result)
        assert isinstance(cfq, CashFlowQualityResult)

    def test_accrual_ratio_formula(self, fd_t, fd_t1, mscore_result):
        cfq = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore_result)
        avg_assets = (fd_t.total_assets + fd_t1.total_assets) / 2
        expected = (fd_t.net_income - fd_t.cash_from_operations) / avg_assets
        assert cfq.accrual_ratio == pytest.approx(expected)

    def test_negative_accruals_quality_is_alta(self):
        # CFO >> NI → negative accruals → "Alta"
        fd_t_high_cfo = _make_fd(net_income=10_000, cfo=200_000)
        fd_t1_ = _make_fd()
        ms = BeneishMScore(current=fd_t_high_cfo, prior=fd_t1_).calculate()
        cfq = CashFlowQuality(current=fd_t_high_cfo, prior=fd_t1_).calculate(ms)
        assert cfq.earnings_quality == "Alta"

    def test_high_accruals_manipulator_is_critical(self):
        # NI >> CFO → high accruals; manipulator M-Score
        fd_high = _make_fd(net_income=500_000, cfo=10_000, revenues=1_500_000)
        fd_low  = _make_fd()
        ms = BeneishMScore(current=fd_high, prior=fd_low).calculate()
        cfq = CashFlowQuality(current=fd_high, prior=fd_low).calculate(ms)
        # Accruals = (500k - 10k) / avg_assets ≫ 5%
        if ms.is_manipulator:
            assert cfq.alert_level == AlertLevel.CRITICAL

    def test_to_dict_has_required_keys(self, fd_t, fd_t1, mscore_result):
        cfq = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore_result)
        d = cfq.to_dict()
        assert "Accrual Ratio" in d
        assert "Earnings Quality" in d
        assert "Alert Level" in d

    def test_str_representation(self, fd_t, fd_t1, mscore_result):
        cfq = CashFlowQuality(current=fd_t, prior=fd_t1).calculate(mscore_result)
        s = str(cfq)
        assert "Accrual Ratio" in s
        assert "Qualidade" in s


class TestAlertLevelOrdering:
    def test_critical_has_highest_rank(self):
        assert AlertLevel.CRITICAL.rank > AlertLevel.HIGH_RISK.rank
        assert AlertLevel.HIGH_RISK.rank > AlertLevel.WATCH.rank
        assert AlertLevel.WATCH.rank > AlertLevel.NORMAL.rank

    def test_stars_assigned(self):
        assert "★★★" in AlertLevel.CRITICAL.stars
        assert "★★" in AlertLevel.HIGH_RISK.stars
        assert "★" in AlertLevel.WATCH.stars
        assert "✓" in AlertLevel.NORMAL.stars


# ---------------------------------------------------------------------------
# rank_market — red flags
# ---------------------------------------------------------------------------

class TestDetectRedFlags:
    def _ms_with(self, **kwargs) -> MScoreResult:
        defaults = dict(dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
                        depi=1.0, sgai=1.0, lvgi=1.0, tata=0.0, m_score=-2.5)
        defaults.update(kwargs)
        return MScoreResult(**defaults)

    def test_returns_list(self, mscore_result):
        flags = detect_red_flags(mscore_result)
        assert isinstance(flags, list)

    def test_returns_at_most_top_n(self, mscore_result):
        flags = detect_red_flags(mscore_result, top_n=3)
        assert len(flags) <= 3

    def test_high_dsri_triggers_flag(self):
        ms = self._ms_with(dsri=2.0)
        flags = detect_red_flags(ms, top_n=8)
        assert any("DSRI" in f for f in flags)

    def test_high_tata_triggers_flag(self):
        ms = self._ms_with(tata=0.10)
        flags = detect_red_flags(ms, top_n=8)
        assert any("TATA" in f for f in flags)

    def test_neutral_indices_produce_no_flags(self):
        ms = self._ms_with(dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
                            depi=1.0, sgai=1.0, lvgi=1.0, tata=-0.05)
        flags = detect_red_flags(ms, top_n=3)
        assert flags == []

    def test_tata_most_impactful_flag_appears_first(self):
        # TATA coefficient 4.679 >> others; with high TATA it should rank #1
        ms = self._ms_with(tata=0.10, dsri=1.50, gmi=1.20)
        flags = detect_red_flags(ms, top_n=3)
        assert flags and "TATA" in flags[0]

    def test_flag_contains_formatted_value(self):
        ms = self._ms_with(dsri=1.82)
        flags = detect_red_flags(ms, top_n=8)
        dsri_flags = [f for f in flags if "DSRI" in f]
        assert dsri_flags
        assert "1.82" in dsri_flags[0]


# ---------------------------------------------------------------------------
# rank_market — _apply_sector_stats and _to_dataframe
# ---------------------------------------------------------------------------

def _make_company_result(ticker, sector, m_score, accrual_ratio=0.02, error=None):
    if error:
        return CompanyResult(
            ticker=ticker, sector=sector, year_t=2023,
            mscore=None, cfq=None, red_flags=[], error=error,
        )
    ms = MScoreResult(
        dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
        depi=1.0, sgai=1.0, lvgi=1.0, tata=accrual_ratio,
        m_score=m_score,
    )
    quality = "Alta" if accrual_ratio < 0.01 else ("Moderada" if accrual_ratio < 0.05 else "Baixa")
    from advisor_brain_fsa.accruals import AlertLevel, CashFlowQualityResult
    alert = AlertLevel.CRITICAL if (ms.is_manipulator and quality == "Baixa") \
        else AlertLevel.HIGH_RISK if ms.is_manipulator \
        else AlertLevel.WATCH if quality == "Baixa" \
        else AlertLevel.NORMAL
    cfq = CashFlowQualityResult(
        accrual_ratio=accrual_ratio,
        earnings_quality=quality,
        alert_level=alert,
    )
    return CompanyResult(
        ticker=ticker, sector=sector, year_t=2023,
        mscore=ms, cfq=cfq, red_flags=["Flag A", "Flag B"],
    )


class TestApplySectorStats:
    def test_sector_avg_computed_correctly(self):
        results = [
            _make_company_result("PETR4", "Energia", -1.5),
            _make_company_result("CSAN3", "Energia", -2.5),
            _make_company_result("VALE3", "Mineração", -3.0),
        ]
        _apply_sector_stats(results)
        energia_avg = (-1.5 + -2.5) / 2
        assert results[0].sector_avg_mscore == pytest.approx(energia_avg)
        assert results[2].sector_avg_mscore == pytest.approx(-3.0)

    def test_mscore_vs_sector_computed(self):
        results = [
            _make_company_result("PETR4", "Energia", -1.5),
            _make_company_result("CSAN3", "Energia", -2.5),
        ]
        _apply_sector_stats(results)
        avg = (-1.5 + -2.5) / 2  # = -2.0
        assert results[0].mscore_vs_sector == pytest.approx(-1.5 - avg)
        assert results[1].mscore_vs_sector == pytest.approx(-2.5 - avg)

    def test_failed_results_excluded_from_avg(self):
        results = [
            _make_company_result("PETR4", "Energia", -1.5),
            _make_company_result("CSAN3", "Energia", 0.0, error="not found"),
        ]
        _apply_sector_stats(results)
        assert results[0].sector_avg_mscore == pytest.approx(-1.5)
        assert math.isnan(results[1].sector_avg_mscore)


class TestToDataframe:
    def test_returns_dataframe(self):
        results = [_make_company_result("PETR4", "Energia", -1.5)]
        df = _to_dataframe(results, top_flags=3)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        results = [_make_company_result("PETR4", "Energia", -1.5)]
        df = _to_dataframe(results, top_flags=3)
        for col in ["Ticker", "Setor", "M-Score", "Nível de Alerta",
                    "Accrual Ratio", "DSRI", "TATA", "Red Flag 1"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_by_alert_desc_then_mscore_desc(self):
        results = [
            _make_company_result("A", "Energia", -2.5),           # Normal
            _make_company_result("B", "Energia", -1.0),           # High Risk
            _make_company_result("C", "Energia", -1.2, accrual_ratio=0.07),  # Critical
        ]
        _apply_sector_stats(results)
        df = _to_dataframe(results, top_flags=3)
        alert_levels = df["Nível de Alerta"].tolist()
        # Crítico or Alto Risco should come before Normal
        normal_idx = next(i for i, v in enumerate(alert_levels) if v == "Normal")
        for i in range(normal_idx):
            assert alert_levels[i] in ("Crítico", "Alto Risco", "Atenção")

    def test_error_row_has_nan_mscore(self):
        results = [_make_company_result("PETR4", "Energia", 0.0, error="not found")]
        df = _to_dataframe(results, top_flags=3)
        assert math.isnan(df.iloc[0]["M-Score"])

    def test_red_flag_columns_filled(self):
        results = [_make_company_result("PETR4", "Energia", -1.5)]
        df = _to_dataframe(results, top_flags=3)
        assert df.iloc[0]["Red Flag 1"] == "Flag A"
        assert df.iloc[0]["Red Flag 2"] == "Flag B"


# ---------------------------------------------------------------------------
# report_generator
# ---------------------------------------------------------------------------

class TestBuildMarkdown:
    def _sample_df(self) -> pd.DataFrame:
        results = [
            _make_company_result("PETR4", "Energia", -1.0, accrual_ratio=0.07),
            _make_company_result("VALE3", "Mineração", -2.5, accrual_ratio=0.005),
            _make_company_result("XXXX3", "Outros", 0.0, error="not found"),
        ]
        _apply_sector_stats(results)
        return _to_dataframe(results, top_flags=3)

    def test_returns_string(self):
        df = self._sample_df()
        md = build_markdown(df, year_t=2023)
        assert isinstance(md, str) and len(md) > 100

    def test_contains_header(self):
        df = self._sample_df()
        md = build_markdown(df, year_t=2023)
        assert "Ranking de Risco" in md
        assert "2023" in md

    def test_contains_beneish_reference(self):
        md = build_markdown(self._sample_df())
        assert "DSRI" in md
        assert "TATA" in md
        assert "−1.78" in md

    def test_contains_sector_section(self):
        md = build_markdown(self._sample_df())
        assert "Normalização Setorial" in md

    def test_contains_legend(self):
        md = build_markdown(self._sample_df())
        assert "Legenda" in md
        assert "Crítico" in md

    def test_failed_tickers_section_present(self):
        md = build_markdown(self._sample_df())
        assert "Indisponíveis" in md or "N/D" in md

    def test_executive_summary_counts(self):
        md = build_markdown(self._sample_df())
        assert "Resumo Executivo" in md


class TestGenerateReport:
    def test_writes_md_and_csv(self, tmp_path):
        results = [_make_company_result("PETR4", "Energia", -1.5)]
        _apply_sector_stats(results)
        df = _to_dataframe(results, top_flags=3)
        md_path, csv_path = generate_report(df, output_dir=tmp_path, year_t=2023)
        assert md_path.exists()
        assert csv_path.exists()

    def test_csv_has_expected_columns(self, tmp_path):
        results = [_make_company_result("PETR4", "Energia", -1.5)]
        _apply_sector_stats(results)
        df = _to_dataframe(results, top_flags=3)
        _, csv_path = generate_report(df, output_dir=tmp_path, year_t=2023)
        loaded = pd.read_csv(csv_path)
        assert "Ticker" in loaded.columns
        assert "M-Score" in loaded.columns

    def test_creates_output_dir_if_absent(self, tmp_path):
        new_dir = tmp_path / "subdir" / "reports"
        results = [_make_company_result("VALE3", "Mineração", -2.5)]
        _apply_sector_stats(results)
        df = _to_dataframe(results, top_flags=3)
        generate_report(df, output_dir=new_dir, year_t=2023)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# rank_market — integration with mocked CVMDataFetcher
# ---------------------------------------------------------------------------

class TestRankMarketIntegration:
    def _mock_fetcher(self, fd_t, fd_t1):
        """Return a CVMDataFetcher mock that returns fixed FinancialData."""
        mock = MagicMock()
        mock.get_financial_data.return_value = (fd_t, fd_t1)
        return mock

    def test_rank_market_returns_dataframe(self, fd_t, fd_t1):
        with patch(
            "advisor_brain_fsa.rank_market.CVMDataFetcher"
        ) as MockFetcher:
            instance = MockFetcher.return_value
            instance.get_financial_data.return_value = (fd_t, fd_t1)

            df = rank_market(
                tickers=["PETR4", "VALE3"],
                year_t=2023, year_t1=2022,
                retry_delay=0,
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_failed_ticker_appears_in_dataframe(self, fd_t, fd_t1):
        call_count = 0

        def side_effect(query, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Company not found")
            return (fd_t, fd_t1)

        with patch("advisor_brain_fsa.rank_market.CVMDataFetcher") as MockFetcher:
            instance = MockFetcher.return_value
            instance.get_financial_data.side_effect = side_effect

            df = rank_market(
                tickers=["UNKNWN", "VALE3"],
                year_t=2023, year_t1=2022,
                retry_delay=0,
            )

        assert len(df) == 2
        error_rows = df[df["Erro"] != ""]
        assert len(error_rows) == 1
        assert error_rows.iloc[0]["Ticker"] == "UNKNWN"

    def test_dataframe_sorted_by_risk(self, fd_t, fd_t1):
        """Highest-risk companies should appear at the top."""
        # Make one ticker return high accruals (bad) and one return low
        fd_bad = _make_fd(net_income=500_000, cfo=1_000)  # very high accruals

        call_map = {"PETR4": (fd_bad, fd_t1), "VALE3": (fd_t, fd_t1)}

        def side_effect(query, **kwargs):
            return call_map[query]

        with patch("advisor_brain_fsa.rank_market.CVMDataFetcher") as MockFetcher:
            instance = MockFetcher.return_value
            instance.get_financial_data.side_effect = side_effect
            df = rank_market(tickers=["PETR4", "VALE3"], year_t=2023, retry_delay=0)

        ok = df[df["Erro"] == ""]
        if len(ok) == 2:
            # First row should be at least as risky as the second
            alert_ranks = {
                "Crítico": 3, "Alto Risco": 2, "Atenção": 1, "Normal": 0
            }
            rank_first  = alert_ranks.get(ok.iloc[0]["Nível de Alerta"], -1)
            rank_second = alert_ranks.get(ok.iloc[1]["Nível de Alerta"], -1)
            assert rank_first >= rank_second
