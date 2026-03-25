"""
Unit tests for data_fetcher, cvm_accounts, and ticker_map.

All network calls are mocked — no real HTTP requests are made.
"""

from __future__ import annotations

import io
import os
import sys
import unicodedata
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.ticker_map import resolve_company, TICKER_TO_KEYWORD
from advisor_brain_fsa.cvm_accounts import ACCOUNT_SPECS, SPEC_BY_FIELD, REQUIRED_FIELDS
from advisor_brain_fsa.data_fetcher import CVMDataFetcher, fetch_data, _normalise
from advisor_brain_fsa.beneish_mscore import FinancialData


# ---------------------------------------------------------------------------
# Helpers — build fake CVM CSV content
# ---------------------------------------------------------------------------

_COLUMNS = [
    "CNPJ_CIA", "DENOM_CIA", "CD_CVM", "VERSAO",
    "DT_REFER", "DT_INI_EXERC", "DT_FIM_EXERC",
    "CD_CONTA", "DS_CONTA", "VL_CONTA", "ST_CONTA_FIXA",
]


def _make_rows(company: str, year: int, accounts: dict[str, float]) -> list[dict]:
    """Generate one row per account code for a given company/year."""
    rows = []
    for code, value in accounts.items():
        rows.append({
            "CNPJ_CIA": "33.000.167/0001-01",
            "DENOM_CIA": company,
            "CD_CVM": "9512",
            "VERSAO": 1,
            "DT_REFER": f"{year}-12-31",
            "DT_INI_EXERC": f"{year}-01-01",
            "DT_FIM_EXERC": f"{year}-12-31",
            "CD_CONTA": code,
            "DS_CONTA": f"Conta {code}",
            "VL_CONTA": value,
            "ST_CONTA_FIXA": "S",
        })
    return rows


def _make_csv_bytes(rows: list[dict]) -> bytes:
    df = pd.DataFrame(rows, columns=_COLUMNS)
    return df.to_csv(sep=";", index=False, encoding="latin-1").encode("latin-1")


def _make_fake_zip(statement_csvs: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP containing multiple named CSVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in statement_csvs.items():
            zf.writestr(name, content)
    return buf.getvalue()


# Minimal account values that satisfy all 13 FinancialData fields.
_ACCOUNTS_T = {
    # DRE
    "3.01":  1_200_000.0,    # revenues
    "3.02":  -720_000.0,     # COGS (negative in CVM)
    "3.04.02": -120_000.0,   # SGA (negative)
    "3.11":   120_000.0,     # net_income
    # BPA
    "1":      900_000.0,     # total_assets
    "1.01":   300_000.0,     # current_assets
    "1.01.06":100_000.0,     # receivables
    "1.02.03":400_000.0,     # pp_and_e
    "1.01.01.02": 50_000.0,  # securities
    # BPP
    "2.01":    90_000.0,     # current_liabilities
    "2.02.01":180_000.0,     # long_term_debt
    # DFC_MI
    "6.01":   150_000.0,     # cash_from_operations
    "6.01.01.02": -40_000.0, # depreciation (negative add-back)
}

_ACCOUNTS_T1 = {
    "3.01":  1_000_000.0,
    "3.02":  -600_000.0,
    "3.04.02": -100_000.0,
    "3.11":   100_000.0,
    "1":      800_000.0,
    "1.01":   260_000.0,
    "1.01.06": 80_000.0,
    "1.02.03":360_000.0,
    "1.01.01.02": 40_000.0,
    "2.01":    80_000.0,
    "2.02.01":160_000.0,
    "6.01":   130_000.0,
    "6.01.01.02": -36_000.0,
}


def _build_fake_zip_for_year(year: int, accounts: dict[str, float]) -> bytes:
    company = "PETROBRAS"
    rows = _make_rows(company, year, accounts)
    csv_bytes = _make_csv_bytes(rows)
    return _make_fake_zip({
        f"dfp_cia_aberta_BPA_con_{year}.csv": csv_bytes,
        f"dfp_cia_aberta_BPP_con_{year}.csv": csv_bytes,
        f"dfp_cia_aberta_DRE_con_{year}.csv": csv_bytes,
        f"dfp_cia_aberta_DFC_MI_con_{year}.csv": csv_bytes,
    })


# ---------------------------------------------------------------------------
# ticker_map tests
# ---------------------------------------------------------------------------

class TestResolveCompany:
    def test_known_ticker_returns_keyword(self):
        kw, qt = resolve_company("PETR4")
        assert kw == "PETROBRAS"
        assert qt == "ticker"

    def test_case_insensitive_ticker(self):
        kw, qt = resolve_company("petr4")
        assert kw == "PETROBRAS"
        assert qt == "ticker"

    def test_cnpj_format_detected(self):
        kw, qt = resolve_company("33.000.167/0001-01")
        assert qt == "cnpj"
        assert "33" in kw

    def test_cnpj_without_formatting(self):
        _, qt = resolve_company("33000167000101")
        assert qt == "cnpj"

    def test_unknown_string_treated_as_name(self):
        kw, qt = resolve_company("Ambev")
        assert qt == "name"
        assert "AMBEV" in kw

    def test_all_tickers_have_keywords(self):
        for ticker, keyword in TICKER_TO_KEYWORD.items():
            assert isinstance(keyword, str) and len(keyword) > 0, (
                f"Ticker {ticker} has empty keyword"
            )

    def test_vale_ticker(self):
        kw, _ = resolve_company("VALE3")
        assert "VALE" in kw


# ---------------------------------------------------------------------------
# cvm_accounts tests
# ---------------------------------------------------------------------------

class TestCvmAccounts:
    def test_all_required_fields_have_specs(self):
        assert REQUIRED_FIELDS == {s.field_name for s in ACCOUNT_SPECS}

    def test_spec_by_field_complete(self):
        financial_data_fields = {
            "revenues", "cost_of_goods_sold", "sales_general_admin_expenses",
            "receivables", "total_assets", "current_assets", "pp_and_e",
            "securities", "total_long_term_debt", "current_liabilities",
            "depreciation", "net_income", "cash_from_operations",
        }
        assert financial_data_fields == set(SPEC_BY_FIELD.keys())

    def test_each_spec_has_at_least_one_code(self):
        for spec in ACCOUNT_SPECS:
            assert len(spec.codes) >= 1, f"{spec.field_name} has no codes"

    def test_sign_values_are_valid(self):
        for spec in ACCOUNT_SPECS:
            assert spec.sign in (1, -1), f"{spec.field_name} has invalid sign"

    def test_statements_are_known(self):
        valid = {"BPA", "BPP", "DRE", "DFC_MI"}
        for spec in ACCOUNT_SPECS:
            assert spec.statement in valid, f"{spec.field_name}: unknown statement {spec.statement}"


# ---------------------------------------------------------------------------
# CVMDataFetcher — unit tests with mocked HTTP
# ---------------------------------------------------------------------------

class TestCVMDataFetcherMocked:
    """All network I/O is mocked."""

    def _make_fetcher(self, tmp_path: Path) -> CVMDataFetcher:
        return CVMDataFetcher(cache_dir=tmp_path, force_download=False)

    def _patch_download(self, fetcher: CVMDataFetcher, year: int, zip_bytes: bytes):
        """Write fake zip bytes directly to the cache to simulate a download."""
        dest = fetcher._zip_path(year)
        dest.write_bytes(zip_bytes)

    def test_load_statement_reads_csv_from_zip(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        zip_bytes = _build_fake_zip_for_year(2023, _ACCOUNTS_T)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "DRE")
        assert not df.empty
        assert "CD_CONTA" in df.columns
        assert "VL_CONTA" in df.columns

    def test_filter_by_name_finds_petrobras(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        zip_bytes = _build_fake_zip_for_year(2023, _ACCOUNTS_T)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "DRE")
        filtered = fetcher._filter_company(df, "PETROBRAS", "ticker", 2023)
        assert not filtered.empty
        assert filtered["DENOM_CIA"].iloc[0] == "PETROBRAS"

    def test_filter_by_cnpj(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        zip_bytes = _build_fake_zip_for_year(2023, _ACCOUNTS_T)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "DRE")
        filtered = fetcher._filter_company(
            df, "33.000.167/0001-01", "cnpj", 2023
        )
        assert not filtered.empty

    def test_resolve_account_primary_code(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        zip_bytes = _build_fake_zip_for_year(2023, _ACCOUNTS_T)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "DRE")
        filtered = fetcher._filter_company(df, "PETROBRAS", "ticker", 2023)

        spec = SPEC_BY_FIELD["revenues"]
        value = fetcher._resolve_account(filtered, spec)
        assert value == pytest.approx(1_200_000.0)

    def test_resolve_account_sign_correction_for_cogs(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        zip_bytes = _build_fake_zip_for_year(2023, _ACCOUNTS_T)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "DRE")
        filtered = fetcher._filter_company(df, "PETROBRAS", "ticker", 2023)

        spec = SPEC_BY_FIELD["cost_of_goods_sold"]
        value = fetcher._resolve_account(filtered, spec)
        # CVM stores as -720_000; sign=-1 → corrected to +720_000
        assert value == pytest.approx(720_000.0)

    def test_resolve_account_fallback_code(self, tmp_path):
        """Primary code missing → should fall back to second candidate."""
        fetcher = self._make_fetcher(tmp_path)
        # Use 1.01.03 instead of 1.01.06 for receivables
        accounts = dict(_ACCOUNTS_T)
        del accounts["1.01.06"]
        accounts["1.01.03"] = 90_000.0
        zip_bytes = _build_fake_zip_for_year(2023, accounts)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "BPA")
        filtered = fetcher._filter_company(df, "PETROBRAS", "ticker", 2023)

        spec = SPEC_BY_FIELD["receivables"]
        value = fetcher._resolve_account(filtered, spec)
        assert value == pytest.approx(90_000.0)

    def test_resolve_account_returns_nan_when_all_codes_missing(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        # Accounts with no receivable code at all
        accounts = {k: v for k, v in _ACCOUNTS_T.items()
                    if k not in ("1.01.06", "1.01.03", "1.01.04")}
        zip_bytes = _build_fake_zip_for_year(2023, accounts)
        self._patch_download(fetcher, 2023, zip_bytes)

        df = fetcher._load_statement(2023, "BPA")
        filtered = fetcher._filter_company(df, "PETROBRAS", "ticker", 2023)

        spec = SPEC_BY_FIELD["receivables"]
        import math
        value = fetcher._resolve_account(filtered, spec)
        assert math.isnan(value)

    def test_build_financial_data_returns_financial_data(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        for yr, accs in [(2023, _ACCOUNTS_T), (2022, _ACCOUNTS_T1)]:
            self._patch_download(fetcher, yr, _build_fake_zip_for_year(yr, accs))

        fd = fetcher._build_financial_data("PETROBRAS", "ticker", 2023)
        assert isinstance(fd, FinancialData)
        assert fd.revenues == pytest.approx(1_200_000.0)
        assert fd.cost_of_goods_sold == pytest.approx(720_000.0)
        assert fd.net_income == pytest.approx(120_000.0)

    def test_get_financial_data_returns_tuple(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        for yr, accs in [(2023, _ACCOUNTS_T), (2022, _ACCOUNTS_T1)]:
            self._patch_download(fetcher, yr, _build_fake_zip_for_year(yr, accs))

        fd_t, fd_t1 = fetcher.get_financial_data("PETR4", year_t=2023, year_t1=2022)
        assert isinstance(fd_t, FinancialData)
        assert isinstance(fd_t1, FinancialData)

    def test_get_financial_data_different_years(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        for yr, accs in [(2023, _ACCOUNTS_T), (2022, _ACCOUNTS_T1)]:
            self._patch_download(fetcher, yr, _build_fake_zip_for_year(yr, accs))

        fd_t, fd_t1 = fetcher.get_financial_data("PETR4", year_t=2023, year_t1=2022)
        # T should have higher revenues than T-1
        assert fd_t.revenues > fd_t1.revenues

    def test_get_raw_dataframe_returns_dataframe(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        self._patch_download(fetcher, 2023, _build_fake_zip_for_year(2023, _ACCOUNTS_T))

        df = fetcher.get_raw_dataframe("PETR4", year=2023)
        assert isinstance(df, pd.DataFrame)
        assert "CD_CONTA" in df.columns
        assert "statement" in df.columns

    def test_full_pipeline_produces_valid_mscore(self, tmp_path):
        """End-to-end: fake data → fetch_data → BeneishMScore."""
        fetcher = self._make_fetcher(tmp_path)
        for yr, accs in [(2023, _ACCOUNTS_T), (2022, _ACCOUNTS_T1)]:
            self._patch_download(fetcher, yr, _build_fake_zip_for_year(yr, accs))

        fd_t, fd_t1 = fetcher.get_financial_data("PETR4", year_t=2023, year_t1=2022)

        from advisor_brain_fsa import BeneishMScore
        import math
        result = BeneishMScore(current=fd_t, prior=fd_t1).calculate()
        assert math.isfinite(result.m_score)
        assert result.classification in ("Non-Manipulator", "Potential Manipulator")

    def test_download_triggered_when_cache_empty(self, tmp_path):
        """_download_zip should be called when zip is not cached."""
        fetcher = self._make_fetcher(tmp_path)

        fake_zip = _build_fake_zip_for_year(2023, _ACCOUNTS_T)

        with patch.object(fetcher, "_download_zip", return_value=None) as mock_dl:
            # Pre-populate cache manually so _load_statement doesn't fail
            dest = fetcher._zip_path(2023)
            dest.write_bytes(fake_zip)
            mock_dl.return_value = dest

            fetcher._load_statement(2023, "DRE")
            # First call triggers the download
            mock_dl.assert_called_once_with(2023)

    def test_in_memory_cache_avoids_double_load(self, tmp_path):
        fetcher = self._make_fetcher(tmp_path)
        self._patch_download(fetcher, 2023, _build_fake_zip_for_year(2023, _ACCOUNTS_T))

        with patch.object(fetcher, "_download_zip", wraps=fetcher._download_zip) as spy:
            fetcher._load_statement(2023, "DRE")
            fetcher._load_statement(2023, "DRE")   # second call — should use cache
            assert spy.call_count == 1             # download called only once


# ---------------------------------------------------------------------------
# _normalise helper
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_strips_accents(self):
        assert _normalise("Petróleo") == "PETROLEO"

    def test_folds_to_upper(self):
        assert _normalise("petrobras") == "PETROBRAS"

    def test_collapses_whitespace(self):
        assert _normalise("  ABC   DEF  ") == "ABC DEF"
