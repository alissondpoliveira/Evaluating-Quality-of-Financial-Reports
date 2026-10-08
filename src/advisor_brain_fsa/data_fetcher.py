"""
CVM DFP Data Fetcher
---------------------
Downloads, caches, and parses the CVM Demonstrações Financeiras Padronizadas
(DFP) open-data files, then maps the accounting lines to the inputs required
by BeneishMScore.

Quick start
-----------
    from advisor_brain_fsa.data_fetcher import fetch_data

    data_t, data_t1 = fetch_data("PETR4")          # by B3 ticker
    data_t, data_t1 = fetch_data("33.000.167/0001-01")  # by CNPJ
    data_t, data_t1 = fetch_data("Petrobras")       # by name fragment

    from advisor_brain_fsa import BeneishMScore
    result = BeneishMScore(current=data_t, prior=data_t1).calculate()
    print(result)

Data source
-----------
Portal CVM Dados Abertos — DFP
  https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/

Zip file naming convention:
  dfp_cia_aberta_{year}.zip

CSV files inside:
  dfp_cia_aberta_BPA_con_{year}.csv  — Balanço Patrimonial Ativo
  dfp_cia_aberta_BPP_con_{year}.csv  — Balanço Patrimonial Passivo
  dfp_cia_aberta_DRE_con_{year}.csv  — Demonstração de Resultado
  dfp_cia_aberta_DFC_MI_con_{year}.csv — Fluxo de Caixa (Indireto)
"""

from __future__ import annotations

import io
import logging
import os
import unicodedata
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import urllib.request
import urllib.error

from .beneish_mscore import FinancialData
from .cvm_accounts import ACCOUNT_SPECS, AccountSpec, SPEC_BY_FIELD
from .ticker_map import resolve_company

# Re-export so callers can do: from advisor_brain_fsa.data_fetcher import fetch_cvm_company_registry
from .cvm_registry import fetch_cvm_company_registry as fetch_cvm_company_registry  # noqa: F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CVM_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS"
ZIP_NAME_TPL = "dfp_cia_aberta_{year}.zip"

# Statement suffix used in consolidated ("con") files; "ind" for standalone.
# Consolidated is the default. Companies without subsidiaries do not publish a
# consolidated DFP (or publish it with every account at zero, e.g. TIM S.A.);
# for them the individual statements cover the same entity, so they are used instead.
_MODALITY = "con"
_FALLBACK_MODALITY = "ind"

# Map our logical statement names to the file-name segment used by CVM.
_STATEMENT_FILE_MAP = {
    "BPA": "BPA",
    "BPP": "BPP",
    "DRE": "DRE",
    "DFC_MI": "DFC_MI",
}

# CVM CSV encoding and separator
_CSV_ENCODING = "latin-1"
_CSV_SEP = ";"

# Default cache directory
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "cache"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Fold to upper ASCII, strip accents and extra whitespace."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_str.upper().split())


def _build_zip_url(year: int) -> str:
    return f"{CVM_BASE_URL}/{ZIP_NAME_TPL.format(year=year)}"


def _csv_name_inside_zip(statement_key: str, year: int, modality: str = _MODALITY) -> str:
    suffix = _STATEMENT_FILE_MAP[statement_key]
    return f"dfp_cia_aberta_{suffix}_{modality}_{year}.csv"


# ---------------------------------------------------------------------------
# CVMDataFetcher
# ---------------------------------------------------------------------------

class CVMDataFetcher:
    """
    Downloads and caches CVM DFP ZIP archives, parses the relevant CSVs,
    filters by company, and resolves accounting lines to FinancialData objects.

    Parameters
    ----------
    cache_dir : Path | str | None
        Directory to store downloaded ZIP files.  Defaults to data/cache/.
    force_download : bool
        Re-download even if the ZIP is already cached.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        force_download: bool = False,
        timeout: int = 60,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.force_download = force_download
        self.timeout = timeout
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: (year, statement_key) → raw DataFrame
        self._df_cache: Dict[Tuple[int, str], pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_financial_data(
        self,
        query: str,
        year_t: Optional[int] = None,
        year_t1: Optional[int] = None,
    ) -> Tuple[FinancialData, FinancialData]:
        """
        Return (FinancialData_T, FinancialData_T1) ready for BeneishMScore.

        Parameters
        ----------
        query : str
            B3 ticker, CNPJ, or company name fragment.
        year_t : int | None
            Current period year.  Defaults to last calendar year.
        year_t1 : int | None
            Prior period year.  Defaults to year_t - 1.

        Returns
        -------
        Tuple[FinancialData, FinancialData]
        """
        current_year = date.today().year
        year_t = year_t or (current_year - 1)
        year_t1 = year_t1 or (year_t - 1)

        keyword, query_type = resolve_company(query)
        logger.info("Query='%s' → keyword='%s' (type=%s)", query, keyword, query_type)

        fd_t = self._build_financial_data(keyword, query_type, year_t)
        fd_t1 = self._build_financial_data(keyword, query_type, year_t1)
        return fd_t, fd_t1

    def get_raw_dataframe(
        self,
        query: str,
        year: int,
        statements: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return the raw filtered DataFrame for a company/year, merging the
        requested statement types.

        Useful for exploratory analysis before committing to FinancialData.

        Parameters
        ----------
        query : str
            Ticker, CNPJ, or name fragment.
        year : int
            Reference year (DT_FIM_EXERC year).
        statements : list[str] | None
            Subset of ["BPA", "BPP", "DRE", "DFC_MI"].
            Defaults to all four.
        """
        keyword, query_type = resolve_company(query)
        stmts = statements or list(_STATEMENT_FILE_MAP.keys())
        parts = []
        for stmt in stmts:
            df = self._load_statement(year, stmt)
            filtered = self._filter_company(df, keyword, query_type, year)
            filtered = filtered.copy()
            filtered["statement"] = stmt
            parts.append(filtered)
        if not parts:
            return pd.DataFrame()
        merged = pd.concat(parts, ignore_index=True)
        # Keep only the most recent restatement version per account
        merged = (
            merged.sort_values("VERSAO")
            .drop_duplicates(subset=["CD_CONTA", "DT_FIM_EXERC"], keep="last")
            .reset_index(drop=True)
        )
        return merged

    # ------------------------------------------------------------------
    # Download / cache layer
    # ------------------------------------------------------------------

    def _zip_path(self, year: int) -> Path:
        return self.cache_dir / ZIP_NAME_TPL.format(year=year)

    def _download_zip(self, year: int) -> Path:
        url = _build_zip_url(year)
        dest = self._zip_path(year)

        if dest.exists() and not self.force_download:
            logger.info("Cache hit: %s", dest)
            return dest

        logger.info("Downloading %s → %s", url, dest)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "advisor-brain-fsa/1.0 (financial research)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
            dest.write_bytes(data)
            logger.info("Saved %d bytes to %s", len(data), dest)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"CVM server returned HTTP {exc.code} for year {year}. "
                f"Check if the DFP for {year} has been published at {CVM_BASE_URL}/"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Network error downloading CVM data for year {year}: {exc.reason}"
            ) from exc
        return dest

    # ------------------------------------------------------------------
    # Statement loader
    # ------------------------------------------------------------------

    def _load_statement(self, year: int, statement_key: str, modality: str = _MODALITY) -> pd.DataFrame:
        """Load one statement CSV from the ZIP, with in-memory caching."""
        cache_key = (year, statement_key, modality)
        if cache_key in self._df_cache:
            return self._df_cache[cache_key]

        zip_path = self._download_zip(year)
        csv_name = _csv_name_inside_zip(statement_key, year, modality)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                available = zf.namelist()
                # Find case-insensitive match
                matches = [n for n in available if n.lower() == csv_name.lower()]
                if not matches:
                    raise FileNotFoundError(
                        f"'{csv_name}' not found in {zip_path.name}. "
                        f"Available: {available}"
                    )
                with zf.open(matches[0]) as f:
                    raw_bytes = f.read()

        except zipfile.BadZipFile as exc:
            # Possibly a corrupt/partial download — remove and let caller retry.
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Corrupt ZIP for year {year}. Deleted cache, please retry."
            ) from exc

        df = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep=_CSV_SEP,
            encoding=_CSV_ENCODING,
            dtype={"CD_CONTA": str, "CNPJ_CIA": str},
            low_memory=False,
        )

        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Parse date columns
        for col in ("DT_REFER", "DT_INI_EXERC", "DT_FIM_EXERC"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Ensure VL_CONTA is numeric
        if "VL_CONTA" in df.columns:
            df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")

        self._df_cache[cache_key] = df
        logger.info("Loaded %s (%d rows) for year %d", statement_key, len(df), year)
        return df

    # ------------------------------------------------------------------
    # Company filter
    # ------------------------------------------------------------------

    def _filter_company(
        self,
        df: pd.DataFrame,
        keyword: str,
        query_type: str,
        year: int,
    ) -> pd.DataFrame:
        """Return rows for the target company in the given fiscal year."""
        # Filter by fiscal year-end
        mask_year = df["DT_FIM_EXERC"].dt.year == year

        if query_type == "cnpj":
            cnpj_clean = keyword.replace(".", "").replace("/", "").replace("-", "")
            mask_company = df["CNPJ_CIA"].str.replace(r"\D", "", regex=True) == cnpj_clean
        else:
            # Name-based search (normalise both sides)
            df_names = df["DENOM_CIA"].apply(_normalise)
            kw_norm = _normalise(keyword)
            mask_company = df_names.str.contains(kw_norm, regex=False, na=False)

        filtered = df[mask_year & mask_company].copy()

        if filtered.empty:
            # Second attempt: relax year filter (some filings are submitted the
            # following year — use latest available fiscal year close to target).
            logger.warning(
                "No data for year %d with keyword='%s'. Trying adjacent years.",
                year, keyword,
            )
            mask_company_only = (
                df["DENOM_CIA"].apply(_normalise).str.contains(
                    _normalise(keyword), regex=False, na=False
                )
                if query_type != "cnpj" else
                df["CNPJ_CIA"].str.replace(r"\D", "", regex=True) == keyword.replace(".", "").replace("/", "").replace("-", "")
            )
            candidates = df[mask_company_only]
            if candidates.empty:
                raise ValueError(
                    f"Company not found for query '{keyword}' in any available year. "
                    "Check the ticker/name/CNPJ or verify the DFP data was published."
                )
            available_years = candidates["DT_FIM_EXERC"].dt.year.dropna().unique()
            closest = min(available_years, key=lambda y: abs(y - year))
            logger.warning("Falling back to fiscal year %d for '%s'.", closest, keyword)
            filtered = candidates[candidates["DT_FIM_EXERC"].dt.year == closest].copy()

        # Keep only the latest restatement version
        if "VERSAO" in filtered.columns:
            latest_version = filtered["VERSAO"].max()
            filtered = filtered[filtered["VERSAO"] == latest_version]

        # Keep only annual (LAST_DRE or LAST or the longest-period entries)
        # Some DFPs include interim quarters — keep only the full-year row.
        if "DT_INI_EXERC" in filtered.columns and "DT_FIM_EXERC" in filtered.columns:
            filtered["_period_days"] = (
                filtered["DT_FIM_EXERC"] - filtered["DT_INI_EXERC"]
            ).dt.days
            max_days = filtered["_period_days"].max()
            # Allow 330–370 days (annual window)
            if max_days and max_days >= 300:
                filtered = filtered[filtered["_period_days"] >= 330]
            filtered = filtered.drop(columns=["_period_days"])

        return filtered.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Account resolution
    # ------------------------------------------------------------------

    def _resolve_account(
        self,
        filtered_df: pd.DataFrame,
        spec: AccountSpec,
    ) -> float:
        """
        Try each code in spec.codes in order.  Returns the first match,
        applying the sign correction defined in the spec.
        Falls back to NaN if nothing matches.

        Sign convention
        ---------------
        ``spec.sign == -1`` means the field must always be a **positive absolute
        value** regardless of how the CVM file stores it.  CVM DRE/DFC files are
        inconsistent: some companies store COGS / D&A as negative (expense
        convention), others as positive (add-back convention in DFC indirect
        method).  Using ``abs()`` instead of ``value * -1`` handles both cases.
        """
        if spec.desc_pattern and not filtered_df.empty:
            desc = filtered_df["DS_CONTA"].astype(str).map(
                lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower())
            mask = (filtered_df["CD_CONTA"].astype(str).str.match(spec.desc_codes)
                    & desc.str.contains(spec.desc_pattern, regex=True))
            if spec.desc_exclude:
                mask &= ~desc.str.contains(spec.desc_exclude, regex=True)
            values = filtered_df.loc[mask, "VL_CONTA"].dropna().astype(float)
            if not values.empty:
                total = float(values.abs().sum()) if spec.sign == -1 else float(values.sum())
                logger.debug("  %-35s by description (%d lines) = %14.2f", spec.field_name, len(values), total)
                return total

        if spec.sum_codes:
            found = []
            for code in spec.codes:
                rows = filtered_df[filtered_df["CD_CONTA"] == code]
                if not rows.empty and not pd.isna(rows["VL_CONTA"].iloc[0]):
                    found.append(float(rows["VL_CONTA"].iloc[0]))
            if found:
                return float(sum(abs(v) for v in found)) if spec.sign == -1 else float(sum(found))

        for code in spec.codes:
            rows = filtered_df[filtered_df["CD_CONTA"] == code]
            if not rows.empty:
                value = rows["VL_CONTA"].iloc[0]
                if pd.isna(value):
                    continue
                raw = float(value)
                # For expense/cost fields (sign=-1) always take the absolute value
                # so the FinancialData contract (positive values) is guaranteed
                # regardless of CVM sign convention.
                corrected = abs(raw) if spec.sign == -1 else raw
                logger.debug(
                    "  %-35s code=%-12s raw=%14.2f  corrected=%14.2f",
                    spec.field_name, code, raw, corrected,
                )
                return corrected

        logger.warning(
            "Account '%s' not resolved for codes %s — using NaN.",
            spec.field_name, spec.codes,
        )
        return float("nan")

    # ------------------------------------------------------------------
    # FinancialData builder
    # ------------------------------------------------------------------

    def _build_financial_data(
        self, keyword: str, query_type: str, year: int
    ) -> FinancialData:
        """Load all statements and assemble one FinancialData for `year`."""
        # Cache loaded statement DataFrames for this call
        def carregar(modality: str) -> Dict[str, pd.DataFrame]:
            return {k: self._filter_company(self._load_statement(year, k, modality), keyword, query_type, year)
                    for k in _STATEMENT_FILE_MAP}

        def tem_valores(dfs: Dict[str, pd.DataFrame]) -> bool:
            bpa = dfs.get("BPA", pd.DataFrame())
            if bpa.empty:
                return False
            ativo = bpa.loc[bpa["CD_CONTA"] == "1", "VL_CONTA"].dropna()
            return bool((ativo != 0).any())

        erro_con: Optional[Exception] = None
        try:
            stmt_dfs = carregar(_MODALITY)
        except ValueError as exc:  # company absent from the consolidated file
            erro_con, stmt_dfs = exc, None
        if stmt_dfs is None or not tem_valores(stmt_dfs):
            try:
                ind = carregar(_FALLBACK_MODALITY)
            except (ValueError, FileNotFoundError):
                ind = None
            if ind is not None and tem_valores(ind):
                logger.warning("Year %d — consolidated statements empty for '%s'; using individual statements.",
                               year, keyword)
                stmt_dfs = ind
            elif stmt_dfs is None:
                raise erro_con

        resolved: Dict[str, float] = {}
        for spec in ACCOUNT_SPECS:
            df_for_stmt = stmt_dfs.get(spec.statement, pd.DataFrame())
            resolved[spec.field_name] = self._resolve_account(df_for_stmt, spec)

        # Report any missing fields before raising
        missing = [k for k, v in resolved.items() if np.isnan(v)]
        if missing:
            logger.warning(
                "Year %d — could not resolve: %s. Will substitute 0.0.", year, missing
            )
            for k in missing:
                resolved[k] = 0.0

        return FinancialData(**resolved)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def fetch_data(
    query: str,
    year_t: Optional[int] = None,
    year_t1: Optional[int] = None,
    cache_dir: Optional[Path | str] = None,
    force_download: bool = False,
) -> Tuple[FinancialData, FinancialData]:
    """
    One-call entry point: download CVM data and return (T, T-1) FinancialData.

    Parameters
    ----------
    query : str
        B3 ticker (e.g. "PETR4"), CNPJ, or company name fragment.
    year_t : int | None
        Current fiscal year.  Defaults to last completed calendar year.
    year_t1 : int | None
        Prior fiscal year.  Defaults to year_t - 1.
    cache_dir : Path | str | None
        Local directory to cache the CVM ZIP files.
    force_download : bool
        Skip cache and re-download.

    Returns
    -------
    (FinancialData, FinancialData)
        Current period (T) and prior period (T-1), ready for BeneishMScore.

    Example
    -------
    >>> from advisor_brain_fsa.data_fetcher import fetch_data
    >>> from advisor_brain_fsa import BeneishMScore
    >>> data_t, data_t1 = fetch_data("PETR4")
    >>> result = BeneishMScore(current=data_t, prior=data_t1).calculate()
    >>> print(result)
    """
    fetcher = CVMDataFetcher(
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return fetcher.get_financial_data(query, year_t=year_t, year_t1=year_t1)
