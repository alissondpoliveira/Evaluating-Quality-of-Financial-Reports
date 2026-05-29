#!/usr/bin/env python3
"""
build_market_cache.py — Offline Batch Processor
================================================
Builds the static market ranking cache used by the Home Dashboard.

This script is meant to run OFFLINE (CLI, cron, Railway job) — never inside
gunicorn.  It:
  1. Downloads / refreshes the CVM Dados Cadastrais (cad_cia_aberta.csv).
  2. Extracts all active, B3-listed, non-financial companies via
     CVMRegistry.get_non_financial_df() (blacklists banks, insurers, etc.).
  3. For each company, downloads the CVM DFP financials (cached ZIP) and
     computes the Beneish M-Score + CFA Accruals defensively (try/except →
     NaN on missing data).
  4. Writes the consolidated results to:
       data/cache/market_ranking_current.json

Usage
-----
    # Full build (all non-financial CVM companies)
    python build_market_cache.py

    # Specify fiscal year and parallelism
    python build_market_cache.py --year 2023 --workers 4

    # Skip cache for CVM registry (force fresh download)
    python build_market_cache.py --refresh-registry

    # Dry-run: show company count, skip scoring
    python build_market_cache.py --dry-run

Output JSON schema
------------------
[
  {
    "Ticker":           "PETR4",          # known B3 ticker or DENOM abbreviation
    "Nome":             "PETROBRAS S.A.", # full DENOM_SOCIAL from CVM
    "CNPJ":             "33000167000101",
    "Setor":            "Energia",
    "Scorer":           "beneish",
    "Score de Risco":   7.23,
    "M-Score":          -1.45,
    "Classificação":    "Manipulador Provável",
    "Nível de Alerta":  "Alto Risco",
    "Accrual Ratio":    0.032,
    "Qualidade Earnings": "Baixa",
    "Red Flag 1":       "DSRI elevado ...",
    "Red Flag 2":       "...",
    "Red Flag 3":       "...",
    "Erro":             ""
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow running from project root without installing the package
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd

from advisor_brain_fsa.cvm_registry import CVMRegistry
from advisor_brain_fsa.data_fetcher import CVMDataFetcher
from advisor_brain_fsa.sector_scorer import BeneishSectorScorer
from advisor_brain_fsa.ticker_map import TICKER_TO_KEYWORD, TICKER_SECTOR
from advisor_brain_fsa.rank_market import detect_red_flags

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR     = _ROOT / "data" / "cache"
_OUTPUT_FILE   = _CACHE_DIR / "market_ranking_current.json"
_DEFAULT_YEAR  = date.today().year - 1
_RETRY_DELAY   = 0.2   # seconds between CVM DFP requests (be polite to the server)

logger = logging.getLogger("build_market_cache")


# ---------------------------------------------------------------------------
# Reverse ticker lookup
# ---------------------------------------------------------------------------

def _build_reverse_ticker_map() -> Dict[str, str]:
    """
    Build a normalised-DENOM → best ticker map from TICKER_TO_KEYWORD.

    Preference order (most informative first):
      PN shares ending in "4" or "11" > ON shares ending in "3" > other.
    """
    def _norm(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s)
        return nfkd.encode("ascii", "ignore").decode("ascii").upper().strip()

    _EXCLUDED = {"BDR", "Bancos", "Seguros", "Financeiro"}
    result: Dict[str, str] = {}

    for ticker, keyword in TICKER_TO_KEYWORD.items():
        if TICKER_SECTOR.get(ticker) in _EXCLUDED:
            continue
        kn = _norm(keyword)
        existing = result.get(kn, "")
        # Prefer tickers ending in "11" then "4" then "3", else first seen
        if (not existing
                or ticker.endswith("11") and not existing.endswith("11")
                or ticker.endswith("4")  and not existing.endswith(("4","11"))
                or ticker.endswith("3")  and not existing.endswith(("3","4","11"))):
            result[kn] = ticker

    return result


_DENOM_TO_TICKER: Dict[str, str] = _build_reverse_ticker_map()


def _ticker_for_denom(denom_social: str) -> str:
    """
    Return best known B3 ticker for a given DENOM_SOCIAL.
    Falls back to a 6-char abbreviation if no ticker is found.
    """
    def _norm(s: str) -> str:
        nfkd = unicodedata.normalize("NFKD", s)
        return nfkd.encode("ascii", "ignore").decode("ascii").upper().strip()

    dn = _norm(denom_social)
    # Try progressively shorter prefixes of each word
    for kw, tk in _DENOM_TO_TICKER.items():
        if kw in dn or dn in kw:
            return tk

    # Abbreviation fallback: first 4 capitalised letters + "3"
    letters = "".join(c for c in dn if c.isalpha())[:4]
    return letters if letters else "?????"


# ---------------------------------------------------------------------------
# Per-company scoring
# ---------------------------------------------------------------------------

def _score_company(
    cnpj: str,
    denom: str,
    sector: str,
    year_t: int,
    fetcher: CVMDataFetcher,
    scorer: BeneishSectorScorer,
) -> dict:
    """Score one company; returns a result dict regardless of success."""
    ticker = _ticker_for_denom(denom)
    base   = {
        "Ticker":             ticker,
        "Nome":               denom,
        "CNPJ":               cnpj,
        "Setor":              sector,
        "Scorer":             "beneish",
        "Score de Risco":     None,
        "M-Score":            None,
        "Classificação":      None,
        "Nível de Alerta":    None,
        "Accrual Ratio":      None,
        "Qualidade Earnings": None,
        "Red Flag 1":         "",
        "Red Flag 2":         "",
        "Red Flag 3":         "",
        "Erro":               "",
    }

    try:
        fd_t, fd_t1 = fetcher.get_financial_data(
            cnpj, year_t=year_t, year_t1=year_t - 1
        )
        sr    = scorer.score(fd_t, fd_t1)
        ms    = sr.mscore_result
        cfq   = sr.cfq_result
        flags = sr.red_flags + ["", "", ""]

        base.update({
            "Score de Risco":     round(sr.risk_score, 4),
            "M-Score":            round(ms.m_score,  4) if ms  else None,
            "Classificação":      sr.classification,
            "Nível de Alerta":    sr.alert_level.value,
            "Accrual Ratio":      round(cfq.accrual_ratio, 4) if cfq else None,
            "Qualidade Earnings": cfq.earnings_quality if cfq else None,
            "Red Flag 1":         flags[0],
            "Red Flag 2":         flags[1],
            "Red Flag 3":         flags[2],
        })
    except Exception as exc:
        base["Erro"] = str(exc)[:200]

    return base


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(
    year_t: int = _DEFAULT_YEAR,
    workers: int = 1,
    refresh_registry: bool = False,
    dry_run: bool = False,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Run the full batch scoring pipeline and write the JSON output.

    Parameters
    ----------
    year_t : int
        Fiscal year for Beneish M-Score calculation.
    workers : int
        Number of parallel threads for CVM DFP requests.
        Keep low (1–2) to avoid hammering the CVM server.
    refresh_registry : bool
        Force re-download of cad_cia_aberta.csv.
    dry_run : bool
        Print company count and exit without scoring.
    output_path : Path | None
        Override default output file path.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or _OUTPUT_FILE

    logger.info("=== build_market_cache START (year_t=%d, workers=%d) ===", year_t, workers)

    # ── Step 1: load CVM registry ─────────────────────────────────────────
    registry = CVMRegistry.get_instance(cache_dir=str(_CACHE_DIR))
    if refresh_registry:
        logger.info("Forcing registry refresh...")
        registry.refresh()

    # ── Step 2: get non-financial company list ────────────────────────────
    nf_df = registry.get_non_financial_df()
    logger.info("Non-financial companies found: %d", len(nf_df))

    if nf_df.empty:
        logger.warning("No non-financial companies found. Check CVM registry filters.")
        return out

    if dry_run:
        print(f"[dry-run] Would score {len(nf_df)} companies for year {year_t}.")
        print(f"[dry-run] Output would be written to: {out}")
        # Show first 10 as sample
        cols = ["DENOM_SOCIAL", "_SECTOR_LABEL", "_CNPJ_DIGITS"]
        available = [c for c in cols if c in nf_df.columns]
        print(nf_df[available].head(10).to_string(index=False))
        return out

    # Resolve CNPJ column
    cnpj_col = next(
        (c for c in ["_CNPJ_DIGITS", "CNPJ_CIA", "CNPJ"] if c in nf_df.columns), None
    )
    if cnpj_col is None:
        raise RuntimeError("Cannot find CNPJ column in non-financial DataFrame.")

    # ── Step 3: batch scoring ─────────────────────────────────────────────
    fetcher = CVMDataFetcher(cache_dir=str(_CACHE_DIR))
    scorer  = BeneishSectorScorer()
    total   = len(nf_df)
    results: List[dict] = []

    def _work(args):
        idx, row = args
        cnpj  = str(row.get(cnpj_col, "")).strip()
        denom = str(row.get("DENOM_SOCIAL", "")).strip()
        sector = str(row.get("_SECTOR_LABEL", "Outros")).strip()
        if not cnpj and not denom:
            return {"Ticker": "?", "Nome": denom, "CNPJ": cnpj,
                    "Setor": sector, "Scorer": "beneish",
                    "Score de Risco": None, "M-Score": None,
                    "Classificação": None, "Nível de Alerta": None,
                    "Accrual Ratio": None, "Qualidade Earnings": None,
                    "Red Flag 1": "", "Red Flag 2": "", "Red Flag 3": "",
                    "Erro": "CNPJ and name both empty"}
        query = cnpj if cnpj else denom
        result = _score_company(query, denom, sector, year_t, fetcher, scorer)
        if idx % 20 == 0 or idx == total - 1:
            ok  = "OK"  if not result["Erro"] else "ERR"
            logger.info("[%d/%d] %s %s — %s", idx + 1, total, ok, denom[:40], result["Erro"][:60] if result["Erro"] else "")
        time.sleep(_RETRY_DELAY)
        return result

    rows_iter = list(nf_df.iterrows())

    if workers <= 1:
        for idx, row in rows_iter:
            results.append(_work((idx, row)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_work, (idx, row)): idx for idx, row in rows_iter}
            for fut in as_completed(futures):
                results.append(fut.result())

    logger.info(
        "Scoring complete: %d total, %d OK, %d errors",
        len(results),
        sum(1 for r in results if not r.get("Erro")),
        sum(1 for r in results if r.get("Erro")),
    )

    # ── Step 4: sort and write JSON (atomic via temp file) ───────────────
    def _sort_key(r: dict):
        score = r.get("Score de Risco")
        return (0 if score is None else -score)  # descending risk score, errors last

    results.sort(key=_sort_key)

    # Write to a temp file alongside the output and rename atomically so
    # gunicorn never reads a partially-written file.
    tmp = out.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(out)  # atomic on POSIX; os.replace() fallback on Windows

    n_ok  = sum(1 for r in results if not r.get("Erro"))
    n_err = sum(1 for r in results if r.get("Erro"))
    logger.info(
        "=== build_market_cache DONE → %s  |  %d scored  |  %d errors ===",
        out, n_ok, n_err,
    )
    if n_err:
        logger.warning(
            "%d companies could not be scored (missing CVM data or parsing errors). "
            "Run with --log-level DEBUG to see details.",
            n_err,
        )
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the static market ranking cache for the Home Dashboard."
    )
    p.add_argument(
        "--year", type=int, default=_DEFAULT_YEAR,
        help=f"Fiscal year for M-Score calculation (default: {_DEFAULT_YEAR})",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel scoring threads (default: 1; keep ≤2 to respect CVM rate limits)",
    )
    p.add_argument(
        "--refresh-registry", action="store_true",
        help="Force re-download of cad_cia_aberta.csv before scoring",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print company count and exit without scoring",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help=f"Override output file path (default: {_OUTPUT_FILE})",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out = build(
        year_t=args.year,
        workers=args.workers,
        refresh_registry=args.refresh_registry,
        dry_run=args.dry_run,
        output_path=Path(args.output) if args.output else None,
    )

    if not args.dry_run:
        print(f"Cache written → {out}")
