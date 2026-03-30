"""
CVM Cadastral Registry
-----------------------
Tarefa 1 — Ingestão de Dados Cadastrais:
    fetch_cvm_company_registry() baixa o CSV de Dados Cadastrais de
    Companhias Abertas do Portal CVM e armazena em cache local.

Tarefa 2 — Classificação Automática por Scorer:
    classify_setor_ativ() lê a coluna SETOR_ATIV e devolve
    (scorer_type, sector_label) usando regras de substring.

Tarefa 3 — Integração com Tickers (B3):
    CVMRegistry.resolve_ticker_sector() cruza TICKER_TO_KEYWORD com
    DENOM_SOCIAL via busca aproximada por palavras para resolver o setor
    de tickers desconhecidos em tempo real.

Data source:
    https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .sector_scorer import BeneishSectorScorer, SectorScorer
from .ticker_map import TICKER_TO_KEYWORD, TICKER_SECTOR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CVM_CAD_URL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
)
_CACHE_FILENAME = "cad_cia_aberta.csv"
_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "cache"
_CSV_ENCODING = "latin-1"
_CSV_SEP = ";"

# ---------------------------------------------------------------------------
# Blacklist de setores financeiros — excluídos do ranking Beneish
# Palavras normalizadas (uppercase ASCII, sem acentos) presentes em SETOR_ATIV
# que classificam a empresa como instituição financeira.
# ---------------------------------------------------------------------------

FINANCIAL_BLACKLIST_KEYWORDS: frozenset[str] = frozenset({
    "BANCO",
    "BANCOS",
    "SEGUR",       # SEGURO, SEGUROS, SEGURADORA
    "RESSEGUR",    # RESSEGURO, RESSEGURADORA
    "PREVID",      # PREVIDENCIA, PREVIDÊNCIA
    "CAPITALIZACAO",
    "INTERMEDIACAO FINANCEIRA",
    "INTERMEDIACAO",
    "ARRENDAMENTO",
    "SECURITIZACAO",
    "CREDITO",
    "FINANCIAMENTO",
    "LEASING",
    "CAMBIO",
    "FINANCEI",    # SERVIÇOS FINANCEIROS, INTERMEDIAÇÃO FINANCEIRA
    "HOLDING FINANCEIR",
    "FUNDO DE INVESTIMENT",
    "CORRETORA",
    "DISTRIBUIDORA DE VALORES",
})


# ---------------------------------------------------------------------------
# Tarefa 2 — Regras SETOR_ATIV → (scorer_type, sector_label)
# Ordem: mais específico → mais geral. Primeiro match vence.
# ---------------------------------------------------------------------------

_SETOR_RULES: List[Tuple[Tuple[str, ...], str, str]] = [
    # Seguros / Previdência / Resseguro → InsuranceScorer
    (
        ("SEGUR", "PREVID", "RESSEGUR", "CAPITALIZACAO", "SEGUROS", "RESSEGUROS"),
        "insurance",
        "Seguros",
    ),
    # Bancos / Crédito / Intermediação → BankingScorer
    (
        (
            "BANCO",
            "BANCOS",
            "CREDITO",
            "INTERMEDIACAO",
            "INTERMEDIACAO FINANCEIRA",
            "FINANCIAMENTO",
            "ARRENDAMENTO",
            "LEASING",
            "CAMBIO",
            "FINANCEI",          # captura "Serviços Financeiros", "Intermediação Financeira"
            "SERV FINANC",
        ),
        "banking",
        "Bancos",
    ),
    # Mercado de capitais / holdings financeiras / participações → BankingScorer (Financeiro)
    (
        (
            "BOLSA",
            "CORRETORA",
            "DISTRIBUIDOR",
            "ADMINISTRACAO DE FUNDOS",
            "HOLDING FINANCEIR",
            "HOLDING",           # captura "Holdings Diversificadas", "Holding Pura"
            "FUNDO DE INVESTIMENT",
            "PARTICIPAC",        # captura "Participações", "Sociedade de Participações"
        ),
        "banking",
        "Financeiro",
    ),
    # BDRs — Recibos de Depósito Brasileiros (emissores estrangeiros na B3)
    (
        ("RECIBO DE DEPOSITO", "BDR", "DEPOSITARY RECEIPT"),
        "beneish",
        "BDR",
    ),
    # Setores não-financeiros → BeneishSectorScorer
    (("PETROLEO", "GAS", "COMBUSTIVEL", "EXPLORACAO DE PETROLE"), "beneish", "Energia"),
    (("ELETRIC", "ENERGIA ELETR", "TRANSMISSAO DE ENERGIA"),      "beneish", "Energia"),
    (("SANEAMENTO", "AGUA E ESGOTOS"),                             "beneish", "Saneamento"),
    (("MINERACAO", "MINERIO"),                                     "beneish", "Mineração"),
    (("SIDERURGI", "METALURGI", "PRODUCAO DE ACO"),                "beneish", "Siderurgia"),
    (("PAPEL", "CELULOSE", "FLORESTA"),                            "beneish", "Papel & Celulose"),
    (("TELECOMUNIC",),                                             "beneish", "Telecom"),
    (("ALIMENT", "BEBIDA", "FRIGORIF", "CARNES"),                  "beneish", "Alimentos"),
    (("VAREJO", "COMERCIO VAREJ"),                                 "beneish", "Varejo"),
    (("SAUDE", "HOSPITAL", "FARMAC", "DIAGNOSTICO"),               "beneish", "Saúde"),
    (("TECNOLOGIA", "SOFTWAR", "INFORM"),                          "beneish", "Tecnologia"),
    (("CONSTRUCAO", "INCORPORA", "IMOBI"),                         "beneish", "Construção"),
    (("TRANSPORTE", "LOGISTIC", "FERROVIARI", "AVIACAO"),          "beneish", "Transporte"),
    (("EDUCACAO", "ENSINO"),                                       "beneish", "Educação"),
    (("AGRO", "AGRICULT", "CANA"),                                 "beneish", "Agronegócio"),
    (("QUIMICA", "PETROQUIMICA", "FERTILIZANT"),                   "beneish", "Química"),
    (("TEXTIL", "VESTUARIO", "CALCADO"),                           "beneish", "Consumo"),
    (("INDUSTRIA", "MAQUINA", "EQUIPAMENTO", "ELETRODOMEST"),      "beneish", "Indústria"),
    (("LOCACAO", "ALUGUEL"),                                       "beneish", "Locação"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Fold to upper ASCII, strip accents and extra whitespace."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return " ".join(nfkd.encode("ascii", "ignore").decode("ascii").upper().split())


def classify_setor_ativ(setor_ativ: str) -> Tuple[str, str]:
    """
    Map a CVM SETOR_ATIV string to ``(scorer_type, sector_label)``.

    Case-insensitive: input is normalised to uppercase ASCII via ``_normalise()``
    before comparison. Keywords in ``_SETOR_RULES`` are already uppercase, so
    any CVM capitalisation variant matches correctly.

    Parameters
    ----------
    setor_ativ : str
        Raw value from the SETOR_ATIV column of cad_cia_aberta.csv.

    Returns
    -------
    scorer_type : ``"beneish"`` | ``"banking"`` | ``"insurance"``
    sector_label : human-readable label aligned with ticker_map.py SECTOR_LABELS
    """
    norm = _normalise(setor_ativ)   # → uppercase ASCII, no accents
    for keywords, scorer_type, label in _SETOR_RULES:
        if any(kw in norm for kw in keywords):
            return scorer_type, label
    return "beneish", "Outros"


# ---------------------------------------------------------------------------
# P1.5 — Dynamic ticker → company-name translation via yfinance
# ---------------------------------------------------------------------------

# Regex for B3 ticker format: 4 uppercase letters + 1-2 digits (e.g. PETR4, BPAC11)
_TICKER_RE = re.compile(r"^[A-Z]{4}[0-9]{1,2}$")


def _yfinance_longname(ticker: str, timeout: float = 3.0) -> Optional[str]:
    """
    Query Yahoo Finance for the company's long name given a B3 ticker.

    Appends ".SA" suffix (São Paulo Exchange convention used by Yahoo Finance).
    The network call is executed in a thread and cancelled after *timeout*
    seconds to avoid blocking the Dash callback worker.

    Returns ``None`` on timeout, ImportError (yfinance not installed), or any
    network/parsing error.
    """
    def _fetch() -> Optional[str]:
        try:
            import yfinance as yf  # lazy import — optional dependency
            info = yf.Ticker(f"{ticker}.SA").info
            return info.get("longName") or info.get("shortName")
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fetch)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning("_yfinance_longname: timeout after %.1fs for %s", timeout, ticker)
            return None


# ---------------------------------------------------------------------------
# Tarefa 1 — CVMRegistry (download + cache + lookup)
# ---------------------------------------------------------------------------

class CVMRegistry:
    """
    In-process, disk-cached registry of CVM Dados Cadastrais.

    Parameters
    ----------
    cache_dir : Path | None
        Where to store ``cad_cia_aberta.csv``.  Defaults to data/cache/.
    max_age_days : int
        Re-download the CSV when it is older than this many days.
    """

    # Module-level singleton so the DataFrame is loaded at most once per process.
    _instance: Optional["CVMRegistry"] = None

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        max_age_days: int = 7,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        cache_dir: Optional[Path | str] = None,
        max_age_days: int = 7,
    ) -> "CVMRegistry":
        """Return (or create) the module-level singleton."""
        if cls._instance is None:
            cls._instance = cls(cache_dir=cache_dir, max_age_days=max_age_days)
        return cls._instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Path:
        return self.cache_dir / _CACHE_FILENAME

    def _is_stale(self) -> bool:
        p = self._cache_path()
        if not p.exists():
            return True
        age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
        return age > timedelta(days=self.max_age_days)

    def _download(self) -> None:
        dest = self._cache_path()
        logger.info("Downloading CVM cadastral registry → %s", dest)
        for attempt in range(1, 5):
            try:
                req = urllib.request.Request(
                    CVM_CAD_URL,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; AdvisorBrainFSA/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    dest.write_bytes(resp.read())
                logger.info(
                    "CVM registry downloaded (%d bytes)", dest.stat().st_size
                )
                return
            except urllib.error.URLError as exc:
                logger.warning("Download attempt %d/4 failed: %s", attempt, exc)
                if attempt < 4:
                    time.sleep(2 ** attempt)  # 2s, 4s, 8s backoff
        raise RuntimeError(
            "Failed to download CVM cadastral registry after 4 attempts. "
            f"URL: {CVM_CAD_URL}"
        )

    def _build_df(self) -> pd.DataFrame:
        if self._is_stale():
            self._download()

        df = pd.read_csv(
            self._cache_path(),
            sep=_CSV_SEP,
            encoding=_CSV_ENCODING,
            dtype=str,
            on_bad_lines="skip",
        )
        # Normalise column names (strip whitespace, upper)
        df.columns = [c.strip().upper() for c in df.columns]

        # Keep only active companies
        if "SIT" in df.columns:
            df = df[df["SIT"].str.upper().str.strip() == "ATIVO"].copy()

        # Derive scorer / sector classification from SETOR_ATIV
        setor_col = next(
            (c for c in ["SETOR_ATIV", "SETOR"] if c in df.columns), None
        )
        if setor_col:
            classified = df[setor_col].apply(
                lambda s: pd.Series(
                    classify_setor_ativ(str(s)),
                    index=["_SCORER_TYPE", "_SECTOR_LABEL"],
                )
            )
            df[["_SCORER_TYPE", "_SECTOR_LABEL"]] = classified
        else:
            logger.warning("SETOR_ATIV column not found; defaulting all to beneish/Outros")
            df["_SCORER_TYPE"] = "beneish"
            df["_SECTOR_LABEL"] = "Outros"

        # Pre-normalised name for fast fuzzy lookup
        if "DENOM_SOCIAL" in df.columns:
            df["_DENOM_NORM"] = df["DENOM_SOCIAL"].apply(_normalise)

        # Normalised CNPJ digits column (14 digits, no separators)
        cnpj_col = next(
            (c for c in ["CNPJ_CIA", "CNPJ"] if c in df.columns), None
        )
        if cnpj_col:
            df["_CNPJ_DIGITS"] = df[cnpj_col].apply(
                lambda x: "".join(c for c in str(x) if c.isdigit())
            )

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public property — lazy load
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._build_df()
        return self._df

    def refresh(self) -> None:
        """Force re-download and reload."""
        p = self._cache_path()
        if p.exists():
            p.unlink()
        self._df = None
        _ = self.df  # trigger reload

    # ------------------------------------------------------------------
    # Tarefa 3 — Lookup API
    # ------------------------------------------------------------------

    def lookup_by_cnpj(self, cnpj: str) -> Optional[Dict]:
        """
        Resolve CNPJ → registry metadata.

        Accepts formatted (``33.000.167/0001-01``) or raw digits.
        Matches on full 14 digits or 8-digit CNPJ_BASICO.

        Returns
        -------
        dict with keys ``denom_social``, ``sector``, ``scorer_type``,
        ``setor_ativ``, ``cnpj_digits`` — or ``None`` if not found.
        """
        if "_CNPJ_DIGITS" not in self.df.columns:
            return None
        raw = "".join(c for c in str(cnpj) if c.isdigit())
        if not raw:
            return None

        if len(raw) >= 14:
            mask = self.df["_CNPJ_DIGITS"] == raw[:14]
        else:
            # Partial CNPJ_BASICO (8 digits)
            mask = self.df["_CNPJ_DIGITS"].str[:8] == raw[:8]

        hits = self.df[mask]
        if hits.empty:
            return None
        row = hits.iloc[0]
        return {
            "denom_social": row.get("DENOM_SOCIAL", ""),
            "sector":       row.get("_SECTOR_LABEL", "Outros"),
            "scorer_type":  row.get("_SCORER_TYPE", "beneish"),
            "setor_ativ":   row.get("SETOR_ATIV", ""),
            "cnpj_digits":  row.get("_CNPJ_DIGITS", raw),
        }

    def search_by_name(self, name_fragment: str, top_n: int = 5) -> List[Dict]:
        """
        Fuzzy search: all words in ``name_fragment`` must appear in DENOM_SOCIAL.

        Returns up to ``top_n`` matches as dicts with keys
        ``denom_social``, ``sector``, ``scorer_type``, ``setor_ativ``, ``cnpj``.
        """
        if "_DENOM_NORM" not in self.df.columns:
            return []
        words = _normalise(name_fragment).split()
        if not words:
            return []
        mask = self.df["_DENOM_NORM"].apply(
            lambda x: all(w in x for w in words)
        )
        hits = self.df[mask].head(top_n)
        cnpj_col = next(
            (c for c in ["CNPJ_CIA", "CNPJ"] if c in self.df.columns), None
        )
        results = []
        for _, row in hits.iterrows():
            results.append(
                {
                    "denom_social": row.get("DENOM_SOCIAL", ""),
                    "sector":       row.get("_SECTOR_LABEL", "Outros"),
                    "scorer_type":  row.get("_SCORER_TYPE", "beneish"),
                    "setor_ativ":   row.get("SETOR_ATIV", ""),
                    "cnpj":         row.get(cnpj_col, "") if cnpj_col else "",
                }
            )
        return results

    def resolve_ticker_sector(self, ticker: str) -> Tuple[str, str]:
        """
        Best-effort resolution of an *unknown* ticker to ``(sector_label, scorer_type)``.

        Strategy
        --------
        1. Use TICKER_TO_KEYWORD to derive a search fragment (e.g. ITUB4 → "ITAUUNIBANCO").
        2. Fuzzy-search in DENOM_SOCIAL (all words must match).
        3. Return the classification of the top hit.
        4. Fall back to ``("Outros", "beneish")`` when nothing is found.

        BDR tickers (e.g. MSFT34, AAPL34) are short-circuited via the static
        TICKER_SECTOR map — they have no CVM cadastral entry (foreign issuers).
        Other tickers in TICKER_SECTOR also bypass the registry lookup so
        the static map always takes precedence.
        """
        upper = ticker.upper()
        # Fast path: use static TICKER_SECTOR when available (includes BDRs)
        if upper in TICKER_SECTOR:
            sector_label = TICKER_SECTOR[upper]
            logger.debug(
                "resolve_ticker_sector(%s) → %s/beneish (static map)",
                upper, sector_label,
            )
            return sector_label, "beneish"

        # Slow path: fuzzy CVM registry lookup for unknown tickers
        keyword = TICKER_TO_KEYWORD.get(upper, upper)
        hits = self.search_by_name(keyword, top_n=1)
        if hits:
            h = hits[0]
            logger.debug(
                "resolve_ticker_sector(%s) → %s via '%s'",
                upper, h["sector"], h["denom_social"],
            )
            return h["sector"], h["scorer_type"]
        logger.debug("resolve_ticker_sector(%s) → no match, defaulting", upper)
        return "Outros", "beneish"

    # ------------------------------------------------------------------
    # Chain of Responsibility — unified free-form query resolver
    # ------------------------------------------------------------------

    _CNPJ_RE = re.compile(r"^\d{2}[\.\-]?\d{3}[\.\-]?\d{3}[/\-]?\d{4}[\-]?\d{2}$")

    def resolve_query(self, query: str) -> Optional[Dict]:
        """
        Resolve a free-form query (ticker, CNPJ, or name) to registry metadata
        using a three-priority Chain of Responsibility:

        Priority 1 — Static ticker map (fast path, zero I/O)
            Covers all 135 known B3 tickers including BDRs (MSFT34, AAPL34…).
            BDRs have no CVM cadastral entry; without this step they would
            fall through to the fuzzy search and return no results.

        Priority 2 — CNPJ lookup
            Strips all non-digit characters with regex before querying the
            registry, so both "33.000.167/0001-01" and "33000167000101" work.
            Requires at least 8 digits (CNPJ_BASICO partial match).

        Priority 3 — Fuzzy name matching
            All words in the query must appear in DENOM_SOCIAL (normalised
            ASCII, case-insensitive). Returns the top-1 hit.

        Returns
        -------
        dict with keys ``denom_social``, ``sector``, ``scorer_type``,
        ``setor_ativ``, ``cnpj_digits``, ``source`` — or ``None`` if
        no match is found at any priority level.
        """
        q = query.strip()
        upper = q.upper()

        # ── P1: Static ticker map ────────────────────────────────────────────
        if upper in TICKER_SECTOR:
            sector_label = TICKER_SECTOR[upper]
            logger.debug("resolve_query(%s) → P1/static sector=%s", q, sector_label)
            return {
                "denom_social": TICKER_TO_KEYWORD.get(upper, upper),
                "sector":       sector_label,
                "scorer_type":  "beneish",
                "setor_ativ":   "",
                "cnpj_digits":  "",
                "source":       "static_map",
            }

        # ── P1.5: yfinance dynamic ticker translation ────────────────────────
        # Handles B3 tickers NOT in the 135-ticker static map (small/mid caps,
        # newly listed companies, etc.).  Resolves ticker → company long name
        # and forwards to fuzzy matching (P3) so the CVM registry can be used.
        # Example: "GFSA3" → "GAFISA S.A." → matches DENOM_SOCIAL in the CSV.
        if _TICKER_RE.match(upper):
            long_name = _yfinance_longname(upper)
            if long_name:
                logger.info(
                    "resolve_query(%s) → P1.5/yfinance long_name='%s'", q, long_name
                )
                hits = self.search_by_name(long_name, top_n=1)
                if hits:
                    h = hits[0]
                    h["source"] = "yfinance_fuzzy"
                    return h
            else:
                logger.debug(
                    "resolve_query(%s) → P1.5/yfinance returned nothing", q
                )

        # ── P2: CNPJ lookup (digits-only, via regex strip) ───────────────────
        digits = re.sub(r"\D", "", q)           # remove all non-digit chars
        if len(digits) >= 8:
            hit = self.lookup_by_cnpj(digits)
            if hit:
                hit["source"] = "cnpj_lookup"
                logger.debug(
                    "resolve_query(%s) → P2/cnpj denom=%s", q, hit.get("denom_social")
                )
                return hit

        # ── P3: Fuzzy name matching ──────────────────────────────────────────
        hits = self.search_by_name(q, top_n=1)
        if hits:
            h = hits[0]
            h["source"] = "fuzzy_name"
            logger.debug(
                "resolve_query(%s) → P3/fuzzy denom=%s", q, h.get("denom_social")
            )
            return h

        logger.debug("resolve_query(%s) → no match at any priority", q)
        return None

    def get_scorer_instance(self, scorer_type: str) -> SectorScorer:  # noqa: ARG002
        """Return BeneishSectorScorer (single scorer for all non-financial companies)."""
        return BeneishSectorScorer()

    # ------------------------------------------------------------------
    # Tarefa 2 — Dynamic non-financial company universe
    # ------------------------------------------------------------------

    def get_non_financial_df(self) -> pd.DataFrame:
        """
        Return a filtered DataFrame of active B3-listed non-financial companies.

        Filters applied (in order):
        1. SIT_REG == 'ATIVO'  (or SIT == 'ATIVO' in older CSV versions)
           → only companies with active CVM registration.
        2. TP_MERC contains 'BOLSA'
           → only companies listed on B3 (discards FI, debentures, etc.).
        3. SETOR_ATIV does NOT contain any keyword from FINANCIAL_BLACKLIST_KEYWORDS
           → removes banks, insurers, credit companies, and other institutions
             for which the Beneish M-Score is not applicable.

        Returns
        -------
        pd.DataFrame
            Columns include at minimum: DENOM_SOCIAL, CNPJ_CIA (or CNPJ),
            SETOR_ATIV, _SECTOR_LABEL, _SCORER_TYPE, _CNPJ_DIGITS, _DENOM_NORM.
        """
        df = self.df.copy()

        # ── Filter 1: active registration ────────────────────────────────
        # CVM changed the column name between dataset versions.
        sit_col = next((c for c in ["SIT_REG", "SIT"] if c in df.columns), None)
        if sit_col:
            df = df[df[sit_col].str.upper().str.strip() == "ATIVO"].copy()
        else:
            logger.warning("get_non_financial_df: SIT_REG/SIT column not found, skipping status filter")

        # ── Filter 2: B3-listed (TP_MERC contains 'BOLSA') ───────────────
        if "TP_MERC" in df.columns:
            df = df[
                df["TP_MERC"].str.upper().str.strip().str.contains("BOLSA", na=False)
            ].copy()
        else:
            logger.warning("get_non_financial_df: TP_MERC column not found, skipping market filter")

        # ── Filter 3: blacklist financial sectors ─────────────────────────
        setor_col = next((c for c in ["SETOR_ATIV", "SETOR"] if c in df.columns), None)
        if setor_col:
            def _is_financial(setor: str) -> bool:
                norm = _normalise(str(setor))
                return any(kw in norm for kw in FINANCIAL_BLACKLIST_KEYWORDS)

            df = df[~df[setor_col].apply(_is_financial)].copy()
        else:
            logger.warning("get_non_financial_df: SETOR_ATIV column not found, skipping financial blacklist")

        logger.info(
            "get_non_financial_df: %d non-financial B3-listed active companies", len(df)
        )
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary helpers (useful for audit / UI)
    # ------------------------------------------------------------------

    def scorer_distribution(self) -> Dict[str, int]:
        """Return counts of companies per scorer_type across the registry."""
        if "_SCORER_TYPE" not in self.df.columns:
            return {}
        return self.df["_SCORER_TYPE"].value_counts().to_dict()

    def sector_distribution(self) -> Dict[str, int]:
        """Return counts of companies per sector_label."""
        if "_SECTOR_LABEL" not in self.df.columns:
            return {}
        return self.df["_SECTOR_LABEL"].value_counts().to_dict()


# ---------------------------------------------------------------------------
# Tarefa 1 — Public convenience function
# ---------------------------------------------------------------------------

def fetch_cvm_company_registry(
    cache_dir: Optional[Path | str] = None,
    max_age_days: int = 7,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download (or load from disk cache) the CVM Dados Cadastrais de
    Companhias Abertas and return as a DataFrame enriched with
    ``_SCORER_TYPE``, ``_SECTOR_LABEL``, ``_DENOM_NORM``, and
    ``_CNPJ_DIGITS`` columns derived from SETOR_ATIV.

    Parameters
    ----------
    cache_dir : Path | str | None
        Local directory for the cached CSV.  Defaults to ``data/cache/``.
    max_age_days : int
        Re-download when the cached file is older than this many days.
    force_refresh : bool
        Ignore cache age and force a fresh download.

    Returns
    -------
    pd.DataFrame
    """
    registry = CVMRegistry(cache_dir=cache_dir, max_age_days=max_age_days)
    if force_refresh:
        registry.refresh()
    return registry.df
