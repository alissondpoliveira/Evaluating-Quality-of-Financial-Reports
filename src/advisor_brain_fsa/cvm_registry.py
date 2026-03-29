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

import logging
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .sector_scorer import (
    BankingScorer,
    BeneishSectorScorer,
    InsuranceScorer,
    SectorScorer,
)
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

    Parameters
    ----------
    setor_ativ : str
        Raw value from the SETOR_ATIV column of cad_cia_aberta.csv.

    Returns
    -------
    scorer_type : ``"beneish"`` | ``"banking"`` | ``"insurance"``
    sector_label : human-readable label aligned with ticker_map.py SECTOR_LABELS
    """
    norm = _normalise(setor_ativ)
    for keywords, scorer_type, label in _SETOR_RULES:
        if any(kw in norm for kw in keywords):
            return scorer_type, label
    return "beneish", "Outros"


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
            # Map sector label → scorer_type
            if sector_label in ("Bancos", "Financeiro"):
                scorer_type = "banking"
            elif sector_label == "Seguros":
                scorer_type = "insurance"
            else:
                scorer_type = "beneish"
            logger.debug(
                "resolve_ticker_sector(%s) → %s/%s (static map)",
                upper, sector_label, scorer_type,
            )
            return sector_label, scorer_type

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

    def get_scorer_instance(self, scorer_type: str) -> SectorScorer:
        """Instantiate the right SectorScorer from a scorer_type string."""
        if scorer_type == "insurance":
            return InsuranceScorer()
        if scorer_type == "banking":
            return BankingScorer()
        return BeneishSectorScorer()

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
