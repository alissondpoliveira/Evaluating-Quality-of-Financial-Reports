"""
Ticker → CVM Company Name mapping
----------------------------------
The CVM DFP portal does not use stock tickers; it identifies companies by
CNPJ and by the "DENOM_CIA" field (commercial name).  This module provides:

  1. A static lookup table for the most-traded B3 tickers.
  2. A normalised string-match fallback for names not in the table.

Usage
-----
    from advisor_brain_fsa.ticker_map import resolve_company

    name_fragment = resolve_company("PETR4")   # "PETROBRAS"
    name_fragment = resolve_company("11.328.869/0001-99")  # CNPJ passthrough
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Static ticker → search keyword table
# Keyword is matched case-insensitively against DENOM_CIA in the CVM files.
# ---------------------------------------------------------------------------

TICKER_TO_KEYWORD: dict[str, str] = {
    # Energy
    "PETR3": "PETROBRAS",
    "PETR4": "PETROBRAS",
    "CSAN3": "COSAN",
    "UGPA3": "ULTRAPAR",
    "VBBR3": "VIBRA",
    "RRRP3": "3R PETROLEUM",
    "PRIO3": "PRIO",

    # Banks & Finance
    "ITUB3": "ITAUUNIBANCO",
    "ITUB4": "ITAUUNIBANCO",
    "BBDC3": "BRADESCO",
    "BBDC4": "BRADESCO",
    "BBAS3": "BANCO DO BRASIL",
    "SANB11": "SANTANDER",
    "ITSA3": "ITAUSA",
    "ITSA4": "ITAUSA",
    "BRSR3": "BANRISUL",
    "BRSR6": "BANRISUL",
    "BPAC11": "BTG PACTUAL",

    # Mining & Steel
    "VALE3": "VALE",
    "CSNA3": "COMPANHIA SIDERURGICA NACIONAL",
    "GGBR3": "GERDAU",
    "GGBR4": "GERDAU",
    "GOAU3": "METALURGICA GERDAU",
    "GOAU4": "METALURGICA GERDAU",
    "USIM3": "USIMINAS",
    "USIM5": "USIMINAS",

    # Utilities
    "ELET3": "CENTRAIS ELETRICAS BRASILEIRAS",
    "ELET6": "CENTRAIS ELETRICAS BRASILEIRAS",
    "CPFE3": "CPFL ENERGIA",
    "CMIG3": "CEMIG",
    "CMIG4": "CEMIG",
    "EGIE3": "ENGIE BRASIL",
    "TAEE11": "TRANSMISSAO PAULISTA",
    "TRPL3": "TRANSMISSAO PAULISTA",
    "TRPL4": "TRANSMISSAO PAULISTA",
    "SBSP3": "SABESP",
    "CSMG3": "COPASA",
    "SAPR3": "SANEPAR",
    "SAPR11": "SANEPAR",
    "ENEV3": "ENEVA",
    "AURE3": "AUREN",

    # Telecom
    "VIVT3": "TELEFONICA",
    "TIMS3": "TIM",
    "OIBR3": "OI S.A",

    # Consumer / Retail
    "ABEV3": "AMBEV",
    "LREN3": "LOJAS RENNER",
    "MGLU3": "MAGAZINE LUIZA",
    "VIIA3": "VIA",
    "PCAR3": "GRUPO PAO DE ACUCAR",
    "ASAI3": "SENDAS DISTRIBUIDORA",
    "CRFB3": "CARREFOUR",
    "SMTO3": "SAO MARTINHO",
    "JBSS3": "JBS",
    "MRFG3": "MARFRIG",
    "BEEF3": "MINERVA",
    "BRFS3": "BRF",
    "MDIA3": "M DIAS BRANCO",

    # Real Estate / Homebuilders
    "MRVE3": "MRV",
    "CYRE3": "CYRELA",
    "EZTC3": "EZTEC",
    "EVEN3": "EVEN",
    "DIRR3": "DIRECIONAL",

    # Logistics / Transport
    "RAIL3": "RUMO",
    "CCRO3": "CCR",
    "ECOR3": "ECORODOVIAS",
    "GOLL4": "GOL LINHAS AEREAS",
    "AZUL4": "AZUL",

    # Health
    "RDOR3": "REDE D OR",
    "HAPV3": "HAPVIDA",
    "GNDI3": "NOTRE DAME INTERMEDICA",
    "FLRY3": "FLEURY",

    # Tech / Data
    "TOTVS3": "TOTVS",
    "LWSA3": "LOCAWEB",
    "INTB3": "INTELBRAS",
    "POSI3": "POSITIVO",

    # Pulp & Paper
    "SUZB3": "SUZANO",
    "KLBN3": "KLABIN",
    "KLBN11": "KLABIN",

    # Agribusiness
    "SLCE3": "SLC AGRICOLA",
    "ARZZ3": "AREZZO",
    "AGRO3": "BRASILAGRO",

    # Construction / Engineering
    "CVCB3": "CVC",
    "HYPE3": "HYPERA",

    # Aerospace
    "EMBR3": "EMBRAER",
}

_CNPJ_RE = re.compile(r"^\d{2}[\.\-]?\d{3}[\.\-]?\d{3}[/\-]?\d{4}[\-]?\d{2}$")


def _normalise(text: str) -> str:
    """Remove accents and fold to upper ASCII."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_str.upper().strip()


def resolve_company(query: str) -> tuple[str, str]:
    """
    Resolve a query string to a (search_keyword, query_type) tuple.

    Parameters
    ----------
    query : str
        A B3 ticker (e.g. "PETR4"), a CNPJ (e.g. "33.000.167/0001-01"),
        or a free-text company name fragment (e.g. "Petrobras").

    Returns
    -------
    keyword : str
        String to match against DENOM_CIA (for tickers/names) or
        CNPJ_CIA (for CNPJs).
    query_type : str
        One of "cnpj", "ticker", or "name".
    """
    q = query.strip()

    # 1. CNPJ pattern
    if _CNPJ_RE.match(q.replace(" ", "")):
        return q, "cnpj"

    # 2. Known ticker
    upper = q.upper()
    if upper in TICKER_TO_KEYWORD:
        return TICKER_TO_KEYWORD[upper], "ticker"

    # 3. Fallback: treat as a name fragment
    return _normalise(q), "name"
