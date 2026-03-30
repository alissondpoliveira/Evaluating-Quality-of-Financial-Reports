"""
Ticker → CVM Company Name mapping
----------------------------------
Maps B3 tickers to CVM search keywords and sector classifications.
Covers the main liquid names across all B3 industry groups.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Static ticker → search keyword
# ---------------------------------------------------------------------------

TICKER_TO_KEYWORD: dict[str, str] = {
    # Energia
    "PETR3": "PETROBRAS",       "PETR4": "PETROBRAS",
    "CSAN3": "COSAN",           "UGPA3": "ULTRAPAR",
    "VBBR3": "VIBRA",           "RRRP3": "3R PETROLEUM",
    "PRIO3": "PRIO",            "RECV3": "PETRORECONCAVO",
    "BRAV3": "BRAVA ENERGIA",

    # Bancos
    "ITUB3": "ITAUUNIBANCO",    "ITUB4": "ITAUUNIBANCO",
    "BBDC3": "BRADESCO",        "BBDC4": "BRADESCO",
    "BBAS3": "BANCO DO BRASIL", "SANB11": "SANTANDER",
    "ITSA3": "ITAUSA",          "ITSA4": "ITAUSA",
    "BRSR3": "BANRISUL",        "BRSR6": "BANRISUL",
    "BPAC11": "BTG PACTUAL",    "ABCB4": "ABC BRASIL",
    "BMGB4": "BMG",

    # Seguros
    "BBSE3": "BB SEGURIDADE PARTICIPACOES",
    "PSSA3": "PORTO SEGURO",
    "SULA11": "SUL AMERICA",
    "IRBR3": "IRB BRASIL RESSEGUROS",
    "CXSE3": "CAIXA SEGURIDADE",

    # Financeiro (não-banco)
    "B3SA3": "B3",
    "CIEL3": "CIELO",
    "CASH3": "MELIUZ",

    # Mineração & Siderurgia
    "VALE3": "VALE",
    "CSNA3": "COMPANHIA SIDERURGICA NACIONAL",
    "GGBR3": "GERDAU",          "GGBR4": "GERDAU",
    "GOAU3": "METALURGICA GERDAU", "GOAU4": "METALURGICA GERDAU",
    "USIM3": "USIMINAS",        "USIM5": "USIMINAS",
    "CMIN3": "CSN MINERACAO",

    # Utilidades
    "ELET3": "CENTRAIS ELETRICAS BRASILEIRAS",
    "ELET6": "CENTRAIS ELETRICAS BRASILEIRAS",
    "CPFE3": "CPFL ENERGIA",
    "CMIG3": "CEMIG",           "CMIG4": "CEMIG",
    "EGIE3": "ENGIE BRASIL",
    "TAEE11": "TRANSMISSAO PAULISTA",
    "TRPL3": "TRANSMISSAO PAULISTA", "TRPL4": "TRANSMISSAO PAULISTA",
    "SBSP3": "SABESP",
    "CSMG3": "COPASA",
    "SAPR3": "SANEPAR",         "SAPR11": "SANEPAR",
    "ENEV3": "ENEVA",           "AURE3": "AUREN",
    "EQTL3": "EQUATORIAL",      "ISAE4": "ISA ENERGIA",

    # Telecom
    "VIVT3": "TELEFONICA",
    "TIMS3": "TIM",
    "OIBR3": "OI S.A",

    # Consumo, Varejo & Alimentos
    "ABEV3": "AMBEV",           "LREN3": "LOJAS RENNER",
    "MGLU3": "MAGAZINE LUIZA",  "PCAR3": "GRUPO PAO DE ACUCAR",
    "ASAI3": "SENDAS DISTRIBUIDORA", "CRFB3": "CARREFOUR",
    "JBSS3": "JBS",             "MRFG3": "MARFRIG",
    "BEEF3": "MINERVA",         "BRFS3": "BRF",
    "MDIA3": "M DIAS BRANCO",   "HYPE3": "HYPERA",
    "SOMA3": "GRUPO SOMA",      "ARZZ3": "AREZZO",
    "ALPA4": "ALPARGATAS",      "NTCO3": "NATURA",
    "SMTO3": "SAO MARTINHO",    "CAML3": "CAMIL ALIMENTOS",

    # Imobiliário & Real Estate
    "MRVE3": "MRV",             "CYRE3": "CYRELA",
    "EZTC3": "EZTEC",           "EVEN3": "EVEN",
    "DIRR3": "DIRECIONAL",      "MULT3": "MULTIPLAN",
    "IGUV3": "IGUATEMI",        "ALOS3": "ALLOS",

    # Logística, Transporte & Aviação
    "RAIL3": "RUMO",            "CCRO3": "CCR",
    "ECOR3": "ECORODOVIAS",     "GOLL4": "GOL LINHAS AEREAS",
    "AZUL4": "AZUL",            "POMO4": "MARCOPOLO",

    # Locação de Veículos & Equipamentos
    "RENT3": "LOCALIZA",        "MOVI3": "MOVIDA",
    "VAMO3": "VAMOS LOCACAO",

    # Saúde
    "RDOR3": "REDE D OR",       "HAPV3": "HAPVIDA",
    "GNDI3": "NOTRE DAME INTERMEDICA",
    "FLRY3": "FLEURY",          "QUAL3": "QUALICORP",
    "DASA3": "DIAGNOSTICOS DA AMERICA",

    # Tecnologia & Dados
    "TOTVS3": "TOTVS",          "LWSA3": "LOCAWEB",
    "INTB3": "INTELBRAS",       "POSI3": "POSITIVO",

    # Indústria & Engenharia
    "WEGE3": "WEG",             "EMBR3": "EMBRAER",
    "RAPT4": "RANDON",          "FRAS3": "FRAS-LE",

    # Química & Petroquímica
    "BRKM5": "BRASKEM",         "UNIP6": "UNIPAR",

    # Papel & Celulose
    "SUZB3": "SUZANO",
    "KLBN3": "KLABIN",          "KLBN11": "KLABIN",

    # Agronegócio
    "SLCE3": "SLC AGRICOLA",    "AGRO3": "BRASILAGRO",
    "JALL3": "JALLES MACHADO",

    # Educação
    "COGN3": "COGNA",           "YDUQ3": "YDUQS",
    "SEER3": "SER EDUCACIONAL", "ANIM3": "ANIMA",
    "AFYA3": "AFYA",

    # BDRs (Brazilian Depositary Receipts — nível III, negociados na B3)
    "MSFT34": "MICROSOFT",      "AAPL34": "APPLE",
    "GOOGL34": "ALPHABET",      "AMZO34": "AMAZON",
    "TSLA34": "TESLA",          "META34": "META PLATFORMS",
    "NVDC34": "NVIDIA",         "JPMC34": "JPMORGAN CHASE",
    "BERK34": "BERKSHIRE HATHAWAY", "VISA34": "VISA",
    "NFLX34": "NETFLIX",        "DISB34": "WALT DISNEY",
    "GOGL34": "ALPHABET",       "AMGN34": "AMGEN",
}

# ---------------------------------------------------------------------------
# Sector labels
# ---------------------------------------------------------------------------

SECTOR_LABELS: dict[str, str] = {
    "Energia":          "Petróleo, Gás e Combustíveis",
    "Bancos":           "Instituições Financeiras — Bancos",
    "Seguros":          "Seguros e Resseguros",
    "Financeiro":       "Serviços Financeiros (não-banco)",
    "Mineração":        "Mineração e Siderurgia",
    "Utilidades":       "Energia Elétrica, Saneamento e Gás",
    "Telecom":          "Telecomunicações",
    "Consumo":          "Consumo, Varejo e Alimentos",
    "Imobiliário":      "Construção Civil e Real Estate",
    "Logística":        "Logística, Transporte e Aviação",
    "Locação":          "Locação de Veículos e Equipamentos",
    "Saúde":            "Saúde e Serviços Médicos",
    "Tecnologia":       "Tecnologia da Informação",
    "Indústria":        "Indústria e Engenharia",
    "Química":          "Química e Petroquímica",
    "Papel & Celulose": "Papel, Celulose e Embalagens",
    "Agronegócio":      "Agronegócio e Insumos",
    "Educação":         "Educação",
    "BDR":              "BDRs — Recibos de Depósito Brasileiros",
    "Outros":           "Demais Setores",
}

# Financial sectors are excluded from Beneish ranking (CVM blacklist).
# FINANCIAL_GROUP is kept for reference but is intentionally empty —
# all companies now use BeneishSectorScorer.
FINANCIAL_GROUP: frozenset[str] = frozenset()

TICKER_SECTOR: dict[str, str] = {
    # Energia
    "PETR3": "Energia",   "PETR4": "Energia",
    "CSAN3": "Energia",   "UGPA3": "Energia",
    "VBBR3": "Energia",   "RRRP3": "Energia",
    "PRIO3": "Energia",   "RECV3": "Energia",
    "BRAV3": "Energia",

    # Bancos
    "ITUB3": "Bancos",    "ITUB4": "Bancos",
    "BBDC3": "Bancos",    "BBDC4": "Bancos",
    "BBAS3": "Bancos",    "SANB11": "Bancos",
    "ITSA3": "Bancos",    "ITSA4": "Bancos",
    "BRSR3": "Bancos",    "BRSR6": "Bancos",
    "BPAC11": "Bancos",   "ABCB4": "Bancos",
    "BMGB4": "Bancos",

    # Seguros
    "BBSE3": "Seguros",   "PSSA3": "Seguros",
    "SULA11": "Seguros",  "IRBR3": "Seguros",
    "CXSE3": "Seguros",

    # Financeiro
    "B3SA3": "Financeiro", "CIEL3": "Financeiro",
    "CASH3": "Financeiro",

    # Mineração
    "VALE3": "Mineração",  "CSNA3": "Mineração",
    "GGBR3": "Mineração",  "GGBR4": "Mineração",
    "GOAU3": "Mineração",  "GOAU4": "Mineração",
    "USIM3": "Mineração",  "USIM5": "Mineração",
    "CMIN3": "Mineração",

    # Utilidades
    "ELET3": "Utilidades",  "ELET6": "Utilidades",
    "CPFE3": "Utilidades",  "CMIG3": "Utilidades",
    "CMIG4": "Utilidades",  "EGIE3": "Utilidades",
    "TAEE11": "Utilidades", "TRPL3": "Utilidades",
    "TRPL4": "Utilidades",  "SBSP3": "Utilidades",
    "CSMG3": "Utilidades",  "SAPR3": "Utilidades",
    "SAPR11": "Utilidades", "ENEV3": "Utilidades",
    "AURE3": "Utilidades",  "EQTL3": "Utilidades",
    "ISAE4": "Utilidades",

    # Telecom
    "VIVT3": "Telecom",  "TIMS3": "Telecom",
    "OIBR3": "Telecom",

    # Consumo
    "ABEV3": "Consumo",  "LREN3": "Consumo",
    "MGLU3": "Consumo",  "PCAR3": "Consumo",
    "ASAI3": "Consumo",  "CRFB3": "Consumo",
    "SMTO3": "Consumo",  "JBSS3": "Consumo",
    "MRFG3": "Consumo",  "BEEF3": "Consumo",
    "BRFS3": "Consumo",  "MDIA3": "Consumo",
    "HYPE3": "Consumo",  "SOMA3": "Consumo",
    "ARZZ3": "Consumo",  "ALPA4": "Consumo",
    "NTCO3": "Consumo",  "CAML3": "Consumo",

    # Imobiliário
    "MRVE3": "Imobiliário",  "CYRE3": "Imobiliário",
    "EZTC3": "Imobiliário",  "EVEN3": "Imobiliário",
    "DIRR3": "Imobiliário",  "MULT3": "Imobiliário",
    "IGUV3": "Imobiliário",  "ALOS3": "Imobiliário",

    # Logística
    "RAIL3": "Logística",  "CCRO3": "Logística",
    "ECOR3": "Logística",  "GOLL4": "Logística",
    "AZUL4": "Logística",  "EMBR3": "Logística",
    "POMO4": "Logística",

    # Locação
    "RENT3": "Locação",  "MOVI3": "Locação",
    "VAMO3": "Locação",

    # Saúde
    "RDOR3": "Saúde",   "HAPV3": "Saúde",
    "GNDI3": "Saúde",   "FLRY3": "Saúde",
    "QUAL3": "Saúde",   "DASA3": "Saúde",

    # Tecnologia
    "TOTVS3": "Tecnologia",  "LWSA3": "Tecnologia",
    "INTB3": "Tecnologia",   "POSI3": "Tecnologia",

    # Indústria
    "WEGE3": "Indústria",  "RAPT4": "Indústria",
    "FRAS3": "Indústria",

    # Química
    "BRKM5": "Química",  "UNIP6": "Química",

    # Papel & Celulose
    "SUZB3": "Papel & Celulose",
    "KLBN3": "Papel & Celulose",  "KLBN11": "Papel & Celulose",

    # Agronegócio
    "SLCE3": "Agronegócio",  "AGRO3": "Agronegócio",
    "JALL3": "Agronegócio",

    # Educação
    "COGN3": "Educação",  "YDUQ3": "Educação",
    "SEER3": "Educação",  "ANIM3": "Educação",
    "AFYA3": "Educação",

    # BDRs
    "MSFT34": "BDR",    "AAPL34": "BDR",
    "GOOGL34": "BDR",   "AMZO34": "BDR",
    "TSLA34": "BDR",    "META34": "BDR",
    "NVDC34": "BDR",    "JPMC34": "BDR",
    "BERK34": "BDR",    "VISA34": "BDR",
    "NFLX34": "BDR",    "DISB34": "BDR",
    "GOGL34": "BDR",    "AMGN34": "BDR",
}


def get_sector(ticker: str) -> str:
    return TICKER_SECTOR.get(ticker.upper(), "Outros")


def get_sector_dynamic(ticker: str, use_registry: bool = True) -> str:
    """
    Like ``get_sector`` but falls back to the CVM cadastral registry for
    tickers not present in the static map.

    Parameters
    ----------
    ticker : str
        B3 ticker (e.g. ``"PETR4"``).
    use_registry : bool
        When *True* and the ticker is unknown, query CVMRegistry (may trigger
        a network download on first call).  Set to *False* to skip the lookup
        and return ``"Outros"`` immediately.
    """
    sector = TICKER_SECTOR.get(ticker.upper())
    if sector:
        return sector
    if not use_registry:
        return "Outros"
    # Lazy import to avoid circular dependency at module load time
    from .cvm_registry import CVMRegistry  # noqa: PLC0415
    registry = CVMRegistry.get_instance()
    resolved_sector, _ = registry.resolve_ticker_sector(ticker)
    return resolved_sector


def is_financial_sector(sector: str) -> bool:
    """True for sectors that use specialised financial risk scorers (not Beneish)."""
    return sector in FINANCIAL_GROUP


_CNPJ_RE = re.compile(r"^\d{2}[\.\-]?\d{3}[\.\-]?\d{3}[/\-]?\d{4}[\-]?\d{2}$")


def _normalise(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("ascii").upper().strip()


def resolve_company(query: str) -> tuple[str, str]:
    """
    Resolve query → (search_keyword, query_type).
    query_type: "cnpj" | "ticker" | "name"
    """
    q = query.strip()
    if _CNPJ_RE.match(q.replace(" ", "")):
        return q, "cnpj"
    upper = q.upper()
    if upper in TICKER_TO_KEYWORD:
        return TICKER_TO_KEYWORD[upper], "ticker"
    return _normalise(q), "name"
