"""
CVM Account Code Mapping
------------------------
Maps the standardised account codes used in CVM's DFP files to the
field names expected by FinancialData (BeneishMScore inputs).

CVM DFP file structure
-----------------------
  Statement  | File suffix             | Description
  -----------|-------------------------|---------------------------
  BPA        | BPA_con / BPA_ind       | Balanço Patrimonial Ativo
  BPP        | BPP_con / BPP_ind       | Balanço Patrimonial Passivo
  DRE        | DRE_con / DRE_ind       | Demonstração de Resultado
  DFC_MI     | DFC_MI_con / DFC_MI_ind | Fluxo de Caixa (Indireto)

Account codes follow the CVM taxonomy published in Instrução CVM 480.
Multiple codes per field are tried in priority order; the first code
that exists in the data wins.

References
----------
- Portal CVM Dados Abertos: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/DFP/DADOS/
- Instrução CVM n° 480/2009 — Anexo 9-1 (DFP layout)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class AccountSpec:
    """Specification for one accounting line item."""

    field_name: str           # matches FinancialData attribute
    statement: str            # BPA | BPP | DRE | DFC_MI
    codes: List[str]          # tried in order; first found wins
    description: str
    sign: int = 1             # apply sign correction if CVM stores as negative


# ---------------------------------------------------------------------------
# Master account mapping
# ---------------------------------------------------------------------------
# Codes are ordered from most-specific to most-generic as fallbacks.
# CVM DRE codes can vary slightly between companies (especially SGA breakdown),
# so we provide several candidates.

ACCOUNT_SPECS: List[AccountSpec] = [

    # ── Income Statement (DRE) ───────────────────────────────────────────────

    AccountSpec(
        field_name="revenues",
        statement="DRE",
        codes=["3.01"],
        description="Receita de Venda de Bens e/ou Serviços (Receita Líquida)",
    ),
    AccountSpec(
        field_name="cost_of_goods_sold",
        statement="DRE",
        codes=["3.02"],
        description="Custo dos Bens e/ou Serviços Vendidos (CPV)",
        sign=-1,   # CVM stores COGS as negative; we need the absolute value
    ),
    AccountSpec(
        field_name="sales_general_admin_expenses",
        statement="DRE",
        # 3.04.02 = Gerais e Administrativas; 3.04.01 = Vendas; 3.04 = total despesas
        codes=["3.04.02", "3.04.01", "3.04"],
        description="Despesas Gerais e Administrativas (SGA)",
        sign=-1,
    ),
    AccountSpec(
        field_name="net_income",
        statement="DRE",
        codes=["3.11", "3.09"],     # 3.11 consolidated, 3.09 some older layouts
        description="Lucro/Prejuízo Consolidado do Período",
    ),

    # ── Balance Sheet – Assets (BPA) ─────────────────────────────────────────

    AccountSpec(
        field_name="total_assets",
        statement="BPA",
        codes=["1"],
        description="Ativo Total",
    ),
    AccountSpec(
        field_name="current_assets",
        statement="BPA",
        codes=["1.01"],
        description="Ativo Circulante",
    ),
    AccountSpec(
        field_name="receivables",
        statement="BPA",
        # 1.01.06 = Contas a Receber (newer taxonomy)
        # 1.01.03 = older taxonomy used before 2020 reclassification
        codes=["1.01.06", "1.01.03", "1.01.04"],
        description="Contas a Receber / Clientes",
    ),
    AccountSpec(
        field_name="pp_and_e",
        statement="BPA",
        codes=["1.02.03", "1.02.04"],
        description="Imobilizado (PP&E líquido)",
    ),
    AccountSpec(
        field_name="securities",
        statement="BPA",
        # 1.01.01.02 = Aplicações Financeiras de curto prazo
        # 1.02.01.02 = Aplicações Financeiras de longo prazo
        codes=["1.01.01.02", "1.01.02", "1.01.01"],
        description="Aplicações Financeiras / Títulos e Valores Mobiliários",
    ),

    # ── Balance Sheet – Liabilities (BPP) ───────────────────────────────────

    AccountSpec(
        field_name="current_liabilities",
        statement="BPP",
        codes=["2.01"],
        description="Passivo Circulante",
    ),
    AccountSpec(
        field_name="total_long_term_debt",
        statement="BPP",
        # 2.02.01 = Empréstimos e Financiamentos (longo prazo)
        # 2.02    = Passivo Não Circulante total (fallback)
        codes=["2.02.01", "2.02"],
        description="Empréstimos e Financiamentos de Longo Prazo",
    ),

    # ── Cash Flow Statement (DFC – Método Indireto) ──────────────────────────

    AccountSpec(
        field_name="cash_from_operations",
        statement="DFC_MI",
        codes=["6.01"],
        description="Caixa Gerado nas Operações (CFO)",
    ),
    AccountSpec(
        field_name="depreciation",
        statement="DFC_MI",
        # In the indirect method, D&A is added back under operating adjustments.
        # 6.01.01.02 or 6.01.01.03 depending on the company's chart of accounts.
        codes=["6.01.01.02", "6.01.01.03", "6.01.01.04", "6.01.02"],
        description="Depreciação, Amortização e Exaustão (D&A add-back em DFC)",
        sign=-1,   # stored as negative adjustment; absolute value needed
    ),
]

# Convenience lookup: field_name → AccountSpec
SPEC_BY_FIELD: dict[str, AccountSpec] = {s.field_name: s for s in ACCOUNT_SPECS}

# All required FinancialData fields (must all resolve for a valid result)
REQUIRED_FIELDS = frozenset(
    s.field_name for s in ACCOUNT_SPECS
)

# ---------------------------------------------------------------------------
# Banking-specific supplementary account specs
# ---------------------------------------------------------------------------
# These are used to enrich risk analysis for financial institutions.
# They map to *additional* fields not present in standard FinancialData.
# The SectorScorer for banking uses standard FinancialData as proxies but
# these specs document the preferred CVM codes for future direct extraction.

BANKING_ACCOUNT_SPECS: List[AccountSpec] = [
    AccountSpec(
        field_name="loan_portfolio",
        statement="BPA",
        # 1.01.04 = Operações de Crédito (most banks)
        # 1.01.03 = older taxonomy; 1.01.07 = some layouts
        codes=["1.01.04", "1.01.03", "1.01.07", "1.02.01.01"],
        description="Carteira de Crédito / Operações de Crédito",
    ),
    AccountSpec(
        field_name="loan_loss_provision",
        statement="BPA",
        # PCLD is often a sub-item of loan portfolio (negative)
        codes=["1.01.04.01", "1.01.03.01", "1.01.07.01"],
        description="Provisão para Créditos de Liquidação Duvidosa (PCLD)",
        sign=-1,
    ),
    AccountSpec(
        field_name="shareholders_equity",
        statement="BPP",
        codes=["2.03"],
        description="Patrimônio Líquido (Patrimônio de Referência — Basileia)",
    ),
    AccountSpec(
        field_name="net_interest_income",
        statement="DRE",
        # 3.01 = Resultado de Intermediação Financeira (net)
        # 3.01.01 = Receitas de Intermediação (gross, before funding costs)
        codes=["3.01", "3.01.01"],
        description="Resultado de Intermediação Financeira",
    ),
    AccountSpec(
        field_name="funding_expenses",
        statement="DRE",
        # 3.02 = Despesas de Intermediação Financeira (funding + provisions)
        codes=["3.02", "3.02.01"],
        description="Despesas de Intermediação Financeira",
        sign=-1,
    ),
]

# ---------------------------------------------------------------------------
# Insurance-specific supplementary account specs
# ---------------------------------------------------------------------------
# For insurance companies, the standard DRE structure maps naturally:
#   revenues              → Prêmios Retidos (3.01 or 3.01.01)
#   cost_of_goods_sold    → Sinistros Retidos (3.02 or 3.02.01)
#   sga_expenses          → Despesas de Comercialização + Admin
# These specs document the preferred codes for direct extraction.

INSURANCE_ACCOUNT_SPECS: List[AccountSpec] = [
    AccountSpec(
        field_name="retained_premiums",
        statement="DRE",
        codes=["3.01.01", "3.01"],
        description="Prêmios Retidos Líquidos de Resseguro",
    ),
    AccountSpec(
        field_name="retained_claims",
        statement="DRE",
        # Sinistros are stored negative in CVM
        codes=["3.02.01", "3.03.01", "3.02"],
        description="Sinistros Retidos",
        sign=-1,
    ),
    AccountSpec(
        field_name="insurance_commercial_expenses",
        statement="DRE",
        codes=["3.03.01", "3.04.01", "3.03"],
        description="Despesas de Comercialização — Seguros",
        sign=-1,
    ),
    AccountSpec(
        field_name="reinsurance_result",
        statement="DRE",
        codes=["3.01.02", "3.02.02"],
        description="Resultado com Resseguro",
    ),
]
