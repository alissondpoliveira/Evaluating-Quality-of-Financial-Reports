"""
Altman Z''-Score — versão para mercados emergentes
---------------------------------------------------
Risco de insolvência a partir de quatro índices contábeis, sem preço de mercado e sem receita.

    X1 = (ativo circulante − passivo circulante) ÷ ativo total        capital de giro
    X2 = lucros retidos ÷ ativo total                                   lucro acumulado
    X3 = EBIT ÷ ativo total                                             rentabilidade operacional
    X4 = patrimônio líquido ÷ passivo total                             folga do capital

    Z''      = 6,56·X1 + 3,26·X2 + 6,72·X3 + 1,05·X4
    EM score = Z'' + 3,25

Zonas do EM score (equivalentes às de Z'': 1,10 e 2,60):
  · acima de 5,85      → "Segura"
  · de 4,35 a 5,85     → "Cinzenta"
  · abaixo de 4,35     → "Risco"

Por que esta versão: o Z original (1968) usa o valor de mercado das ações e a receita, foi
estimado com indústrias americanas e só vale para quem tem ação negociada. O Z'' troca o valor de
mercado pelo patrimônio contábil e tira a receita (que varia muito entre setores); o EM score
soma a constante 3,25 para alinhar o resultado à escala de rating de títulos americanos.

No Brasil, lucros retidos = reservas de lucros + lucros ou prejuízos acumulados (ver
cvm_accounts.ALTMAN_ACCOUNT_SPECS); passivo total = ativo total − patrimônio líquido.

O Z-Score mede risco de insolvência, não manipulação de resultado: complementa o M-Score e o
accrual ratio, não os substitui. As réguas não foram estimadas com empresas brasileiras.

Referências
-----------
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate
  bankruptcy. Journal of Finance, 23(4), 589–609.
- Altman, E. I., Hartzell, J. & Peck, M. (1995). Emerging Markets Corporate Bonds: A Scoring
  System. Salomon Brothers.
- Altman, E. I. (2005). An emerging market credit scoring system for corporate bonds. Emerging
  Markets Review, 6(4), 311–323.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

COEFICIENTES = {"x1": 6.56, "x2": 3.26, "x3": 6.72, "x4": 1.05}
CONSTANTE_EM = 3.25
LIMIAR_SEGURA = 5.85
LIMIAR_RISCO = 4.35
ZONAS = ["Segura", "Cinzenta", "Risco"]


@dataclass(frozen=True)
class ZScoreResult:
    x1: float
    x2: float
    x3: float
    x4: float
    z: float          # Z'' (sem a constante)
    em_score: float   # Z'' + 3,25
    zona: str

    def contribuicoes(self) -> dict:
        return {k: COEFICIENTES[k] * getattr(self, k) for k in COEFICIENTES}


def zona(em_score: float) -> str:
    if em_score > LIMIAR_SEGURA:
        return "Segura"
    if em_score >= LIMIAR_RISCO:
        return "Cinzenta"
    return "Risco"


def calcular(total_assets: float, current_assets: float, current_liabilities: float,
             retained_earnings: Optional[float], ebit: Optional[float],
             shareholders_equity: Optional[float]) -> Optional[ZScoreResult]:
    """Z''-Score de mercados emergentes, ou None se faltar conta ou o ativo/passivo não for positivo."""
    if None in (retained_earnings, ebit, shareholders_equity) or not total_assets or total_assets <= 0:
        return None
    passivo_total = total_assets - shareholders_equity
    if passivo_total <= 0:
        return None
    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = shareholders_equity / passivo_total
    z = COEFICIENTES["x1"] * x1 + COEFICIENTES["x2"] * x2 + COEFICIENTES["x3"] * x3 + COEFICIENTES["x4"] * x4
    em = z + CONSTANTE_EM
    return ZScoreResult(x1=x1, x2=x2, x3=x3, x4=x4, z=z, em_score=em, zona=zona(em))
