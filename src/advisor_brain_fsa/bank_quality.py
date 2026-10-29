"""
Qualidade dos lucros de instituições financeiras — triagem no estilo do M-Score
================================================================================
O M-Score de Beneish não se aplica a bancos: não há estoque, margem bruta nem imobilizado
relevante, e o "accrual" que mais mexe no lucro de um banco é a provisão para perdas de
crédito. A leitura do currículo do CFA (Level II, Financial Statement Analysis — Analysis of
Financial Institutions) organiza a análise pelo CAMELS (Capital, Asset quality, Management,
Earnings, Liquidity, Sensitivity) e aponta como riscos à qualidade dos lucros de um banco:
estimativas de perda de crédito (provisão), ganhos de valor justo e outros resultados não
recorrentes, e crescimento acelerado da carteira.

Este módulo traduz isso em índices de um período contra o mesmo período do ano anterior,
como os do M-Score (índice > 1 aponta na direção do sinal):

    IPR  Índice de provisão        (provisão/carteira)_t-1 ÷ (provisão/carteira)_t
                                    > 1,25 com ativos problemáticos sem melhora: provisão
                                    caiu sem que o risco caísse — lucro "ajudado" pela estimativa
    ICO  Índice de cobertura       cobertura_t-1 ÷ cobertura_t
                                    > 1,25: estoque de perda esperada cobre menos os problemáticos
    IAP  Índice de ativos problem. (problemáticos/carteira)_t ÷ (problemáticos/carteira)_t-1
                                    > 1,30 com problemáticos de pelo menos 3% da carteira (a mediana
                                    do sistema subiu 17% entre jun/2025 e jun/2026: régua acima do ciclo)
    ICC  Índice de crescimento     carteira_t ÷ carteira_t-1
                                    > 1,30: crescimento acelerado dilui a inadimplência e
                                    antecipa receita de juros antes das perdas
    VRN  Variação do não recorrente (não recorrente/receita)_t − (não recorrente/receita)_t-1
                                    > 0,15: o lucro passou a depender mais de câmbio, valor
                                    justo, derivativos, participações e outras receitas
    ICA  Índice de capital         Basileia_t-1 ÷ Basileia_t
                                    > 1,15 com Basileia abaixo de 15%: capital caindo e já perto
                                    do mínimo, mais pressão sobre o resultado
    BAS  Basileia abaixo de 10,5%  mínimo regulatório de 8% mais o adicional de conservação

Não há coeficientes estimados (Beneish estimou os dele com uma amostra de manipuladoras; não
há amostra equivalente de bancos brasileiros). A pontuação é a contagem de sinais acesos e o
nível de alerta segue a contagem: 0 Normal, 1 Atenção, 2 Alto Risco, 3 ou mais Crítico.
As réguas são escolhas do projeto, não valores da literatura.
"""

from __future__ import annotations

REGUAS = {"ipr": 1.25, "ico": 1.25, "iap": 1.30, "icc": 1.30, "vrn": 0.15, "ica": 1.15, "bas": 0.105}
_IAP_MINIMO = 0.03
_ICA_BASILEIA = 0.15
ALERTAS = ["Normal", "Atenção", "Alto Risco", "Crítico"]


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


def _soma(d, *campos):
    return sum(d.get(c) or 0.0 for c in campos)


def carteira(d) -> float:
    """Carteira bruta: crédito, arrendamento, outras operações de crédito e cartão pós-pago."""
    return _soma(d, "credito_bruto", "arrendamento_bruto", "outros_creditos_bruto", "pagamentos_pos_bruto")


def estoque_perda(d) -> float:
    return -_soma(d, "credito_perda_esperada", "arrendamento_perda_esperada",
                  "outros_creditos_perda_esperada", "pagamentos_pos_perda_esperada")


def provisao(d) -> float:
    """Despesa de perda esperada de crédito no período (positiva = despesa)."""
    return -_soma(d, "perda_esperada_credito", "perda_esperada_arrendamento",
                  "perda_esperada_outros_creditos", "perda_esperada_pagamentos")


def receita(d) -> float:
    """Receita operacional bruta: juros, rendas de crédito e serviços."""
    return _soma(d, "juros_aplicacoes", "juros_tvm", "juros_credito", "rendas_arrendamento",
                 "rendas_outros_creditos", "servicos_pagamento", "tarifas", "outros_servicos")


def nao_recorrente(d) -> float:
    """Câmbio, valor justo e outros resultados de TVM, derivativos, participações e outras receitas/despesas."""
    return _soma(d, "cambio_aplicacoes", "cambio_tvm", "cambio_credito", "cambio_captacoes", "valor_justo_tvm",
                 "outros_tvm", "resultado_derivativos", "resultado_participacoes", "outras_receitas", "outras_despesas")


def problematicos(d):
    return _div(d.get("ativos_problematicos"), d.get("carteira_total"))


def cobertura(d):
    return _div(estoque_perda(d), d.get("ativos_problematicos"))


def indicadores(d, meses: int) -> dict:
    """Retrato de um período (CAMELS resumido). Fluxos anualizados pelo número de meses da DRE."""
    anual = 12 / meses
    g, pl = carteira(d), d.get("patrimonio_liquido")
    return {
        "basileia": d.get("basileia"),
        "capital_principal": d.get("capital_principal"),
        "alavancagem": d.get("razao_alavancagem"),
        "problematicos": problematicos(d),
        "inadimplencia": _div(d.get("inadimplencia"), d.get("carteira_total")),
        "cobertura": cobertura(d),
        "custo_risco": _div(provisao(d) * anual, g),
        "roe": _div((d.get("lucro_liquido") or 0.0) * anual, pl) if pl and pl > 0 else None,
        "roa": _div((d.get("lucro_liquido") or 0.0) * anual, d.get("ativo_total")),
        "credito_depositos": _div(g, d.get("depositos")),
        "liquidez": _div(_soma(d, "disponibilidades", "aplicacoes_interfinanceiras", "tvm"), d.get("ativo_total")),
        "nao_recorrente": _div(nao_recorrente(d), receita(d)) if receita(d) > 0 else None,
    }


def avaliar(t: dict, t1: dict, meses: int) -> dict:
    """Índices t contra t-1, sinais e alerta. Sem carteira de crédito nos dois períodos, só o retrato."""
    saida = {"ind": indicadores(t, meses), "ind_t1": indicadores(t1, meses), "indices": {}, "sinais": [], "avaliavel": False}
    g, g1 = carteira(t), carteira(t1)
    if g <= 0 or g1 <= 0 or not t.get("carteira_total") or not t1.get("carteira_total"):
        saida["motivo"] = "sem carteira de crédito nos dois períodos"
        return saida
    if not (t.get("patrimonio_liquido") or 0) > 0:
        saida["motivo"] = "patrimônio líquido não positivo"
        return saida
    saida["avaliavel"] = True
    ix = saida["indices"]
    taxa, taxa1 = _div(provisao(t), g), _div(provisao(t1), g1)
    iap = _div(problematicos(t), problematicos(t1))
    ix["ipr"] = _div(taxa1, taxa) if taxa and taxa > 0 and taxa1 and taxa1 > 0 else None
    ix["ico"] = _div(cobertura(t1), cobertura(t))
    ix["iap"] = iap
    ix["icc"] = g / g1
    nr, nr1 = saida["ind"]["nao_recorrente"], saida["ind_t1"]["nao_recorrente"]
    ix["vrn"] = None if nr is None or nr1 is None else nr - nr1
    ix["ica"] = _div(t1.get("basileia"), t.get("basileia"))

    sem_melhora = iap is None or iap >= 1
    # provisão que vira reversão (taxa ≤ 0) enquanto o ano anterior provisionava conta como queda máxima
    if ((ix["ipr"] is not None and ix["ipr"] > REGUAS["ipr"]) or (taxa is not None and taxa <= 0 < (taxa1 or 0))) and sem_melhora:
        saida["sinais"].append("ipr")
    if ix["ico"] is not None and ix["ico"] > REGUAS["ico"]:
        saida["sinais"].append("ico")
    if iap is not None and iap > REGUAS["iap"] and (problematicos(t) or 0) >= _IAP_MINIMO:
        saida["sinais"].append("iap")
    if ix["icc"] > REGUAS["icc"]:
        saida["sinais"].append("icc")
    if ix["vrn"] is not None and ix["vrn"] > REGUAS["vrn"]:
        saida["sinais"].append("vrn")
    if ix["ica"] is not None and ix["ica"] > REGUAS["ica"] and (t.get("basileia") or 1) < _ICA_BASILEIA:
        saida["sinais"].append("ica")
    if t.get("basileia") is not None and 0 < t["basileia"] < REGUAS["bas"]:
        saida["sinais"].append("bas")
    saida["alerta"] = ALERTAS[min(len(saida["sinais"]), 3)]
    return saida
