"""
IF.data (Banco Central) — dados trimestrais das instituições financeiras
=========================================================================
Fonte: os arquivos JSON que o próprio site do IF.data (www3.bcb.gov.br/ifdata) carrega.
A API OData do IF.data na plataforma Olinda responde erro 500 para qualquer data-base
(testado em 28/10/2026, também de outra rede), então o coletor usa o mesmo caminho do site:

    rest/relatorios2025a2030           → lista de datas-base e arquivos de cada uma
    rest/arquivos?nomeArquivo=<f>      → cada arquivo

Arquivos usados por data-base (AAAAMM):
    cadastroAAAAMM_1009.json  cadastro dos Conglomerados Prudenciais e Instituições Independentes
                              (c0 código, c2 nome, c3 TCB, c4 TD, c7 controle, c10 UF, c12 segmento)
    dadosAAAAMM_1.json        balanço e DRE  ({"values": [{"e": código, "v": [{"i": lid, "v": valor}]}]})
    dadosAAAAMM_3.json        carteira de crédito (inclui inadimplência e ativos problemáticos)
    dadosAAAAMM_5.json        capital (Basileia, capital principal, alavancagem)

O "lid" de cada conta vem do arquivo infoAAAAMM.json e é o mesmo em todas as datas-base da
série 2025-2030 (plano contábil da Resolução CMN 4.966, em vigor desde jan/2025). Valores em reais.
A DRE de junho e de dezembro acumula o semestre; a de março e setembro, o trimestre.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www3.bcb.gov.br/ifdata/rest"
_SERIE = "relatorios2025a2030"
_TIMEOUT = 180

# lid → campo. Arquivo 1: balanço e DRE.
CONTAS_1 = {
    140220: "ativo_total",
    140198: "disponibilidades",
    140199: "aplicacoes_interfinanceiras",
    140200: "tvm",
    140201: "credito_bruto",            # operações de crédito, valor contábil bruto (e1)
    140202: "credito_perda_esperada",   # estoque de perda esperada (e2, negativo)
    140207: "arrendamento_bruto",       # f1
    140208: "arrendamento_perda_esperada",
    140212: "outros_creditos_bruto",    # outras operações com característica de crédito (g1)
    140213: "outros_creditos_perda_esperada",
    145831: "pagamentos_pos_bruto",     # recebíveis de transações de pagamento pós-pagas (h1, cartão)
    145832: "pagamentos_pos_perda_esperada",
    140228: "depositos",
    140239: "captacoes",
    140244: "passivo_exigivel",
    140246: "patrimonio_liquido",
    141821: "juros_aplicacoes",         # a1
    141822: "cambio_aplicacoes",        # a2
    141826: "juros_tvm",                # b1
    141827: "cambio_tvm",               # b2
    141828: "valor_justo_tvm",          # b3
    141829: "outros_tvm",               # b4
    141831: "juros_credito",            # c1
    141832: "cambio_credito",           # c2
    141836: "rendas_arrendamento",      # d
    141837: "rendas_outros_creditos",   # e
    141840: "perda_esperada_credito",   # f3 (despesa de provisão, negativa)
    142851: "perda_esperada_arrendamento",       # f4
    142852: "perda_esperada_outros_creditos",    # f5
    141844: "cambio_captacoes",         # g2
    141847: "despesas_captacao",        # g
    141849: "resultado_derivativos",    # i
    141851: "resultado_intermediacao",  # k
    141852: "servicos_pagamento",       # l1
    141853: "perda_esperada_pagamentos",  # l2
    141856: "tarifas",                  # m
    141857: "outros_servicos",          # n
    141863: "resultado_participacoes",  # s
    141864: "outras_receitas",          # t
    141865: "outras_despesas",          # u
    141867: "resultado_antes_ir",       # w
    141868: "ir_csll",                  # x
    141870: "lucro_liquido",            # z
}
# Arquivo 3: carteira de crédito ativa por carteiras de instrumentos financeiros.
CONTAS_3 = {24454: "carteira_total", 148834: "inadimplencia", 148833: "ativos_problematicos"}
# Arquivo 5: informações de capital.
CONTAS_5 = {79664: "basileia", 79659: "capital_principal", 79661: "razao_alavancagem", 79665: "rwa"}

TCB = {
    "b1": "Banco comercial ou múltiplo",
    "b2": "Banco múltiplo sem carteira comercial, de investimento ou de câmbio",
    "b3S": "Cooperativa de crédito",
    "b3C": "Central de cooperativas",
    "b4": "Banco de desenvolvimento",
    "n1": "Não bancária de crédito",
    "n2": "Não bancária do mercado de capitais",
    "n4": "Instituição de pagamento",
}


def _baixar(nome: str, cache_dir: Path) -> object:
    """Um arquivo do IF.data, com cache em disco (os arquivos de uma data-base não mudam,
    exceto na republicação de dezembro; o cache é por nome e pode ser apagado)."""
    destino = cache_dir / ("ifdata_" + nome.replace("/", "_"))
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    url = f"{_BASE}/arquivos" if nome != _SERIE else f"{_BASE}/{_SERIE}"
    params = {"nomeArquivo": f"ifdata_2025_2030/{nome}"} if nome != _SERIE else None
    for tentativa in range(3):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            dados = r.json()
            break
        except (requests.RequestException, ValueError) as exc:
            if tentativa == 2:
                raise
            logger.warning("IF.data %s: %s (nova tentativa)", nome, exc)
            time.sleep(5 * (tentativa + 1))
    cache_dir.mkdir(parents=True, exist_ok=True)
    if nome != _SERIE:  # a lista de datas-base muda a cada trimestre: não guarda
        destino.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return dados


def datas_disponiveis(cache_dir: Path) -> list[int]:
    """Datas-base (AAAAMM) publicadas na série 2025-2030, em ordem crescente."""
    return sorted(int(x["dt"]) for x in _baixar(_SERIE, cache_dir))


def _valores(nome: str, cache_dir: Path, contas: dict[int, str]) -> dict[int, dict[str, float]]:
    bruto = _baixar(nome, cache_dir)
    saida: dict[int, dict[str, float]] = {}
    for ent in bruto["values"]:
        campos = {contas[x["i"]]: x["v"] for x in ent["v"] if x["i"] in contas and x["v"] is not None}
        if campos:
            saida[int(ent["e"])] = campos
    return saida


def carregar(data_base: int, cache_dir: Path) -> dict[str, dict]:
    """Instituições da data-base (código → cadastro + contas). Conta ausente fica fora do dicionário."""
    am = str(data_base)
    cad = _baixar(f"{am}/cadastro{am}_1009.json", cache_dir)
    blocos = [_valores(f"{am}/dados{am}_{n}.json", cache_dir, contas)
              for n, contas in ((1, CONTAS_1), (3, CONTAS_3), (5, CONTAS_5))]
    saida = {}
    for c in cad:
        cod = int(c["c0"])
        reg = {"codigo": c["c0"], "nome": c["c2"].strip(), "tcb": c.get("c3", ""),
               "tipo": TCB.get(c.get("c3", ""), c.get("c3", "")),
               "consolidacao": "Conglomerado" if c.get("c4") == "C" else "Independente",
               "controle": c.get("c7", ""), "uf": c.get("c10", ""), "segmento": c.get("c12", "")}
        for b in blocos:
            reg.update(b.get(cod, {}))
        saida[c["c0"]] = reg
    return saida
