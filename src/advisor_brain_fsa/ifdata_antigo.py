"""
IF.data no plano contábil antigo (2000-2024, antes da Resolução CMN 4.966)
==========================================================================
Usado só pela validação histórica (validacao_bancos.py). Mesmo caminho do site do IF.data que
ifdata_bancos.py, na série rest/relatorios (arquivos em "ifdata/AAAAMM/..."), e os mesmos nomes de
campo, para que bank_quality.avaliar rode sem adaptação.

Os "lid" das contas abaixo são os mesmos de 2014 a 2024 (conferido); o número do arquivo de dados
de cada conta muda com o ano e vem do infoAAAAMM.json. Cadastro dos Conglomerados Prudenciais e
Instituições Independentes: relatório 1004 até jun/2023, 1009 depois (os códigos são os mesmos).

Diferenças de conteúdo em relação ao plano novo:
- ativos problemáticos: carteira classificada nos níveis E a H da Resolução 2.682 (atraso acima
  de 90 dias), aproximação do estágio 3 da 4.966;
- resultado de títulos (TVM) numa linha só, sem separar juros de valor justo;
- despesa de provisão: "Resultado de Provisão para Créditos de Difícil Liquidação".
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www3.bcb.gov.br/ifdata/rest/arquivos"
_TIMEOUT = 300

CONTAS = {
    78182: "ativo_total", 78186: "patrimonio_liquido", 78287: "depositos",
    78188: "disponibilidades", 78189: "aplicacoes_interfinanceiras", 78190: "tvm",
    78191: "credito_bruto", 78192: "credito_perda_esperada",
    78194: "arrendamento_bruto", 78197: "arrendamento_perda_esperada",
    78203: "juros_credito", 78204: "rendas_arrendamento", 78205: "juros_tvm",
    78206: "resultado_derivativos", 78207: "cambio_aplicacoes", 78212: "cambio_captacoes",
    78213: "perda_esperada_credito", 78215: "resultado_intermediacao",
    78216: "outros_servicos", 78217: "tarifas", 78221: "resultado_participacoes",
    78222: "outras_receitas", 78223: "outras_despesas", 78226: "resultado_antes_ir",
    78228: "ir_csll", 78187: "lucro_liquido",
    24454: "carteira_total", 23354: "_E", 23355: "_F", 23356: "_G", 23357: "_H",
    79664: "basileia", 79659: "capital_principal", 79661: "razao_alavancagem", 79665: "rwa",
}


def _baixar(nome: str, cache_dir: Path):
    destino = cache_dir / ("ifdata_antigo_" + nome.replace("/", "_"))
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    for tentativa in range(4):
        try:
            r = requests.get(_BASE, params={"nomeArquivo": "ifdata/" + nome}, timeout=_TIMEOUT)
            r.raise_for_status()
            dados = r.json()
            break
        except (requests.RequestException, ValueError) as exc:
            if tentativa == 3:
                raise
            logger.warning("IF.data antigo %s: %s (nova tentativa)", nome, exc)
            time.sleep(10 * (tentativa + 1))
    cache_dir.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    return dados


def carregar(data_base: int, cache_dir: Path) -> dict[str, dict]:
    """Conglomerados prudenciais e independentes de uma data-base do plano antigo: código → campos."""
    am = str(data_base)
    info = {x["lid"]: x["a"] for x in _baixar(f"{am}/info{am}.json", cache_dir)}
    cad = _baixar(f"{am}/cadastro{am}_{1009 if data_base >= 202309 else 1004}.json", cache_dir)
    vals: dict[int, dict] = {}
    for arq in sorted({info[lid] for lid in CONTAS if lid in info}):
        for ent in _baixar(f"{am}/dados{am}_{arq}.json", cache_dir)["values"]:
            campos = vals.setdefault(int(ent["e"]), {})
            for x in ent["v"]:
                if x["i"] in CONTAS and x["v"] is not None and info.get(x["i"]) == arq:
                    campos[CONTAS[x["i"]]] = x["v"]
    saida = {}
    for c in cad:
        reg = {"codigo": c["c0"], "nome": c["c2"].strip(), "tcb": c.get("c3", ""), "segmento": c.get("c12", "")}
        reg.update(vals.get(int(c["c0"]), {}))
        reg["ativos_problematicos"] = sum(reg.pop(k, 0) or 0 for k in ("_E", "_F", "_G", "_H"))
        reg["inadimplencia"] = reg["ativos_problematicos"]
        saida[c["c0"]] = reg
    return saida


def individuais(data_base: int, cache_dir: Path) -> list[dict]:
    """Cadastro das instituições individuais (c33: raiz do CNPJ; c22: conglomerado prudencial)."""
    am = str(data_base)
    return _baixar(f"{am}/cadastro{am}_1006.json", cache_dir)
