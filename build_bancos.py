#!/usr/bin/env python3
"""
build_bancos.py — dados da aba de instituições financeiras (beneish.alissonprata.io/bancos/)
=============================================================================================
Universo: todos os Conglomerados Prudenciais e Instituições Independentes do IF.data (Banco
Central) na última data-base publicada, comparados com a mesma data-base do ano anterior
(semestre contra semestre em junho e dezembro, trimestre contra trimestre em março e setembro).
Cálculo em src/advisor_brain_fsa/bank_quality.py.

Saída: site/public/bancos.json (lido pela página gerada em site/gerar.py)

Uso:
    python build_bancos.py                 # última data-base com o mesmo mês do ano anterior
    python build_bancos.py --data-base 202606
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from advisor_brain_fsa.bank_quality import REGUAS, avaliar
from advisor_brain_fsa.ifdata_bancos import carregar, datas_disponiveis

_CACHE_DIR = _ROOT / "data" / "cache"
_SAIDA = _ROOT / "site" / "public" / "bancos.json"
logger = logging.getLogger("build_bancos")


def _r(v, casas=4):
    return None if v is None else round(float(v), casas)


def _arred(d: dict, casas=4) -> dict:
    return {k: _r(v, casas) for k, v in d.items()}


def _mediana(vs):
    vs = sorted(v for v in vs if v is not None)
    return None if not vs else (vs[len(vs) // 2] if len(vs) % 2 else (vs[len(vs) // 2 - 1] + vs[len(vs) // 2]) / 2)


# contas que a ficha mostra nos dois períodos
_BASE = ["ativo_total", "patrimonio_liquido", "depositos", "credito_bruto", "pagamentos_pos_bruto", "carteira_total",
         "ativos_problematicos", "inadimplencia", "perda_esperada_credito", "resultado_intermediacao",
         "resultado_antes_ir", "lucro_liquido", "basileia"]


def build(data_base: int) -> Path:
    anterior = data_base - 100
    meses = 6 if data_base % 100 in (6, 12) else 3
    t, t1 = carregar(data_base, _CACHE_DIR), carregar(anterior, _CACHE_DIR)
    logger.info("IF.data %d: %d instituições; %d: %d", data_base, len(t), anterior, len(t1))
    linhas = []
    for cod, d in t.items():
        d1 = t1.get(cod)
        base = {k: d[k] for k in ("codigo", "nome", "tipo", "tcb", "consolidacao", "controle", "uf", "segmento")}
        base["ativo"] = _r(d.get("ativo_total"), 0)
        if not d1:
            base.update(avaliavel=False, motivo=f"sem dados em {anterior}")
            linhas.append(base)
            continue
        r = avaliar(d, d1, meses)
        base.update(avaliavel=r["avaliavel"], ind=_arred(r["ind"]), ind_t1=_arred(r["ind_t1"]))
        if r["avaliavel"]:
            base.update(indices=_arred(r["indices"]), sinais=r["sinais"], eixos=r["eixos"], alerta=r["alerta"])
        else:
            base["motivo"] = r["motivo"]
        base["fin"] = {"t": {k: _r(d.get(k), 4 if k == "basileia" else 0) for k in _BASE},
                       "t1": {k: _r(d1.get(k), 4 if k == "basileia" else 0) for k in _BASE}}
        linhas.append(base)
    linhas.sort(key=lambda x: -(x.get("ativo") or 0))

    # mediana de cada índice entre as avaliadas do mesmo tipo, para a ficha comparar com os pares
    pares: dict[str, dict] = {}
    for tipo in {x["tipo"] for x in linhas if x.get("avaliavel")}:
        grupo = [x for x in linhas if x.get("avaliavel") and x["tipo"] == tipo]
        med = {k: _mediana([x["indices"].get(k) for x in grupo]) for k in ("ipr", "ico", "iap", "icc", "vrn", "ica")}
        med["basileia"] = _mediana([x["ind"].get("basileia") for x in grupo])
        pares[tipo] = {"n": len(grupo), **_arred(med)}

    agora = datetime.now(timezone(timedelta(hours=-3)))
    saida = {"data_base": data_base, "anterior": anterior, "meses": meses,
             "gerado_em": agora.strftime("%d/%m/%Y %H:%M"), "reguas": REGUAS, "pares": pares, "instituicoes": linhas}
    _SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SAIDA.with_suffix(".tmp")
    tmp.write_text(json.dumps(saida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(_SAIDA)
    ok = sum(1 for x in linhas if x["avaliavel"])
    logger.info("Pronto: %d instituições, %d avaliadas → %s", len(linhas), ok, _SAIDA)
    return _SAIDA


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dados da aba de instituições financeiras.")
    p.add_argument("--data-base", type=int)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    datas = datas_disponiveis(_CACHE_DIR)
    alvo = a.data_base or max(d for d in datas if d - 100 in datas)
    build(alvo)
