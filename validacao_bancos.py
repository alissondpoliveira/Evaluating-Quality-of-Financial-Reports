#!/usr/bin/env python3
"""
validacao_bancos.py — validação histórica dos sinais de bancos (bank_quality.py)
=================================================================================
Pergunta: as instituições com sinal aceso pioraram mais depois do que as outras?

Amostra: Conglomerados Prudenciais e Instituições Independentes do IF.data, junho e dezembro de
2015 a 2024, cada data-base contra a mesma do ano anterior (plano antigo, ifdata_antigo.py). As
datas-base de 2025-26 (plano novo) entram só como futuro.

Desfechos:
    estresse   nos 24 meses seguintes, Basileia abaixo de 10,5% ou patrimônio líquido não positivo
               (instituições já nessa situação na data da observação ficam de fora)
    lucro      ROA do mesmo semestre um ano depois menos o ROA atual; e se o lucro vira prejuízo
    regimes    liquidações em andamento na API de regimes especiais do BC (só as em curso: o BC não
               publica a lista histórica), contra os sinais calculados com os dados de 2024

Pesado: baixa cerca de 500 MB do IF.data na primeira vez (cache em data/cache). Não roda no
workflow semanal; o resultado fica em site/public/bancos_validacao.json e só muda quando as
regras mudam.

Uso:
    python validacao_bancos.py
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from advisor_brain_fsa import ifdata_antigo, ifdata_bancos
from advisor_brain_fsa.bank_quality import ALERTAS, avaliar, eixos

_CACHE = _ROOT / "data" / "cache"
_SAIDA = _ROOT / "site" / "public" / "bancos_validacao.json"
_REGIMES = "https://olinda.bcb.gov.br/olinda/servico/regimes_especiais/versao/v1/odata/Regimes?$format=json&$top=10000"
logger = logging.getLogger("validacao_bancos")

ANTIGAS = [a * 100 + m for a in range(2014, 2025) for m in (6, 12)]
SINAIS = ("ipr", "ico", "vrn", "icc", "iap", "ica", "bpm")


def _estresse(r: dict) -> bool:
    b, pl = r.get("basileia"), r.get("patrimonio_liquido")
    return (b is not None and b < 0.105) or (pl is not None and pl <= 0)


def _roa(r: dict | None):
    a = (r or {}).get("ativo_total")
    return (r.get("lucro_liquido") or 0.0) / a if a else None


def _grupo(obs: list[dict], base: float) -> dict:
    n = len(obs)
    ev = sum(o["estresse"] for o in obs)
    droa = [o["droa"] for o in obs if o["droa"] is not None]
    vira = [o["vira_prejuizo"] for o in obs if o["vira_prejuizo"] is not None]
    taxa = ev / n if n else None
    return {"n": n, "eventos": ev, "taxa": taxa, "lift": taxa / base if n and base else None,
            "droa_mediano": statistics.median(droa) if droa else None,
            "vira_prejuizo": sum(vira) / len(vira) if vira else None}


def _regimes(base: dict) -> list[dict]:
    try:
        regs = requests.get(_REGIMES, timeout=60).json()["value"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        logger.warning("regimes especiais indisponíveis: %s", exc)
        return []
    saida, vistos = [], set()
    ind = ifdata_antigo.individuais(202412, _CACHE)
    for r in sorted(regs, key=lambda x: x["Inicio"][-4:] + x["Inicio"][3:5] + x["Inicio"][:2]):
        if r["Inicio"][-4:] < "2025":
            continue
        i = next((c for c in ind if (c.get("c33") or c["c0"]).zfill(8) == r["CnpjRaiz"]), None)
        if i is None:
            continue
        t = base[202412]
        cod = next((k for k, v in t.items() if v["nome"] == (i.get("c22") or "").strip()), None) if i.get("c4") == "C" else i["c0"]
        if cod is None or cod in vistos or cod not in t:
            continue
        vistos.add(cod)
        caso = {"nome": t[cod]["nome"], "inicio": r["Inicio"], "tipo": r["Tipo"].title()}
        for dt in (202406, 202412):
            d, d1 = base[dt].get(cod), base[dt - 100].get(cod)
            a = avaliar(d, d1, 6) if d and d1 else {"avaliavel": False, "motivo": "sem os dois períodos"}
            caso[str(dt)] = {"alerta": a.get("alerta"), "sinais": a.get("sinais", []), "motivo": a.get("motivo")}
        saida.append(caso)
    return saida


def main() -> None:
    base = {dt: ifdata_antigo.carregar(dt, _CACHE) for dt in ANTIGAS}
    novas = [d for d in ifdata_bancos.datas_disponiveis(_CACHE) if d % 100 in (6, 12)]
    for dt in novas:
        base[dt] = ifdata_bancos.carregar(dt, _CACHE)
    todas = sorted(base)

    obs = []
    for dt in ANTIGAS:
        if dt - 100 not in base:
            continue
        futuro = [d for d in todas if dt < d <= dt + 200]
        if len(futuro) < 4:
            continue  # sem 24 meses de futuro
        t, t1 = base[dt], base[dt - 100]
        for cod, d in t.items():
            if cod not in t1 or _estresse(d):
                continue
            r = avaliar(d, t1[cod], 6)
            if not r["avaliavel"]:
                continue
            f12 = base[dt + 100].get(cod)
            r0, r1 = _roa(d), _roa(f12)
            obs.append({"dt": dt, "cod": cod, "sinais": r["sinais"], "alerta": r["alerta"],
                        "eixos": "+".join(k for k, v in eixos(r["sinais"]).items() if v) or "nenhum",
                        "basileia": d.get("basileia"),
                        "estresse": any(base[f].get(cod) is not None and _estresse(base[f][cod]) for f in futuro),
                        "droa": None if r0 is None or r1 is None else r1 - r0,
                        "vira_prejuizo": None if r0 is None or r1 is None or r0 < 0 else r1 < 0})

    taxa_base = sum(o["estresse"] for o in obs) / len(obs)
    faixas = []
    for lo, hi, rot in ((0.105, 0.13, "10,5% a 13%"), (0.13, 0.15, "13% a 15%"), (0.15, 99, "15% ou mais")):
        g = [o for o in obs if o["basileia"] is not None and lo <= o["basileia"] < hi]
        faixas.append({"faixa": rot, **_grupo(g, taxa_base)})
    agora = datetime.now(timezone(timedelta(hours=-3)))
    saida = {
        "gerado_em": agora.strftime("%d/%m/%Y"),
        "periodo": [obs[0]["dt"], obs[-1]["dt"]],
        "observacoes": len(obs), "instituicoes": len({o["cod"] for o in obs}), "taxa_base": taxa_base,
        "eventos": sum(o["estresse"] for o in obs),
        "niveis": {a: _grupo([o for o in obs if o["alerta"] == a], taxa_base) for a in ALERTAS},
        "sinais": {s: _grupo([o for o in obs if s in o["sinais"]], taxa_base) for s in SINAIS},
        "sem_sinal": _grupo([o for o in obs if not o["sinais"]], taxa_base),
        "basileia": faixas,
        "regimes": _regimes(base),
    }
    ev = [o for o in obs if o["estresse"]]
    saida["captura"] = {a: sum(o["alerta"] == a for o in ev) / len(ev) for a in ALERTAS}
    _SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{saida['observacoes']} observações, {saida['instituicoes']} instituições, {saida['periodo']}, "
          f"taxa base {taxa_base:.1%} ({saida['eventos']} eventos)")
    for nome, grupos in (("nível", saida["niveis"]), ("sinal", saida["sinais"])):
        for k, g in grupos.items():
            print(f"  {nome} {k:18} n={g['n']:6} estresse={g['taxa'] or 0:6.1%} lift={g['lift'] or 0:4.1f}x "
                  f"ΔROA={100*(g['droa_mediano'] or 0):+.2f}pp vira prejuízo={g['vira_prejuizo'] or 0:.1%}")
    for f in faixas:
        print(f"  Basileia {f['faixa']:12} n={f['n']:6} estresse={f['taxa'] or 0:.1%}")
    print("  captura:", {k: f"{v:.0%}" for k, v in saida["captura"].items()})
    for c in saida["regimes"]:
        print(f"  regime {c['inicio']} {c['nome'][:40]:40} jun/24: {c['202406']['alerta'] or c['202406']['motivo']}"
              f" | dez/24: {c['202412']['alerta'] or c['202412']['motivo']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    main()
