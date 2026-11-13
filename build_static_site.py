#!/usr/bin/env python3
"""
build_static_site.py — dados da versão estática (beneish.alissonprata.io), últimos 12 meses
==========================================================================
Mesmo lote do build_market_cache.py (cadastro CVM → não financeiras → DFP →
BeneishSectorScorer), com os mesmos módulos de cálculo e sem alterar nenhuma
fórmula. Além do resumo, guarda o que a ficha de cada empresa mostra: os oito
índices, a contribuição de cada um para o M-Score (coeficiente × índice) e os
números de base dos dois exercícios.

Saída: site/public/dados.json (lido por site/gerar.py)

Uso:
    python build_static_site.py                 # 12 meses até o último trimestre entregue pela maioria (ITR + DFP)
    python build_static_site.py --ate 2026-06-30
    python build_static_site.py --year 2025 --workers 2   # comparação anual de duas DFPs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from advisor_brain_fsa import altman_zscore
from advisor_brain_fsa.beneish_mscore import _COEFFICIENTS, _MANIPULATION_THRESHOLD
from advisor_brain_fsa.cvm_registry import CVMRegistry
from advisor_brain_fsa.data_fetcher import CVMDataFetcher
from advisor_brain_fsa.ltm import LTMFetcher, cadeia, um_ano_antes
from advisor_brain_fsa.rank_market import _INDEX_SPECS
from advisor_brain_fsa.sector_scorer import BeneishSectorScorer
from build_market_cache import _CACHE_DIR, _RETRY_DELAY, _ticker_for_denom

_SAIDA = _ROOT / "site" / "public" / "dados.json"
_INDICES = ["dsri", "gmi", "aqi", "sgi", "depi", "sgai", "lvgi", "tata"]
logger = logging.getLogger("build_static_site")


class _AvisosDoColetor(logging.Handler):
    """O coletor só registra no log quando troca de exercício ou zera contas ausentes.
    Este handler liga cada aviso à empresa em cálculo naquela thread, sem mudar o coletor."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.atual: dict[int, str] = {}
        self.avisos: dict[str, dict] = {}
        self.individuais: dict[str, list] = {}  # dado válido, não é aviso: empresa sem controladas

    def emit(self, record):
        cnpj = self.atual.get(record.thread)
        if not cnpj:
            return
        msg = record.getMessage()
        m = re.search(r"Year (\d{4}) .*using individual statements", msg)
        if m:
            self.individuais.setdefault(cnpj, []).append(int(m.group(1)))
            return
        a = self.avisos.setdefault(cnpj, {"exercicio_substituido": [], "contas_ausentes": {}, "indices_neutralizados": []})
        m = re.search(r"Falling back to fiscal year (\d{4})", msg)
        if m:
            a["exercicio_substituido"].append(int(m.group(1)))
        m = re.search(r"Beneish index (\w+) = .* is negative", msg)
        if m:
            a["indices_neutralizados"].append(m.group(1))
        m = re.search(r"Year (\d{4}) .*could not resolve: \[(.*)\]", msg)
        if m:
            a["contas_ausentes"][m.group(1)] = [c.strip(" '\"") for c in m.group(2).split(",") if c.strip()]


_avisos = _AvisosDoColetor()
logging.getLogger("advisor_brain_fsa").addHandler(_avisos)


def _r(v, casas=4):
    return None if v is None else round(float(v), casas)


def _periodo(fim: date, fim_ant: date, ltm: bool) -> dict:
    return {"fim": fim.isoformat(), "fim_ant": fim_ant.isoformat(), "ltm": ltm}


def _dados(cnpj, cadeia, fetcher, ltm):
    """FinancialData dos dois períodos: 12 meses até o trimestre de referência; sem ITR completa,
    o trimestre anterior, e assim até o último exercício anual (DFP)."""
    for q in cadeia:
        if q.month != 12:
            try:
                r = ltm.get(cnpj, q)
            except LookupError:
                continue
            return r.t, r.t1, r.altman_t, _periodo(r.fim, r.fim_ant, True)
        fd_t, fd_t1 = fetcher.get_financial_data(cnpj, year_t=q.year, year_t1=q.year - 1)
        # Sem ativo total num dos anos, o coletor devolveu tudo zerado: os índices viram 1 e o
        # M-Score sai −2,48 com qualidade "Alta", sem significado. Trata como empresa sem dados.
        if not fd_t.total_assets or not fd_t1.total_assets:
            raise ValueError(f"Demonstrações sem valores em {q.year if not fd_t.total_assets else q.year - 1} (ativo total zero)")
        return fd_t, fd_t1, fetcher.get_altman_data(cnpj, q.year), _periodo(q, date(q.year - 1, 12, 31), False)
    raise ValueError("Sem demonstrações para o período")


def _empresa(cnpj, denom, setor, cadeia, fetcher, ltm, scorer) -> dict:
    """Uma empresa: mesmas chamadas de _score_company, com os detalhes da ficha."""
    base = {"ticker": _ticker_for_denom(denom), "nome": denom, "cnpj": cnpj, "setor": setor, "erro": ""}
    _avisos.atual[threading.get_ident()] = cnpj
    try:
        fd_t, fd_t1, extra, base["periodo"] = _dados(cnpj, cadeia, fetcher, ltm)
        if not fd_t.total_assets or not fd_t1.total_assets:
            raise ValueError("Demonstrações sem valores (ativo total zero)")
        sr = scorer.score(fd_t, fd_t1)
        ms, cfq = sr.mscore_result, sr.cfq_result
        base.update({
            "risco": _r(sr.risk_score),
            "mscore": _r(ms.m_score) if ms else None,
            "classificacao": sr.classification,
            "alerta": sr.alert_level.value,
            "accrual": _r(cfq.accrual_ratio) if cfq else None,
            "qualidade": cfq.earnings_quality if cfq else None,
            "flags": [f for f in sr.red_flags if f],
            "indices": {k: _r(getattr(ms, k)) for k in _INDICES} if ms else {},
            "contrib": {k: _r(_COEFFICIENTS[k] * getattr(ms, k)) for k in _INDICES} if ms else {},
            "fin": {"t": {k: _r(v, 0) for k, v in dataclasses.asdict(fd_t).items()},
                    "t1": {k: _r(v, 0) for k, v in dataclasses.asdict(fd_t1).items()}},
        })
        # Altman Z''-Score (mercados emergentes), só do período atual; conta ausente deixa sem Z
        base["fin"]["t"].update({k: _r(v, 0) for k, v in extra.items()})
        z = altman_zscore.calcular(fd_t.total_assets, fd_t.current_assets, fd_t.current_liabilities,
                                   extra["retained_earnings"], extra["ebit"], extra["shareholders_equity"])
        if z:
            base["zscore"] = {"em": _r(z.em_score), "z": _r(z.z), "zona": z.zona,
                              "x": {k: _r(getattr(z, k)) for k in altman_zscore.COEFICIENTES},
                              "contrib": {k: _r(v) for k, v in z.contribuicoes().items()}}
    except Exception as exc:  # o lote original também registra o erro e segue
        base["erro"] = str(exc)[:200]
    finally:
        _avisos.atual.pop(threading.get_ident(), None)
    if _avisos.individuais.get(cnpj):
        base["individual"] = sorted(set(_avisos.individuais[cnpj]))
    av = dict(_avisos.avisos.get(cnpj) or {})
    # o Z-Score relê as demonstrações do ano: o mesmo aviso de troca de exercício pode vir duas vezes
    if av.get("exercicio_substituido"):
        av["exercicio_substituido"] = sorted(set(av["exercicio_substituido"]))
    # Sinal de exibição, sem efeito no cálculo: índice fora de uma faixa larga costuma ser
    # conta mapeada errado ou ano atípico (ex.: recebíveis quase zerados num dos anos).
    extremos = [k for k, v in (base.get("indices") or {}).items()
                if v is not None and ((k == "tata" and abs(v) > 1) or (k != "tata" and (v > 10 or v < 0.1)))]
    if extremos:
        av["indices_extremos"] = extremos
    if any(av.values()):
        base["avisos"] = av
    return base


def build(ref: date, workers: int) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    registry = CVMRegistry.get_instance(cache_dir=str(_CACHE_DIR))
    nf = registry.get_non_financial_df()
    cnpj_col = next(c for c in ["_CNPJ_DIGITS", "CNPJ_CIA", "CNPJ"] if c in nf.columns)
    fetcher, scorer = CVMDataFetcher(cache_dir=str(_CACHE_DIR)), BeneishSectorScorer()
    ltm = LTMFetcher(fetcher)
    if ref is None:
        ref = ltm.periodo_referencia()
    seq = cadeia(ref)
    total = len(nf)
    logger.info("Empresas não financeiras: %d (12 meses até %s; alternativas %s)", total, ref, seq[1:])

    def trabalho(item):
        i, row = item
        cnpj = str(row.get(cnpj_col, "")).strip()
        denom = str(row.get("DENOM_SOCIAL", "")).strip()
        setor = str(row.get("_SECTOR_LABEL", "Outros")).strip()
        r = _empresa(cnpj or denom, denom, setor, seq, fetcher, ltm, scorer)
        if i % 50 == 0:
            logger.info("[%d/%d] %s", i + 1, total, denom[:40])
        time.sleep(_RETRY_DELAY)
        return r

    linhas = list(nf.iterrows())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        resultados = [f.result() for f in as_completed([pool.submit(trabalho, x) for x in linhas])]
    resultados.sort(key=lambda r: (r.get("risco") is None, -(r.get("risco") or 0), r["nome"]))

    agora = datetime.now(timezone(timedelta(hours=-3)))
    saida = {
        "exercicio": ref.year,
        "periodo": _periodo(ref, (date(ref.year - 1, 12, 31) if ref.month == 12 else um_ano_antes(ref)), ref.month != 12),
        "gerado_em": agora.strftime("%d/%m/%Y %H:%M"),
        "limiar": _MANIPULATION_THRESHOLD,
        "coeficientes": _COEFFICIENTS,
        "referencias_alerta": {s.attr: s.threshold for s in _INDEX_SPECS},
        "zscore": {"coeficientes": altman_zscore.COEFICIENTES, "constante": altman_zscore.CONSTANTE_EM,
                   "segura": altman_zscore.LIMIAR_SEGURA, "risco": altman_zscore.LIMIAR_RISCO},
        "empresas": resultados,
    }
    _SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SAIDA.with_suffix(".tmp")
    tmp.write_text(json.dumps(saida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(_SAIDA)
    ok = sum(1 for r in resultados if not r["erro"])
    logger.info("Pronto: %d empresas, %d calculadas, %d sem dados → %s", len(resultados), ok, len(resultados) - ok, _SAIDA)
    return _SAIDA


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dados da versão estática do M-Score.")
    # Sem argumento: últimos 12 meses até o trimestre mais recente que a maioria das companhias já
    # entregou (ITR + DFP). --year força a comparação anual de duas DFPs; --ate fixa o trimestre.
    p.add_argument("--year", type=int, help="exercício anual (DFP) contra o anterior")
    p.add_argument("--ate", help="fim do trimestre dos 12 meses, AAAA-MM-DD")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build(date(a.year, 12, 31) if a.year else (date.fromisoformat(a.ate) if a.ate else None), a.workers)
