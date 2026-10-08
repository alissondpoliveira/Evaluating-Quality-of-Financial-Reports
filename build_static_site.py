#!/usr/bin/env python3
"""
build_static_site.py — dados da versão estática (beneish.alissonprata.io)
==========================================================================
Mesmo lote do build_market_cache.py (cadastro CVM → não financeiras → DFP →
BeneishSectorScorer), com os mesmos módulos de cálculo e sem alterar nenhuma
fórmula. Além do resumo, guarda o que a ficha de cada empresa mostra: os oito
índices, a contribuição de cada um para o M-Score (coeficiente × índice) e os
números de base dos dois exercícios.

Saída: site/public/dados.json (lido por site/gerar.py)

Uso:
    python build_static_site.py                 # último exercício com DFP entregue (ano passado a partir de maio)
    python build_static_site.py --year 2025 --workers 2
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

from advisor_brain_fsa.beneish_mscore import _COEFFICIENTS, _MANIPULATION_THRESHOLD
from advisor_brain_fsa.cvm_registry import CVMRegistry
from advisor_brain_fsa.data_fetcher import CVMDataFetcher
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

    def emit(self, record):
        cnpj = self.atual.get(record.thread)
        if not cnpj:
            return
        msg = record.getMessage()
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


def _empresa(cnpj, denom, setor, ano, fetcher, scorer) -> dict:
    """Uma empresa: mesmas chamadas de _score_company, com os detalhes da ficha."""
    base = {"ticker": _ticker_for_denom(denom), "nome": denom, "cnpj": cnpj, "setor": setor, "erro": ""}
    _avisos.atual[threading.get_ident()] = cnpj
    try:
        fd_t, fd_t1 = fetcher.get_financial_data(cnpj, year_t=ano, year_t1=ano - 1)
        # Sem ativo total num dos anos, o coletor devolveu tudo zerado: os índices viram 1 e o
        # M-Score sai −2,48 com qualidade "Alta", sem significado. Trata como empresa sem dados.
        if not fd_t.total_assets or not fd_t1.total_assets:
            raise ValueError(f"Demonstrações sem valores em {ano if not fd_t.total_assets else ano - 1} (ativo total zero)")
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
    except Exception as exc:  # o lote original também registra o erro e segue
        base["erro"] = str(exc)[:200]
    finally:
        _avisos.atual.pop(threading.get_ident(), None)
    av = dict(_avisos.avisos.get(cnpj) or {})
    # Sinal de exibição, sem efeito no cálculo: índice fora de uma faixa larga costuma ser
    # conta mapeada errado ou ano atípico (ex.: recebíveis quase zerados num dos anos).
    extremos = [k for k, v in (base.get("indices") or {}).items()
                if v is not None and ((k == "tata" and abs(v) > 1) or (k != "tata" and (v > 10 or v < 0.1)))]
    if extremos:
        av["indices_extremos"] = extremos
    if any(av.values()):
        base["avisos"] = av
    return base


def build(ano: int, workers: int) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    registry = CVMRegistry.get_instance(cache_dir=str(_CACHE_DIR))
    nf = registry.get_non_financial_df()
    cnpj_col = next(c for c in ["_CNPJ_DIGITS", "CNPJ_CIA", "CNPJ"] if c in nf.columns)
    fetcher, scorer = CVMDataFetcher(cache_dir=str(_CACHE_DIR)), BeneishSectorScorer()
    total = len(nf)
    logger.info("Empresas não financeiras: %d (exercício %d)", total, ano)

    def trabalho(item):
        i, row = item
        cnpj = str(row.get(cnpj_col, "")).strip()
        denom = str(row.get("DENOM_SOCIAL", "")).strip()
        setor = str(row.get("_SECTOR_LABEL", "Outros")).strip()
        r = _empresa(cnpj or denom, denom, setor, ano, fetcher, scorer)
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
        "exercicio": ano,
        "gerado_em": agora.strftime("%d/%m/%Y %H:%M"),
        "limiar": _MANIPULATION_THRESHOLD,
        "coeficientes": _COEFFICIENTS,
        "referencias_alerta": {s.attr: s.threshold for s in _INDEX_SPECS},
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
    # a DFP do ano X sai até o fim de março de X+1: até abril, o último exercício completo é o retrasado
    hoje = date.today()
    p.add_argument("--year", type=int, default=hoje.year - (1 if hoje.month >= 5 else 2))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    build(a.year, a.workers)
