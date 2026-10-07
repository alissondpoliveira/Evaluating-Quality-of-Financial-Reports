#!/usr/bin/env python3
"""
jev_eventos.py — eventos de qualidade contábil nos documentos da CVM, classificados pelo JEV
=============================================================================================
O M-Score lê só números. A análise forense costuma cruzá-lo com eventos que aparecem nos
documentos da companhia: reapresentação de demonstrações, troca de auditor ou ressalva,
saída do diretor financeiro, investigação, reestruturação de dívida. Este script lê o título
dos fatos relevantes e comunicados ao mercado (IPE/CVM) dos últimos 12 meses das empresas com
M-Score e pergunta ao JEV (TypeSafe, modelo System One) qual desses eventos o documento descreve.

Só exibição: não altera o M-Score nem o nível de alerta. A lista de eventos (EVENTOS) é escolha
analítica e ainda não passou por validação manual nesta taxonomia; a página avisa isso.

Uso:
    python jev_eventos.py --contar     # só conta o que seria enviado (não chama a API)
    python jev_eventos.py              # classifica o que falta no cache (precisa de TYPESAFE_API_KEY)
    python jev_eventos.py --limite 2000

Cache versionado: data/jev/classificacao.json (documento já classificado não é reenviado).
Saída: site/public/eventos.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_DADOS = _ROOT / "site" / "public" / "dados.json"
_SAIDA = _ROOT / "site" / "public" / "eventos.json"
_CACHE = _ROOT / "data" / "jev" / "classificacao.json"
_ZIPS = _ROOT / "data" / "cache"
_IPE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/IPE/DADOS/ipe_cia_aberta_{}.zip"
_API = "https://api.typesafe.ai/v1"
_MODELO = "jev-latest"
CONFIANCA_MINIMA = 0.8  # mesma trava do Grafo de Crédito; nesta taxonomia, ainda sem validação manual
JANELA_DIAS = 365
CATEGORIAS = {"Fato Relevante": "Fato relevante", "Comunicado ao Mercado": "Comunicado ao mercado"}

EVENTOS = {
    "reapresentacao": "Reapresentação, retificação ou republicação de demonstrações financeiras; correção de erro contábil",
    "auditoria": "Troca de auditor independente; parecer com ressalva, abstenção de opinião ou ênfase relevante; divergência com o auditor",
    "saida_executivo": "Renúncia, destituição ou substituição de diretor financeiro, diretor de relações com investidores ou presidente",
    "investigacao": "Investigação, processo, sanção ou acordo com CVM, Polícia Federal, Ministério Público ou CADE; apuração interna de irregularidade",
    "reestruturacao_divida": "Recuperação judicial ou extrajudicial, inadimplemento, vencimento antecipado, waiver ou renegociação de dívida",
    "politica_contabil": "Mudança de política ou estimativa contábil; baixa contábil relevante (impairment) ou ajuste de exercícios anteriores",
    "outro": "Outro assunto, sem relação com a qualidade da informação contábil",
}
PERGUNTA = {"evento": {"type": "choice", "criteria": EVENTOS, "instructions": (
    "Qual destes eventos o documento descreve? Escolha 'outro' se o título não tratar de nenhum deles.")}}
# títulos sem conteúdo: no Grafo de Crédito o JEV respondia com confiança alta e errado; não são enviados
GENERICOS = re.compile(r"^\s*(comunicado( ao mercado)?|fato relevante|outros comunicados.*|esclarecimentos?( sobre .{0,40})?|"
                       r"aviso aos acionistas|comunicado ao mercado - .{0,20})\s*[.:-]?\s*$", re.I)


def _baixar(ano: int) -> Path:
    f = _ZIPS / f"ipe_cia_aberta_{ano}.zip"
    if not f.exists() or (datetime.now().timestamp() - f.stat().st_mtime) > 6 * 3600:
        req = urllib.request.Request(_IPE.format(ano), headers={"User-Agent": "beneish-mscore"})
        with urllib.request.urlopen(req, timeout=300) as r:
            f.write_bytes(r.read())
    return f


def documentos(empresas: dict[str, dict]) -> list[dict]:
    """Fatos relevantes e comunicados dos últimos 12 meses, última versão de cada protocolo."""
    fmt = lambda c: f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:]}"
    alvo = {fmt(c): c for c in empresas}
    corte = (date.today() - timedelta(days=JANELA_DIAS)).isoformat()
    _ZIPS.mkdir(parents=True, exist_ok=True)
    por_protocolo: dict[str, dict] = {}
    for ano in sorted({date.today().year - 1, date.today().year}):
        try:
            z = zipfile.ZipFile(_baixar(ano))
        except Exception as exc:
            print(f"IPE {ano} indisponível: {exc}")
            continue
        nome = next(n for n in z.namelist() if n.endswith(".csv"))
        for l in csv.DictReader(io.TextIOWrapper(z.open(nome), encoding="latin-1"), delimiter=";"):
            cnpj = alvo.get(l["CNPJ_Companhia"])
            cat = CATEGORIAS.get(l["Categoria"])
            titulo = (l["Assunto"] or "").strip()
            if not cnpj or not cat or l["Data_Entrega"][:10] < corte or not titulo or GENERICOS.match(titulo):
                continue
            doc = {"id": l["Protocolo_Entrega"], "cnpj": cnpj, "categoria": cat, "data": l["Data_Entrega"][:10],
                   "titulo": titulo, "url": l["Link_Download"], "versao": int(l["Versao"] or 0)}
            atual = por_protocolo.get(doc["id"])
            if not atual or doc["versao"] > atual["versao"]:
                por_protocolo[doc["id"]] = doc
    return sorted(por_protocolo.values(), key=lambda d: d["data"], reverse=True)


def _chamar(corpo: dict) -> dict:
    chave = os.environ["TYPESAFE_API_KEY"]
    req = urllib.request.Request(f"{_API}/systemone", method="POST", data=json.dumps(corpo).encode(),
                                 headers={"Authorization": f"Bearer {chave}", "Content-Type": "application/json"})
    for tentativa in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and tentativa < 3:
                time.sleep(2 ** tentativa * 2)
                continue
            raise


def classificar(doc: dict, empresa: dict) -> dict:
    estado = (f"Companhia aberta brasileira: {empresa['nome']}. Setor: {empresa['setor']}. "
              f"Documento entregue à CVM: {doc['categoria']}, em {doc['data']}. Título: {doc['titulo']}")
    a = _chamar({"model": _MODELO, "state": estado, "questions": PERGUNTA})["answers"]["evento"]
    return {"evento": a["choice"], "confianca": round(a["confidence"], 3)}


def main() -> None:
    p = argparse.ArgumentParser(description="Eventos de qualidade contábil nos documentos da CVM (JEV).")
    p.add_argument("--contar", action="store_true", help="só conta o que seria enviado")
    p.add_argument("--limite", type=int, default=2500, help="máximo de documentos novos por execução")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    dados = json.loads(_DADOS.read_text(encoding="utf-8"))
    empresas = {e["cnpj"]: e for e in dados["empresas"] if e.get("mscore") is not None}
    docs = documentos(empresas)
    cache = json.loads(_CACHE.read_text(encoding="utf-8")) if _CACHE.exists() else {}
    novos = [d for d in docs if d["id"] not in cache]
    print(f"{len(docs)} documentos na janela de {JANELA_DIAS} dias; {len(novos)} ainda não classificados")

    if not a.contar:
        if not os.environ.get("TYPESAFE_API_KEY"):
            print("TYPESAFE_API_KEY não definida: nada é enviado; a página usa o que já está no cache")
        else:
            lote = novos[: a.limite]

            def um(d):
                try:
                    return d["id"], classificar(d, empresas[d["cnpj"]])
                except Exception as exc:  # um documento com erro não derruba o lote
                    print(f"erro em {d['id']}: {str(exc)[:120]}")
                    return d["id"], None

            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for i, (doc_id, r) in enumerate(pool.map(um, lote), 1):
                    if r:
                        cache[doc_id] = r
                    if i % 200 == 0:
                        _CACHE.parent.mkdir(parents=True, exist_ok=True)
                        _CACHE.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                        print(f"  {i}/{len(lote)}")
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            _CACHE.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            print(f"classificados nesta execução: {len(lote)}; restantes: {len(novos) - len(lote)}")

    por_empresa: dict[str, list] = {}
    for d in docs:
        r = cache.get(d["id"])
        if r and r["evento"] != "outro" and r["confianca"] >= CONFIANCA_MINIMA:
            por_empresa.setdefault(d["cnpj"], []).append(
                {"data": d["data"], "evento": r["evento"], "confianca": r["confianca"], "titulo": d["titulo"],
                 "categoria": d["categoria"], "url": d["url"]})
    classificados = sum(1 for d in docs if d["id"] in cache)
    saida = {
        "gerado_em": datetime.now(timezone(timedelta(hours=-3))).strftime("%d/%m/%Y %H:%M"),
        "janela_dias": JANELA_DIAS, "confianca_minima": CONFIANCA_MINIMA, "validado": False,
        "documentos": len(docs), "classificados": classificados,
        "eventos": {k: v for k, v in EVENTOS.items() if k != "outro"},
        "empresas": por_empresa,
    }
    _SAIDA.write_text(json.dumps(saida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"eventos.json: {sum(len(v) for v in por_empresa.values())} eventos em {len(por_empresa)} empresas "
          f"({classificados} de {len(docs)} documentos classificados)")


if __name__ == "__main__":
    sys.exit(main())
