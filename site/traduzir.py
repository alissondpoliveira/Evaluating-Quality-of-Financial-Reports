"""Traduz para o inglês o que o gerador deixou pendente (site/i18n/pendentes.json) e os artigos sem versão em inglês.

Usa a API do Claude (chave em ANTHROPIC_API_KEY ou CLAUDE_API_KEY). Grava em site/i18n/en.json (memória de
tradução, ver site/i18n.py) e em site/artigos/en/<slug>.json. Depois, rode site/gerar.py de novo.

Uso: python site/traduzir.py [pasta do site]       (padrão: a pasta deste arquivo)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

import anthropic

MODELO = os.getenv("MODELO_TRADUCAO", "claude-sonnet-5")
LOTE = 40

GLOSSARIO = """Glossário obrigatório (pt -> en):
debênture -> debenture; debêntures incentivadas (Lei 12.431) -> incentivized debentures (Law 12,431);
spread justo -> fair spread; pares / pares comparáveis -> peers / comparable peers; grupo de risco -> risk group;
emissor -> issuer; série -> series; escritura -> indenture; fato relevante -> material fact; agente fiduciário -> trustee;
gross-up -> gross-up; desvio-padrão (dp) -> standard deviation (sd); faixa principal -> main band; high yield -> high yield;
curva de juros -> yield curve; abertura / fechamento (de taxa ou spread) -> widening / tightening;
carrego -> carry; PU -> PU (unit price); taxa indicativa -> indicative rate; securitizadora -> securitization company;
lastro -> collateral pool; inadimplência -> delinquency; Grafo de Crédito -> Credit Graph;
Qualidade dos Lucros -> Earnings Quality; companhia aberta -> listed company; demonstrações -> financial statements;
provisão -> provision; ativos problemáticos -> problem assets; resultado não recorrente -> non-recurring income;
Índice de Basileia -> Basel ratio; conglomerado prudencial -> prudential conglomerate; Banco Central -> Central Bank;
Mercado (o site de edições diárias) -> Markets.
Mantenha em português: nomes de empresas, de pessoas, de leis e órgãos brasileiros (CVM, ANBIMA, SND, ANEEL, IF.data,
DFP, ITR, IPE, Copom, Selic, CDI, IPCA, NTN-B), siglas de indicadores (IPR, ICO, VRN, ICC, IAP, ICA, BPM, BAS), e
"Financial Syntax"."""

SISTEMA_LOTE = f"""Você traduz a interface e os textos de método de um site brasileiro de análise de crédito e de
qualidade contábil para inglês americano, para um leitor de finanças. Tom: direto, técnico, sem floreio.
{GLOSSARIO}
Regras:
- Recebe uma lista JSON de textos; devolve SÓ uma lista JSON de mesmo tamanho e mesma ordem, com as traduções.
- Os marcadores ⟦0⟧, ⟦1⟧... são tags HTML e ⟨0⟩, ⟨1⟩... são números: mantenha TODOS, exatamente iguais, cada um uma
  vez, na posição que a frase em inglês pedir. Não traduza nem altere marcadores.
- Texto que é fragmento (começa ou termina no meio da frase) continua fragmento, com o mesmo espaço nas pontas.
- Texto que já está em inglês, nome próprio ou sigla: devolva igual.
- Abreviação de dia da semana (seg., ter.) vira a inglesa (Mon, Tue)."""

SISTEMA_ARTIGO = f"""Você traduz artigos de análise de mercado de um autor brasileiro (CFP, candidato CFA) para inglês
americano, preservando a voz do autor: frases diretas, técnicas, sem floreio, sem acrescentar nem tirar conteúdo.
{GLOSSARIO}
Regras:
- O corpo é HTML: mantenha todas as tags, atributos, links e imagens exatamente como estão; traduza só o texto.
- Números: formato americano (1.234,5 -> 1,234.5; R$ 1,2 bi -> R$ 1.2 billion); datas por extenso em inglês.
- Responda SÓ com três blocos: <titulo>...</titulo> <resumo>...</resumo> <corpo>...</corpo>"""


def _cliente():
    chave = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not chave:
        sys.exit("Defina ANTHROPIC_API_KEY (ou CLAUDE_API_KEY).")
    return anthropic.Anthropic(api_key=chave)


def _pedir(cli, sistema: str, texto: str, max_tokens: int) -> str:
    # em streaming: a biblioteca recusa pedido sem streaming que possa passar de 10 minutos (artigos longos)
    with cli.messages.stream(model=MODELO, max_tokens=max_tokens, system=sistema, messages=[{"role": "user", "content": texto}]) as s:
        r = s.get_final_message()
    if r.stop_reason == "max_tokens":
        raise RuntimeError("resposta truncada")
    return "".join(b.text for b in r.content if b.type == "text")


def _marcadores(s: str) -> list[str]:
    return sorted(re.findall(r"⟦\d+⟧|⟨\d+⟩", s))


def traduzir_pendentes(cli, pasta_i18n: Path) -> None:
    arq_p, arq_m = pasta_i18n / "pendentes.json", pasta_i18n / "en.json"
    if not arq_p.exists():
        print("sem pendentes")
        return
    pend = json.loads(arq_p.read_text(encoding="utf-8"))
    mem = json.loads(arq_m.read_text(encoding="utf-8")) if arq_m.exists() else {}
    pend = [p for p in pend if p not in mem]
    falhas = []
    for i in range(0, len(pend), LOTE):
        lote = pend[i:i + LOTE]
        try:
            bruto = _pedir(cli, SISTEMA_LOTE, json.dumps(lote, ensure_ascii=False), 16000)
            res = json.loads(bruto[bruto.index("["):bruto.rindex("]") + 1])
        except Exception as e:
            print(f"lote {i}: falhou ({e}); vai item a item")
            res = []
            for p in lote:
                try:
                    b = _pedir(cli, SISTEMA_LOTE, json.dumps([p], ensure_ascii=False), 4000)
                    res.append(json.loads(b[b.index("["):b.rindex("]") + 1])[0])
                except Exception:
                    res.append(None)
        if len(res) != len(lote):
            print(f"lote {i}: {len(res)} respostas para {len(lote)} textos; descartado")
            falhas += lote
            continue
        for pt, en in zip(lote, res):
            if isinstance(en, str) and _marcadores(pt) == _marcadores(en):
                mem[pt] = en
            else:
                falhas.append(pt)
        arq_m.write_text(json.dumps(dict(sorted(mem.items())), ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{min(i + LOTE, len(pend))}/{len(pend)}")
    arq_p.write_text(json.dumps(falhas, ensure_ascii=False, indent=1), encoding="utf-8") if falhas else arq_p.unlink()
    print(f"traduzidos {len(pend) - len(falhas)}, falhas {len(falhas)}")


def traduzir_artigos(cli, pasta_artigos: Path) -> None:
    if not pasta_artigos.exists():
        return
    destino = pasta_artigos / "en"
    destino.mkdir(exist_ok=True)
    for arq in sorted(pasta_artigos.glob("*.json")):
        a = json.loads(arq.read_text(encoding="utf-8"))
        origem = hashlib.sha256((a["titulo"] + a["resumo"] + a["corpo"]).encode()).hexdigest()[:16]
        alvo = destino / arq.name
        if alvo.exists() and json.loads(alvo.read_text(encoding="utf-8")).get("origem") == origem:
            continue
        texto = f"<titulo>{a['titulo']}</titulo>\n<resumo>{a['resumo']}</resumo>\n<corpo>{a['corpo']}</corpo>"
        r = _pedir(cli, SISTEMA_ARTIGO, texto, 32000)
        campos = {k: re.search(rf"<{k}>(.*?)</{k}>", r, re.S) for k in ("titulo", "resumo", "corpo")}
        if not all(campos.values()):
            print(f"{arq.stem}: resposta sem os três blocos; fica para a próxima")
            continue
        en = {k: v.group(1).strip() for k, v in campos.items()}
        if len(re.findall(r"<img ", en["corpo"])) != len(re.findall(r"<img ", a["corpo"])):
            print(f"{arq.stem}: imagens não batem; fica para a próxima")
            continue
        alvo.write_text(json.dumps({**en, "origem": origem}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"artigo traduzido: {arq.stem}")


def main() -> None:
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    cli = _cliente()
    traduzir_pendentes(cli, site / "i18n")
    traduzir_artigos(cli, site / "artigos")


if __name__ == "__main__":
    main()
