"""
Últimos 12 meses (LTM) a partir das ITRs trimestrais da CVM
==========================================================
O M-Score, o accrual ratio e o Z-Score foram calibrados com dados anuais. Comparar um trimestre
com outro deixaria os fluxos (lucro, caixa, EBIT) em cerca de 1/4 do tamanho e distorceria os
pesos; a soma dos últimos 12 meses mantém a escala anual e se atualiza a cada trimestre:

    fluxo LTM até Q = exercício anual anterior (DFP) + acumulado do ano até Q (ITR)
                      − acumulado do mesmo período do ano anterior (comparativo na mesma ITR)

    contas de balanço = balanço da ITR em Q

A comparação é LTM até Q contra LTM até Q − 1 ano (ITR do ano anterior e DFP de dois anos
antes). Dezembro não tem ITR: ali os 12 meses são o próprio exercício da DFP, e quem chama usa
o caminho anual de CVMDataFetcher.

Portal CVM Dados Abertos — ITR: https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS/
"""

from __future__ import annotations

import io
import logging
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .beneish_mscore import FinancialData
from .cvm_accounts import ACCOUNT_SPECS, ALTMAN_ACCOUNT_SPECS, AccountSpec
from .data_fetcher import CVMDataFetcher

logger = logging.getLogger(__name__)

ITR_BASE_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/ITR/DADOS"
_STATEMENTS = ("BPA", "BPP", "DRE", "DFC_MI")
_FLUXOS = ("DRE", "DFC_MI")          # acumulam no período; BPA/BPP são posição na data
_VAZIO = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA"]
_COBERTURA_MINIMA = 0.7              # fração das companhias que já entregaram para o trimestre valer


def fim_de_trimestre(d: date) -> date:
    """Último fim de trimestre (31/3, 30/6, 30/9, 31/12) até `d`, inclusive."""
    for m, dia in ((12, 31), (9, 30), (6, 30), (3, 31)):
        f = date(d.year, m, dia)
        if f <= d:
            return f
    return date(d.year - 1, 12, 31)


def trimestre_anterior(q: date) -> date:
    return fim_de_trimestre(date(q.year, q.month, 1) - timedelta(days=1))


def um_ano_antes(d: date) -> date:
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29/2
        return d.replace(year=d.year - 1, day=28)


def cadeia(ref: date) -> List[date]:
    """Períodos tentados a partir de `ref`, do mais recente ao último 31/12 (anual), inclusive."""
    saida, q = [ref], ref
    while not (q.month == 12):
        q = trimestre_anterior(q)
        saida.append(q)
    return saida


@dataclass
class ResultadoLTM:
    t: FinancialData
    t1: FinancialData
    altman_t: Dict[str, Optional[float]]
    fim: date             # fim dos 12 meses atuais
    fim_ant: date         # fim dos 12 meses anteriores
    individual: bool      # demonstrações individuais (sem consolidado com valores)


class LTMFetcher:
    """Monta FinancialData dos últimos 12 meses com ITR + DFP, reaproveitando o coletor anual
    (cache de arquivos, leitura dos CSVs e resolução de contas)."""

    def __init__(self, dfp: CVMDataFetcher):
        self.dfp = dfp
        self.cache_dir = dfp.cache_dir
        self._itr: Dict[Tuple[int, str, str], pd.DataFrame] = {}
        self._idx: Dict[Tuple[str, int, str, str], Dict[str, pd.DataFrame]] = {}
        self._trava = threading.Lock()   # o lote roda em threads: cada arquivo é lido uma vez

    # ── arquivos ────────────────────────────────────────────────────────────

    def _baixar_itr(self, ano: int):
        dest = self.cache_dir / f"itr_cia_aberta_{ano}.zip"
        if dest.exists() and not self.dfp.force_download:
            return dest
        url = f"{ITR_BASE_URL}/itr_cia_aberta_{ano}.zip"
        logger.info("Downloading %s → %s", url, dest)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "advisor-brain-fsa/1.0 (financial research)"})
            with urllib.request.urlopen(req, timeout=self.dfp.timeout) as resp:
                dest.write_bytes(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None   # ano sem nenhuma ITR ainda (janeiro a maio)
            raise
        return dest

    def _carregar_itr(self, ano: int, stmt: str, mod: str) -> pd.DataFrame:
        chave = (ano, stmt, mod)
        if chave in self._itr:
            return self._itr[chave]
        caminho = self._baixar_itr(ano)
        df = pd.DataFrame()
        if caminho is not None:
            nome = f"itr_cia_aberta_{stmt}_{mod}_{ano}.csv"
            with zipfile.ZipFile(caminho) as zf:
                achados = [n for n in zf.namelist() if n.lower() == nome.lower()]
                if achados:
                    df = pd.read_csv(io.BytesIO(zf.read(achados[0])), sep=";", encoding="latin-1",
                                     dtype={"CD_CONTA": str, "CNPJ_CIA": str}, low_memory=False)
                    df.columns = df.columns.str.strip()
                    for c in ("DT_REFER", "DT_INI_EXERC", "DT_FIM_EXERC"):
                        if c in df.columns:
                            df[c] = pd.to_datetime(df[c], errors="coerce")
                    df["VL_CONTA"] = pd.to_numeric(df["VL_CONTA"], errors="coerce")
        self._itr[chave] = df
        return df

    def _por_cnpj(self, tipo: str, ano: int, stmt: str, mod: str) -> Dict[str, pd.DataFrame]:
        chave = (tipo, ano, stmt, mod)
        with self._trava:
            return self._indexar(chave, tipo, ano, stmt, mod)

    def _indexar(self, chave, tipo, ano, stmt, mod) -> Dict[str, pd.DataFrame]:
        if chave not in self._idx:
            df = self._carregar_itr(ano, stmt, mod) if tipo == "itr" else self._carregar_dfp(ano, stmt, mod)
            if df.empty:
                self._idx[chave] = {}
            else:
                digitos = df["CNPJ_CIA"].str.replace(r"\D", "", regex=True)
                self._idx[chave] = {k: g for k, g in df.groupby(digitos)}
        return self._idx[chave]

    def _carregar_dfp(self, ano: int, stmt: str, mod: str) -> pd.DataFrame:
        try:
            return self.dfp._load_statement(ano, stmt, mod)
        except (RuntimeError, FileNotFoundError):
            return pd.DataFrame()

    # ── período de referência ──────────────────────────────────────────────

    def cobertura(self, q: date) -> int:
        """Companhias com demonstração entregue para a data-base `q` (ITR, ou DFP em dezembro)."""
        tipo = "dfp" if q.month == 12 else "itr"
        df = self._carregar_dfp(q.year, "DRE", "con") if tipo == "dfp" else self._carregar_itr(q.year, "DRE", "con")
        if df.empty:
            return 0
        return int(df.loc[df["DT_REFER"] == pd.Timestamp(q), "CNPJ_CIA"].nunique())

    def periodo_referencia(self, hoje: Optional[date] = None) -> date:
        """Fim de trimestre mais recente que a maioria das companhias já entregou."""
        q = fim_de_trimestre(hoje or date.today())
        candidatos = []
        for _ in range(6):
            candidatos.append((q, self.cobertura(q)))
            q = trimestre_anterior(q)
        base = max(n for _, n in candidatos) or 1
        for q, n in candidatos:
            logger.info("Data-base %s: %d companhias entregaram", q, n)
            if n >= _COBERTURA_MINIMA * base:
                return q
        raise RuntimeError("Nenhuma data-base com demonstrações suficientes nos últimos seis trimestres")

    # ── montagem ───────────────────────────────────────────────────────────

    @staticmethod
    def _ultima_versao(g: pd.DataFrame, refer: date) -> pd.DataFrame:
        g = g[g["DT_REFER"] == pd.Timestamp(refer)]
        if g.empty:
            return g
        return g[g["VERSAO"] == g["VERSAO"].max()]

    def _itr_da_empresa(self, cnpj: str, q: date, mod: str) -> Dict[str, pd.DataFrame]:
        return {s: self._ultima_versao(self._por_cnpj("itr", q.year, s, mod).get(cnpj, pd.DataFrame(columns=_VAZIO)), q)
                for s in _STATEMENTS}

    def _anual(self, cnpj: str, fim: date, mod: str) -> Dict[str, pd.DataFrame]:
        """DFP do exercício encerrado em `fim` (só a coluna do próprio exercício)."""
        saida = {}
        for s in _FLUXOS:
            g = self._ultima_versao(self._por_cnpj("dfp", fim.year, s, mod).get(cnpj, pd.DataFrame(columns=_VAZIO)), fim)
            if not g.empty:
                g = g[(g["DT_FIM_EXERC"] == pd.Timestamp(fim)) & ((g["DT_FIM_EXERC"] - g["DT_INI_EXERC"]).dt.days >= 330)]
            saida[s] = g
        return saida

    @staticmethod
    def _tem_ativo(bpa: pd.DataFrame, q: date) -> bool:
        if bpa.empty:
            return False
        v = bpa.loc[(bpa["CD_CONTA"] == "1") & (bpa["DT_FIM_EXERC"] == pd.Timestamp(q)), "VL_CONTA"].dropna()
        return bool((v != 0).any())

    def _pecas(self, cnpj: str, q: date, mod: str):
        """Balanço em q, acumulado do ano até q, acumulado do ano anterior e o exercício anual anterior."""
        itr = self._itr_da_empresa(cnpj, q, mod)
        if not self._tem_ativo(itr["BPA"], q):
            return None
        dre = itr["DRE"]
        atual = dre[dre["DT_FIM_EXERC"] == pd.Timestamp(q)]
        if atual.empty:
            return None
        inicio = atual["DT_INI_EXERC"].min().date()        # início do exercício (nem sempre 1º/1)
        q0, inicio0 = um_ano_antes(q), um_ano_antes(inicio)
        ts = pd.Timestamp
        balanco = {s: itr[s][itr[s]["DT_FIM_EXERC"] == ts(q)] for s in ("BPA", "BPP")}
        acum = {s: itr[s][(itr[s]["DT_FIM_EXERC"] == ts(q)) & (itr[s]["DT_INI_EXERC"] == ts(inicio))] for s in _FLUXOS}
        acum0 = {s: itr[s][(itr[s]["DT_FIM_EXERC"] == ts(q0)) & (itr[s]["DT_INI_EXERC"] == ts(inicio0))] for s in _FLUXOS}
        anual = self._anual(cnpj, inicio - timedelta(days=1), mod)
        if any(f.empty for f in (*acum.values(), *acum0.values(), *anual.values())):
            return None
        return balanco, acum, acum0, anual

    def _resolver(self, specs: List[AccountSpec], pecas, rotulo: str) -> Dict[str, float]:
        balanco, acum, acum0, anual = pecas
        r = self.dfp._resolve_account
        saida: Dict[str, float] = {}
        for spec in specs:
            if spec.statement in _FLUXOS:
                v = r(anual[spec.statement], spec) + r(acum[spec.statement], spec) - r(acum0[spec.statement], spec)
            else:
                v = r(balanco[spec.statement], spec)
            saida[spec.field_name] = v
        return saida

    def _financial_data(self, pecas, q: date) -> FinancialData:
        v = self._resolver(ACCOUNT_SPECS, pecas, str(q))
        faltam = [k for k, x in v.items() if np.isnan(x)]
        if faltam:
            logger.warning("Year %d (12 meses até %s) — could not resolve: %s. Will substitute 0.0.", q.year, q, faltam)
            for k in faltam:
                v[k] = 0.0
        return FinancialData(**v)

    def get(self, cnpj: str, q: date) -> ResultadoLTM:
        """12 meses até `q` contra 12 meses até `q` − 1 ano. LookupError se faltar alguma peça."""
        if q.month == 12:
            raise LookupError("dezembro: use o exercício anual da DFP")
        cnpj = "".join(ch for ch in cnpj if ch.isdigit())
        q0 = um_ano_antes(q)
        for mod in ("con", "ind"):
            p, p0 = self._pecas(cnpj, q, mod), self._pecas(cnpj, q0, mod)
            if p is not None and p0 is not None:
                break
        else:
            raise LookupError(f"ITR ou DFP ausente para os 12 meses até {q} ou {q0}")
        if mod == "ind":
            logger.warning("Year %d — consolidated statements empty for '%s'; using individual statements.", q.year, cnpj)
            logger.warning("Year %d — consolidated statements empty for '%s'; using individual statements.", q0.year, cnpj)
        extra = self._resolver(ALTMAN_ACCOUNT_SPECS, p, str(q))
        return ResultadoLTM(
            t=self._financial_data(p, q), t1=self._financial_data(p0, q0),
            altman_t={k: None if np.isnan(x) else x for k, x in extra.items()},
            fim=q, fim_ant=q0, individual=(mod == "ind"))
