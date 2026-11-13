"""Tests for ltm (últimos 12 meses com ITR + DFP). Sem rede: arquivos da CVM montados à mão no cache."""

from __future__ import annotations

import os
import sys
import zipfile
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.data_fetcher import CVMDataFetcher
from advisor_brain_fsa.ltm import LTMFetcher, cadeia, fim_de_trimestre, um_ano_antes

CNPJ = "11222333000144"
COLS = "CNPJ_CIA;DT_REFER;VERSAO;DENOM_CIA;ORDEM_EXERC;DT_INI_EXERC;DT_FIM_EXERC;CD_CONTA;DS_CONTA;VL_CONTA"


def _linha(refer, ini, fim, conta, valor, versao=1):
    return f"11.222.333/0001-44;{refer};{versao};EMPRESA TESTE SA;ULTIMO;{ini};{fim};{conta};X;{valor}"


def _zip(caminho, arquivos):
    with zipfile.ZipFile(caminho, "w") as z:
        for nome, linhas in arquivos.items():
            z.writestr(nome, "\n".join([COLS, *linhas]))


def _balanco(refer, fim, ativo, circ, pc, pl):
    return ([_linha(refer, "", fim, "1", ativo), _linha(refer, "", fim, "1.01", circ)],
            [_linha(refer, "", fim, "2.01", pc), _linha(refer, "", fim, "2.03", pl)])


def _dfp(tmp, ano, receita, lucro, caixa, ebit):
    fim, ini = f"{ano}-12-31", f"{ano}-01-01"
    bpa, bpp = _balanco(fim, fim, 1000, 400, 200, 500)
    dre = [_linha(fim, ini, fim, "3.01", receita), _linha(fim, ini, fim, "3.05", ebit), _linha(fim, ini, fim, "3.11", lucro)]
    dfc = [_linha(fim, ini, fim, "6.01", caixa)]
    _zip(tmp / f"dfp_cia_aberta_{ano}.zip", {f"dfp_cia_aberta_{s}_con_{ano}.csv": l
                                             for s, l in (("BPA", bpa), ("BPP", bpp), ("DRE", dre), ("DFC_MI", dfc))})


def _itr(tmp, ano, acum, acum_ant, ativo):
    """ITR de 30/6: acumulado do ano (receita, lucro, ebit, caixa) e o comparativo do ano anterior."""
    q, q0 = f"{ano}-06-30", f"{ano - 1}-06-30"
    bpa, bpp = _balanco(q, q, ativo, 500, 250, 600)
    bpa.append(_linha(q, "", f"{ano - 1}-12-31", "1", 999999))      # comparativo de dezembro: ignorado
    dre, dfc = [], []
    for fim, ini, (rec, ll, ebit, cx) in ((q, f"{ano}-01-01", acum), (q0, f"{ano - 1}-01-01", acum_ant)):
        dre += [_linha(q, ini, fim, "3.01", rec), _linha(q, ini, fim, "3.05", ebit), _linha(q, ini, fim, "3.11", ll)]
        dfc.append(_linha(q, ini, fim, "6.01", cx))
    dre.append(_linha(q, f"{ano}-04-01", q, "3.01", 12345))             # coluna só do trimestre: ignorada
    _zip(tmp / f"itr_cia_aberta_{ano}.zip", {f"itr_cia_aberta_{s}_con_{ano}.csv": l
                                             for s, l in (("BPA", bpa), ("BPP", bpp), ("DRE", dre), ("DFC_MI", dfc))})


@pytest.fixture
def ltm(tmp_path):
    _dfp(tmp_path, 2024, receita=90, lucro=9, caixa=12, ebit=15)
    _dfp(tmp_path, 2025, receita=100, lucro=10, caixa=14, ebit=18)
    _itr(tmp_path, 2025, acum=(50, 5, 8, 6), acum_ant=(40, 4, 7, 5), ativo=1100)
    _itr(tmp_path, 2026, acum=(60, 7, 11, 9), acum_ant=(50, 5, 8, 6), ativo=1200)
    return LTMFetcher(CVMDataFetcher(cache_dir=tmp_path))


def test_soma_dos_ultimos_12_meses(ltm):
    r = ltm.get(CNPJ, date(2026, 6, 30))
    assert (r.fim, r.fim_ant, r.individual) == (date(2026, 6, 30), date(2025, 6, 30), False)
    assert r.t.revenues == 100 + 60 - 50
    assert r.t.net_income == 10 + 7 - 5
    assert r.t.cash_from_operations == 14 + 9 - 6
    assert r.altman_t["ebit"] == 18 + 11 - 8
    assert r.t1.revenues == 90 + 50 - 40
    assert r.t1.net_income == 9 + 5 - 4


def test_balanco_do_fim_do_trimestre(ltm):
    r = ltm.get(CNPJ, date(2026, 6, 30))
    assert r.t.total_assets == 1200 and r.t1.total_assets == 1100
    assert r.altman_t["shareholders_equity"] == 600


def test_trimestre_sem_itr_levanta_lookup(ltm):
    with pytest.raises(LookupError):
        ltm.get(CNPJ, date(2026, 9, 30))
    with pytest.raises(LookupError):
        ltm.get(CNPJ, date(2025, 12, 31))


def test_cobertura_e_referencia(ltm):
    assert ltm.cobertura(date(2026, 6, 30)) == 1
    assert ltm.periodo_referencia(date(2026, 8, 10)) == date(2026, 6, 30)


def test_datas():
    assert fim_de_trimestre(date(2026, 11, 13)) == date(2026, 9, 30)
    assert fim_de_trimestre(date(2026, 2, 1)) == date(2025, 12, 31)
    assert cadeia(date(2026, 9, 30)) == [date(2026, 9, 30), date(2026, 6, 30), date(2026, 3, 31), date(2025, 12, 31)]
    assert cadeia(date(2025, 12, 31)) == [date(2025, 12, 31)]
    assert um_ano_antes(date(2028, 2, 29)) == date(2027, 2, 28)
