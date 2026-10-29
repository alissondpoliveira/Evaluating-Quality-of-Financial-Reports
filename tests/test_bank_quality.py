"""
Tests for bank_quality (triagem de qualidade dos lucros de instituições financeiras).

Sem rede: os períodos são montados à mão, com os mesmos nomes de campo do coletor do IF.data.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa import bank_quality
from advisor_brain_fsa.bank_quality import avaliar, carteira, cobertura, eixos, indicadores, nivel, provisao


@pytest.fixture(autouse=True)
def _sem_piso(monkeypatch):
    # os períodos de teste usam valores pequenos; o piso de carteira tem teste próprio
    monkeypatch.setattr(bank_quality, "_CARTEIRA_MINIMA", 0)


def _periodo(**k):
    base = dict(ativo_total=1000.0, patrimonio_liquido=100.0, depositos=600.0, credito_bruto=500.0,
                credito_perda_esperada=-30.0, carteira_total=500.0, ativos_problematicos=25.0,
                inadimplencia=15.0, perda_esperada_credito=-10.0, juros_credito=60.0, juros_tvm=20.0,
                tarifas=10.0, resultado_intermediacao=40.0, lucro_liquido=12.0, basileia=0.14)
    base.update(k)
    return base


class TestComponentes:
    def test_carteira_soma_credito_arrendamento_e_cartao(self):
        assert carteira(_periodo(arrendamento_bruto=20.0, pagamentos_pos_bruto=30.0)) == 550.0

    def test_provisao_positiva_quando_ha_despesa(self):
        assert provisao(_periodo()) == 10.0

    def test_cobertura_estoque_sobre_problematicos(self):
        assert cobertura(_periodo()) == pytest.approx(30.0 / 25.0)

    def test_indicadores_anualiza_semestre(self):
        ind = indicadores(_periodo(), meses=6)
        assert ind["roe"] == pytest.approx(12.0 * 2 / 100.0)
        assert ind["custo_risco"] == pytest.approx(10.0 * 2 / 500.0)
        assert ind["problematicos"] == pytest.approx(0.05)


class TestAvaliar:
    def test_periodos_iguais_sem_sinal(self):
        r = avaliar(_periodo(), _periodo(), meses=6)
        assert r["avaliavel"] and r["sinais"] == [] and r["alerta"] == "Normal"
        assert r["indices"]["ipr"] == pytest.approx(1.0) and r["indices"]["vrn"] == pytest.approx(0.0)

    def test_provisao_cai_com_risco_subindo(self):
        # taxa de provisão 2% → 1% (IPR = 2,0) e problemáticos 5% → 7% (IAP = 1,4, acima de 3%)
        t1 = _periodo()
        t = _periodo(perda_esperada_credito=-5.0, ativos_problematicos=35.0)
        r = avaliar(t, t1, meses=6)
        assert r["indices"]["ipr"] == pytest.approx(2.0)
        assert r["indices"]["iap"] == pytest.approx(1.4)
        # cobertura 30/25 = 1,2 → 30/35: ICO = 1,4
        assert set(r["sinais"]) == {"ipr", "iap", "ico"}
        assert r["alerta"] == "Crítico"

    def test_provisao_cai_com_risco_caindo_nao_e_sinal(self):
        t = _periodo(perda_esperada_credito=-5.0, ativos_problematicos=20.0)
        r = avaliar(t, _periodo(), meses=6)
        assert "ipr" not in r["sinais"]

    def test_reversao_de_provisao_conta_como_queda(self):
        r = avaliar(_periodo(perda_esperada_credito=2.0), _periodo(), meses=6)
        assert r["indices"]["ipr"] is None and "ipr" in r["sinais"]

    def test_crescimento_acelerado(self):
        r = avaliar(_periodo(credito_bruto=700.0, carteira_total=700.0, ativos_problematicos=35.0), _periodo(), meses=6)
        assert r["indices"]["icc"] == pytest.approx(1.4) and "icc" in r["sinais"]

    def test_capital_caindo_so_e_sinal_abaixo_de_15(self):
        alto = avaliar(_periodo(basileia=0.18), _periodo(basileia=0.22), meses=6)
        baixo = avaliar(_periodo(basileia=0.12), _periodo(basileia=0.14), meses=6)
        assert "ica" not in alto["sinais"] and "ica" in baixo["sinais"]

    def test_basileia_negativa_acende_capital_e_minimo(self):
        r = avaliar(_periodo(basileia=-0.05), _periodo(basileia=0.18), meses=6)
        assert {"ica", "bas"} <= set(r["sinais"])

    def test_basileia_perto_do_minimo(self):
        r = avaliar(_periodo(basileia=0.12), _periodo(basileia=0.12), meses=6)
        assert "bpm" in r["sinais"] and "bas" not in r["sinais"] and r["alerta"] == "Atenção"

    def test_patrimonio_negativo_e_capital_abaixo_do_minimo(self):
        r = avaliar(_periodo(patrimonio_liquido=-5.0, basileia=0.14), _periodo(), meses=6)
        assert r["avaliavel"] and "bas" in r["sinais"] and r["alerta"] == "Crítico"

    def test_basileia_abaixo_do_minimo(self):
        assert "bas" in avaliar(_periodo(basileia=0.10), _periodo(basileia=0.10), meses=6)["sinais"]

    def test_nao_recorrente_sobe_e_chega_ao_lucro(self):
        # receita 90 nos dois; outras receitas 0 → 18: VRN = 0,2; resultado antes do IR +12 (repasse 2/3)
        r = avaliar(_periodo(outras_receitas=18.0, resultado_antes_ir=32.0), _periodo(resultado_antes_ir=20.0), meses=6)
        assert r["indices"]["vrn"] == pytest.approx(0.2) and r["indices"]["rep"] == pytest.approx(12 / 18)
        assert "vrn" in r["sinais"]

    def test_nao_recorrente_compensado_nao_e_sinal(self):
        # mesma virada, mas o resultado antes do IR só sobe 2 (hedge compensando): repasse 11%
        r = avaliar(_periodo(outras_receitas=18.0, resultado_antes_ir=22.0), _periodo(resultado_antes_ir=20.0), meses=6)
        assert r["indices"]["vrn"] == pytest.approx(0.2) and "vrn" not in r["sinais"]

    def test_sem_carteira_nao_avaliavel(self):
        r = avaliar(_periodo(credito_bruto=0.0, carteira_total=0.0), _periodo(), meses=6)
        assert not r["avaliavel"] and "alerta" not in r

    def test_carteira_abaixo_do_piso_nao_avaliavel(self, monkeypatch):
        monkeypatch.setattr(bank_quality, "_CARTEIRA_MINIMA", 10e6)
        r = avaliar(_periodo(), _periodo(), meses=6)
        assert not r["avaliavel"] and "10 milhões" in r["motivo"]


class TestNivel:
    def test_um_eixo_so_e_atencao(self):
        assert nivel(["iap", "ica"]) == "Atenção" and nivel(["vrn"]) == "Atenção"

    def test_dois_sinais_de_lucro_e_alto_risco(self):
        assert nivel(["ipr", "vrn"]) == "Lucro a conferir"

    def test_lucro_e_solidez_e_critico(self):
        assert nivel(["ipr", "iap"]) == "Crítico"

    def test_basileia_abaixo_do_minimo_e_critico(self):
        assert nivel(["bas"]) == "Crítico"

    def test_cobertura_com_problematicos_e_um_evento_de_solidez(self):
        assert eixos(["ico", "iap"]) == {"lucro": [], "solidez": ["ico", "iap"]}
        assert nivel(["ico", "iap"]) == "Atenção"
        assert nivel(["ico"]) == "Atenção" and nivel(["ico", "ipr"]) == "Lucro a conferir"
