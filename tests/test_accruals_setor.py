"""Qualidade dos lucros pelo desvio até a mediana do setor (modelo de setor), em módulo."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from advisor_brain_fsa.accruals import AlertLevel, combinar_alerta, medianas_por_setor, qualidade_por_desvio


def test_mediana_so_de_setor_grande():
    pares = [("A", -0.05)] * 10 + [("B", 0.20), ("B", 0.30)]
    med, geral, n = medianas_por_setor(pares)
    assert med == {"A": -0.05}           # B tem 2 empresas: fica sem mediana própria
    assert geral == -0.05 and n == {"A": 10, "B": 2}


def test_desvio_em_modulo():
    assert qualidade_por_desvio(0.0) == "Alta"
    assert qualidade_por_desvio(0.049) == "Alta"
    assert qualidade_por_desvio(-0.06) == "Moderada"     # abaixo da mediana também conta
    assert qualidade_por_desvio(0.10) == "Baixa"
    assert qualidade_por_desvio(-0.50) == "Baixa"        # big bath


def test_alerta():
    assert combinar_alerta(True, "Baixa") is AlertLevel.CRITICAL
    assert combinar_alerta(True, "Alta") is AlertLevel.HIGH_RISK
    assert combinar_alerta(False, "Baixa") is AlertLevel.WATCH
    assert combinar_alerta(False, "Moderada") is AlertLevel.NORMAL


def test_qualidade_setorial_no_lote():
    from build_static_site import _qualidade_setorial

    emp = [{"setor": "A", "accrual": -0.05, "mscore": -3.0, "indices": {"x": 1}, "erro": ""} for _ in range(10)]
    emp += [{"setor": "A", "accrual": -0.40, "mscore": -1.0, "indices": {"x": 1}, "erro": ""},
            {"setor": "Z", "accrual": 0.02, "mscore": -3.0, "indices": {"x": 1}, "erro": ""}]
    _qualidade_setorial(emp)
    assert emp[0]["qualidade"] == "Alta" and emp[0]["alerta"] == "Normal"
    assert emp[10]["qualidade"] == "Baixa" and emp[10]["alerta"] == "Crítico"
    assert emp[10]["accrual_setor"]["grupo"] == "A" and emp[10]["accrual_setor"]["n"] == 11
    assert emp[11]["accrual_setor"]["grupo"] == "" and emp[11]["qualidade"] == "Moderada"   # 0,02 − (−0,05) = 0,07
