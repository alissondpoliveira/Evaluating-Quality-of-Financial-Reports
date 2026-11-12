"""Tests for altman_zscore (Z''-Score de mercados emergentes). Sem rede: valores montados à mão."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.altman_zscore import calcular, zona


def test_formula_e_constante():
    # AT 1000, AC 400, PC 200, lucros retidos 300, EBIT 100, PL 500 (passivo 500)
    r = calcular(1000, 400, 200, 300, 100, 500)
    assert r.x1 == pytest.approx(0.2)
    assert r.x2 == pytest.approx(0.3)
    assert r.x3 == pytest.approx(0.1)
    assert r.x4 == pytest.approx(1.0)
    z = 6.56 * 0.2 + 3.26 * 0.3 + 6.72 * 0.1 + 1.05 * 1.0
    assert r.z == pytest.approx(z)
    assert r.em_score == pytest.approx(z + 3.25)
    assert sum(r.contribuicoes().values()) == pytest.approx(z)


def test_zonas_nos_limites():
    assert zona(5.86) == "Segura"
    assert zona(5.85) == "Cinzenta"
    assert zona(4.35) == "Cinzenta"
    assert zona(4.34) == "Risco"


def test_empresa_em_dificuldade():
    # capital de giro negativo, prejuízo acumulado, EBIT negativo, PL pequeno
    r = calcular(1000, 200, 500, -400, -50, 50)
    assert r.zona == "Risco"


def test_patrimonio_negativo_ainda_calcula():
    r = calcular(1000, 300, 600, -900, -20, -200)
    assert r.x4 == pytest.approx(-200 / 1200)
    assert r.zona == "Risco"


@pytest.mark.parametrize("args", [
    (1000, 400, 200, None, 100, 500),   # sem lucros retidos
    (1000, 400, 200, 300, None, 500),   # sem EBIT
    (1000, 400, 200, 300, 100, None),   # sem PL
    (0, 400, 200, 300, 100, 500),       # sem ativo
    (1000, 400, 200, 300, 100, 1000),   # passivo total zero
])
def test_sem_calculo(args):
    assert calcular(*args) is None
