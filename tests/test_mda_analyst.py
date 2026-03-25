"""
Tests for mda_analyst.py — all API calls mocked.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from advisor_brain_fsa.mda_analyst import (
    PromptBuilder,
    MDAnalyst,
    compute_grade,
    _SYSTEM_PROMPT,
)
from advisor_brain_fsa.beneish_mscore import MScoreResult
from advisor_brain_fsa.accruals import AlertLevel, CashFlowQualityResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mscore(m_score: float = -2.5) -> MScoreResult:
    return MScoreResult(
        dsri=1.05, gmi=1.02, aqi=1.03, sgi=1.10,
        depi=1.01, sgai=1.02, lvgi=1.01, tata=0.02,
        m_score=m_score,
    )


def _make_cfq(accrual_ratio: float = 0.02, quality: str = "Moderada") -> CashFlowQualityResult:
    alert = (
        AlertLevel.CRITICAL   if quality == "Baixa" and _make_mscore().is_manipulator else
        AlertLevel.HIGH_RISK  if _make_mscore().is_manipulator else
        AlertLevel.WATCH      if quality == "Baixa" else
        AlertLevel.NORMAL
    )
    return CashFlowQualityResult(
        accrual_ratio=accrual_ratio,
        earnings_quality=quality,
        alert_level=alert,
    )


@pytest.fixture
def mscore_normal():
    return _make_mscore(-2.5)


@pytest.fixture
def mscore_risky():
    return _make_mscore(-1.0)


@pytest.fixture
def cfq_good():
    return _make_cfq(0.005, "Alta")


@pytest.fixture
def cfq_bad():
    return _make_cfq(0.08, "Baixa")


@pytest.fixture
def red_flags():
    return [
        "DSRI elevado (1.82) — recebíveis crescem mais rápido que receita",
        "TATA alto (0.07) — accruals elevados vs. ativos totais",
        "GMI deteriorado (1.20) — margem bruta em queda",
    ]


# ---------------------------------------------------------------------------
# compute_grade
# ---------------------------------------------------------------------------

class TestComputeGrade:
    def test_grade_a_excellent(self):
        grade, _ = compute_grade(-3.0, 0.005)
        assert grade == "A"

    def test_grade_b_good(self):
        grade, _ = compute_grade(-2.3, 0.02)
        assert grade == "B"

    def test_grade_c_questionable(self):
        grade, _ = compute_grade(-2.0, 0.04)
        assert grade == "C"

    def test_grade_d_manipulator_low_accruals(self):
        grade, _ = compute_grade(-1.5, 0.03)
        assert grade == "D"

    def test_grade_f_manipulator_high_accruals(self):
        grade, _ = compute_grade(-1.0, 0.10)
        assert grade == "F"

    def test_grade_f_label_is_critical(self):
        _, label = compute_grade(-1.0, 0.10)
        assert "crítico" in label.lower() or "fraude" in label.lower()

    def test_grade_a_label_mentions_manipulation(self):
        _, label = compute_grade(-3.0, 0.005)
        assert "manipulação" in label.lower() or "exemplar" in label.lower()

    def test_boundary_mscore_exactly_minus178(self):
        # M-Score exactly -1.78 → NOT a manipulator → depends on accruals
        grade, _ = compute_grade(-1.78, 0.03)
        assert grade in ("B", "C")  # not D or F

    def test_all_grades_return_tuple(self):
        for m, a in [(-3.0, 0.005), (-2.3, 0.02), (-1.9, 0.03), (-1.5, 0.03), (-1.0, 0.10)]:
            result = compute_grade(m, a)
            assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def _build(self, m_score=-2.5, accrual_ratio=0.02):
        ms = _make_mscore(m_score)
        cfq = _make_cfq(accrual_ratio)
        grade, label = compute_grade(m_score, accrual_ratio)
        return PromptBuilder().build(
            ticker="PETR4", sector="Energia", year=2023,
            mscore_result=ms, cfq_result=cfq,
            red_flags=["DSRI elevado (1.82)", "TATA alto (0.07)"],
            grade=grade, grade_label=label,
        )

    def test_contains_ticker(self):
        prompt = self._build()
        assert "PETR4" in prompt

    def test_contains_sector(self):
        prompt = self._build()
        assert "Energia" in prompt

    def test_contains_year(self):
        prompt = self._build()
        assert "2023" in prompt

    def test_contains_all_beneish_indices(self):
        prompt = self._build()
        for index in ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]:
            assert index in prompt

    def test_contains_mscore_value(self):
        prompt = self._build(-2.5)
        assert "-2.5" in prompt or "−2.5" in prompt or "-2.50" in prompt

    def test_contains_accrual_ratio(self):
        prompt = self._build(accrual_ratio=0.02)
        assert "0.02" in prompt or "0.0200" in prompt

    def test_contains_red_flags(self):
        prompt = self._build()
        assert "DSRI elevado" in prompt
        assert "TATA alto" in prompt

    def test_contains_grade(self):
        prompt = self._build(-2.5, 0.005)
        assert "A" in prompt  # grade A

    def test_contains_manipulation_threshold(self):
        prompt = self._build()
        assert "−1.78" in prompt or "-1.78" in prompt

    def test_empty_red_flags_handled(self):
        ms = _make_mscore()
        cfq = _make_cfq()
        grade, label = compute_grade(-2.5, 0.02)
        prompt = PromptBuilder().build(
            ticker="X", sector="Y", year=2023,
            mscore_result=ms, cfq_result=cfq,
            red_flags=[], grade=grade, grade_label=label,
        )
        assert "Nenhum red flag" in prompt


# ---------------------------------------------------------------------------
# MDAnalyst — mocked API
# ---------------------------------------------------------------------------

def _make_stream_mock(text_chunks: list[str]):
    """Create a mock for the anthropic stream context manager."""
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(text_chunks)
    return mock_stream


AI_RESPONSE_CHUNKS = [
    "## 1. Resumo da Fragilidade\n\n",
    "**Nota Qualitativa: B**\n\n",
    "O lucro da empresa apresenta qualidade razoável.\n\n",
    "---\n## 2. Onde Está o \"Batom na Cueca\"\n\n",
    "**Conta Suspeita: Contas a Receber**\n\n",
    "O DSRI elevado sugere inflação de receitas.\n\n",
    "---\n## 3. Perguntas para o Diretor de RI\n\n",
    "1. Qual o prazo médio de recebimento?\n",
    "2. Houve mudança na política de crédito?\n",
    "3. Qual a provisão para devedores duvidosos?\n",
]


class TestMDAnalystMocked:
    def _patch_anthropic(self, text_chunks=None):
        """Return a context manager patch for the anthropic module."""
        chunks = text_chunks or AI_RESPONSE_CHUNKS
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = _make_stream_mock(chunks)

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        return mock_anthropic_module, mock_client

    def test_analyze_returns_string(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            result = analyst.analyze(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        assert isinstance(result, str) and len(result) > 0

    def test_analyze_contains_header(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            result = analyst.analyze(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        assert "Tese de Risco" in result
        assert "PETR4" in result
        assert "2023" in result

    def test_analyze_contains_footer(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            result = analyst.analyze(
                ticker="VALE3", sector="Mineração", year=2022,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        assert "Advisor-Brain-FSA" in result
        assert "CFA Level 2" in result

    def test_analyze_contains_ai_content(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            result = analyst.analyze(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        assert "Resumo da Fragilidade" in result
        assert "Batom na Cueca" in result
        assert "Perguntas para o Diretor de RI" in result

    def test_streaming_yields_chunks(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic(["chunk1", "chunk2", "chunk3"])
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            chunks = list(analyst.analyze_streaming(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            ))
        # First chunk is header, last is footer, middle are AI chunks
        assert any("chunk1" in c for c in chunks)
        assert any("chunk2" in c for c in chunks)

    def test_missing_api_key_raises(self, mscore_normal, cfq_good, red_flags):
        mock_module, _ = self._patch_anthropic()
        analyst = MDAnalyst(api_key="")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            with patch.dict(os.environ, {}, clear=True):
                analyst.api_key = ""
                with pytest.raises(ValueError, match="API key"):
                    analyst.analyze(
                        ticker="PETR4", sector="Energia", year=2023,
                        mscore_result=mscore_normal, cfq_result=cfq_good,
                        red_flags=red_flags,
                    )

    def test_api_called_with_correct_model(self, mscore_normal, cfq_good, red_flags):
        mock_module, mock_client = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test", model="claude-opus-4-6")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            analyst.analyze(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_adaptive_thinking_enabled(self, mscore_normal, cfq_good, red_flags):
        mock_module, mock_client = self._patch_anthropic()
        analyst = MDAnalyst(api_key="sk-test")
        with patch("advisor_brain_fsa.mda_analyst._get_anthropic", return_value=mock_module):
            analyst.analyze(
                ticker="PETR4", sector="Energia", year=2023,
                mscore_result=mscore_normal, cfq_result=cfq_good,
                red_flags=red_flags,
            )
        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert call_kwargs.get("thinking") == {"type": "adaptive"}

    def test_system_prompt_contains_cfa_reference(self):
        assert "CFA Level 2" in _SYSTEM_PROMPT
        assert "Beneish" in _SYSTEM_PROMPT
        assert "português brasileiro" in _SYSTEM_PROMPT.lower() or "português" in _SYSTEM_PROMPT

    def test_system_prompt_defines_output_sections(self):
        assert "Resumo da Fragilidade" in _SYSTEM_PROMPT
        assert "Batom na Cueca" in _SYSTEM_PROMPT
        assert "Perguntas para o Diretor de RI" in _SYSTEM_PROMPT

    def test_report_header_content(self, mscore_normal, cfq_good):
        header = MDAnalyst._report_header(
            ticker="VALE3", year=2023, grade="B", grade_label="Boa qualidade",
            mscore=mscore_normal, cfq=cfq_good,
        )
        assert "VALE3" in header
        assert "2023" in header
        assert "B" in header
        assert "M-Score" in header

    def test_report_footer_content(self):
        footer = MDAnalyst._report_footer("PETR4", 2023, "claude-opus-4-6")
        assert "PETR4" in footer
        assert "2023" in footer
        assert "claude-opus-4-6" in footer
        assert "CFA Level 2" in footer

    def test_missing_anthropic_package_raises_import_error(
        self, mscore_normal, cfq_good, red_flags
    ):
        analyst = MDAnalyst(api_key="sk-test")
        with patch(
            "advisor_brain_fsa.mda_analyst._get_anthropic",
            side_effect=ImportError("No module named 'anthropic'"),
        ):
            with pytest.raises(ImportError, match="anthropic"):
                analyst.analyze(
                    ticker="PETR4", sector="Energia", year=2023,
                    mscore_result=mscore_normal, cfq_result=cfq_good,
                    red_flags=red_flags,
                )

    def test_grade_embedded_in_header_for_risky_company(self, mscore_risky, cfq_bad):
        header = MDAnalyst._report_header(
            ticker="RISKY3", year=2023, grade="F", grade_label="Risco crítico",
            mscore=mscore_risky, cfq=cfq_bad,
        )
        assert "F" in header

    def test_alert_emoji_in_header(self, mscore_normal, cfq_good):
        header = MDAnalyst._report_header(
            ticker="X", year=2023, grade="A", grade_label="",
            mscore=mscore_normal, cfq=cfq_good,
        )
        assert any(e in header for e in ["🔴", "🟠", "🟡", "🟢", "⚪"])

    def test_default_model_is_opus(self):
        analyst = MDAnalyst(api_key="sk-test")
        assert analyst.model == "claude-opus-4-6"
