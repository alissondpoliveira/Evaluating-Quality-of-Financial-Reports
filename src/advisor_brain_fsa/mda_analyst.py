"""
MD&A Analyst — AI-powered Risk Thesis Generator
-------------------------------------------------
Integrates Beneish M-Score + CFA Accruals Quality results with the
Anthropic Claude API to produce a structured "Tese de Risco" report,
grounded in the CFA Level 2 chapter "Evaluating Quality of Financial Reports".

The output is a Markdown report with three mandatory sections:
  1. Resumo da Fragilidade — qualitative grade A–F with rationale
  2. Onde Está o "Batom na Cueca" — the specific distorted account
  3. Perguntas para o Diretor de RI — three technical IR questions

Usage
-----
    from advisor_brain_fsa.mda_analyst import MDAnalyst

    analyst = MDAnalyst()                           # reads ANTHROPIC_API_KEY

    # Non-streaming (returns full Markdown string)
    report = analyst.analyze(
        ticker="PETR4", sector="Energia", year=2023,
        mscore_result=mscore, cfq_result=cfq, red_flags=flags,
    )

    # Streaming (yields text chunks — for live terminal output)
    for chunk in analyst.analyze_streaming(...):
        print(chunk, end="", flush=True)

Dependencies
------------
    pip install anthropic>=0.86.0
"""

from __future__ import annotations

import os
import textwrap
from typing import Generator, Iterator, List, Optional

from .accruals import CashFlowQualityResult
from .beneish_mscore import MScoreResult

# ---------------------------------------------------------------------------
# Lazy import — only crash if the user actually calls analyze()
# ---------------------------------------------------------------------------

def _get_anthropic():
    try:
        import anthropic
        return anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for AI analysis.\n"
            "Install it with:  pip install anthropic"
        ) from exc


# ---------------------------------------------------------------------------
# Default model
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Deterministic grade (A–F) — computed before calling the API
# ---------------------------------------------------------------------------

_GRADE_TABLE = [
    # (mscore_max, accrual_max, grade, label)   — non-manipulator zone only
    (-2.50, 0.01, "A", "Qualidade exemplar — sem sinais de manipulação"),
    (-2.20, 0.03, "B", "Boa qualidade — alertas menores, monitorar"),
    (-1.78, 0.05, "C", "Qualidade questionável — divergências precisam de explicação"),
]


def compute_grade(m_score: float, accrual_ratio: float) -> tuple[str, str]:
    """
    Return (letter_grade, label) based on M-Score and accrual ratio.

    Grading logic (deterministic — does not require the AI):
      A  M-Score ≤ −2.50  AND  Accrual Ratio < 1%
      B  M-Score ≤ −2.20  AND  Accrual Ratio < 3%
      C  M-Score ≤ −1.78  AND  Accrual Ratio < 5%
      D  M-Score > −1.78  (manipulador potencial, accruals aceitáveis)
      F  M-Score > −1.78  AND  Accrual Ratio > 5%
    """
    is_manipulator = m_score > -1.78
    high_accruals  = accrual_ratio > 0.05

    # Manipulator zone — evaluated first, independently of the table
    if is_manipulator and high_accruals:
        return "F", "Múltiplos sinais de fraude contábil — risco crítico"
    if is_manipulator:
        return "D", "Alto risco — M-Score indica possível manipulação"

    # Non-manipulator zone — walk the grade table
    for mscore_max, accrual_max, grade, label in _GRADE_TABLE:
        if m_score <= mscore_max and accrual_ratio < accrual_max:
            return grade, label

    # m_score in (−2.20, −1.78] or accruals moderate
    return "C", "Qualidade questionável — divergências precisam de explicação"


# ---------------------------------------------------------------------------
# System prompt (persona + output contract)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    Você é um analista de investimentos sênior com a designação CFA Charterholder,
    especializado em análise forense de demonstrações financeiras e detecção de
    manipulação contábil. Sua metodologia é fundamentada no currículo do CFA Level 2,
    especificamente no capítulo "Evaluating Quality of Financial Reports" (Beneish, 1999;
    Sloan, 1996), e nas normas IFRS adotadas pelo CPC brasileiro.

    Você recebe dados quantitativos do modelo Beneish M-Score e da métrica CFA de
    qualidade de accruals (CashFlowQuality). Sua tarefa é redigir uma "Tese de Risco"
    que sirva de apoio a um analista sell-side ou gestor de carteira.

    REGRAS ABSOLUTAS:
    1. Responda SOMENTE em português brasileiro. Use terminologia técnica de CFA/IFRS.
    2. Seja direto e objetivo. Não hesite, não use linguagem vaga.
    3. Siga EXATAMENTE o formato especificado — não adicione nem omita seções.
    4. Cada seção deve ser precisa e acionável.
    5. As "Perguntas para o RI" devem ser de nível técnico — como faria um analista
       experiente numa call de resultados, referenciando contas específicas e índices.

    FORMATO DE SAÍDA OBRIGATÓRIO:
    ---
    ## 1. Resumo da Fragilidade

    **Nota Qualitativa: [LETRA]**

    [2–3 frases diretas explicando o que os dados revelam sobre a qualidade do lucro
    reportado, referenciando os índices Beneish mais relevantes e o nível de accruals.
    Mencione explicitamente se o lucro é "accrual-driven" ou "cash-driven".]

    ---
    ## 2. Onde Está o "Batom na Cueca"

    **Conta Suspeita: [NOME EXATO DA LINHA DO BALANÇO/DRE]**

    [1 parágrafo identificando QUAL conta contábil específica está distorcendo o
    resultado e COMO ela está sendo manipulada. Cite o índice Beneish correspondente
    com seu valor numérico. Explique o mecanismo contábil suspeito — ex: "diferimento
    de receita", "capitalização de despesas", "extensão da vida útil do ativo".]

    ---
    ## 3. Perguntas para o Diretor de RI

    1. [Pergunta técnica 1 — específica, com referência a conta/índice/norma IFRS]
    2. [Pergunta técnica 2 — sobre fluxo de caixa vs. lucro ou itens específicos do balanço]
    3. [Pergunta técnica 3 — sobre política contábil, comparação setorial ou mudança de estimativa]
    ---
""")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Assembles the structured user prompt from quantitative analysis results."""

    def build(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: MScoreResult,
        cfq_result: CashFlowQualityResult,
        red_flags: List[str],
        grade: str,
        grade_label: str,
    ) -> str:
        """Return the full user-turn prompt."""
        flags_block = "\n".join(
            f"  - {f}" for f in red_flags
        ) if red_flags else "  - Nenhum red flag acima do limiar detectado"

        return textwrap.dedent(f"""\
            # Dados para Análise — {ticker} | Setor: {sector} | Exercício: {year}

            ## Beneish M-Score
            | Índice | Valor    | Limiar Não-Manipulador (Beneish 1999) | Interpretação |
            |--------|----------|---------------------------------------|---------------|
            | DSRI   | {mscore_result.dsri:>8.4f} | 1.031 | Razão recebíveis/receita t vs t-1 |
            | GMI    | {mscore_result.gmi:>8.4f} | 1.014 | Margem bruta t-1 vs t |
            | AQI    | {mscore_result.aqi:>8.4f} | 1.039 | Qualidade de ativos t vs t-1 |
            | SGI    | {mscore_result.sgi:>8.4f} | 1.134 | Crescimento de vendas |
            | DEPI   | {mscore_result.depi:>8.4f} | 1.017 | Taxa de depreciação t-1 vs t |
            | SGAI   | {mscore_result.sgai:>8.4f} | 1.054 | Despesas G&A/receita t vs t-1 |
            | LVGI   | {mscore_result.lvgi:>8.4f} | 1.000 | Alavancagem t vs t-1 |
            | TATA   | {mscore_result.tata:>8.4f} | -0.012 | (Lucro Líq − CFO) / Ativo Total |

            **M-Score Final: {mscore_result.m_score:+.4f}**
            **Limiar de Manipulação: −1.78**
            **Classificação Beneish: {mscore_result.classification}**

            ## Qualidade de Accruals (CFA Level 2)
            | Métrica            | Valor     |
            |--------------------|-----------|
            | Accrual Ratio      | {cfq_result.accrual_ratio:+.4f} |
            | Earnings Quality   | {cfq_result.earnings_quality} |
            | Nível de Alerta    | {cfq_result.alert_level.value} |

            > Um Accrual Ratio positivo e alto indica que o lucro é predominantemente
            > "accrual-driven" — menos persistente e com maior risco de reversão futura.
            > Fórmula: (Lucro Líquido − CFO) / Ativo Total Médio.

            ## Red Flags Detectados (Top-3 por impacto no M-Score)
            {flags_block}

            ## Nota Qualitativa Pré-calculada
            **Letra: {grade}** — {grade_label}

            ---
            Com base nesses dados, redija a Tese de Risco seguindo EXATAMENTE o formato
            especificado no seu prompt de sistema. Use a Nota Qualitativa acima ({grade}).
            Identifique a conta contábil mais suspeita com base nos índices apresentados.
            Formule três perguntas técnicas que um analista CFA faria ao RI da empresa.
        """)


# ---------------------------------------------------------------------------
# MD&A Analyst
# ---------------------------------------------------------------------------

class MDAnalyst:
    """
    Calls the Anthropic API to generate a structured Risk Thesis report.

    Parameters
    ----------
    api_key : str | None
        Anthropic API key. Falls back to ANTHROPIC_API_KEY environment variable.
    model : str
        Claude model ID. Defaults to claude-opus-4-6.
    max_tokens : int
        Maximum tokens in the AI response.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8_000,
    ) -> None:
        self.api_key   = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model     = model
        self.max_tokens = max_tokens
        self._builder  = PromptBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: MScoreResult,
        cfq_result: CashFlowQualityResult,
        red_flags: List[str],
    ) -> str:
        """
        Generate the full Risk Thesis report (non-streaming).

        Returns
        -------
        str
            Complete Markdown report.
        """
        return "".join(
            self.analyze_streaming(
                ticker=ticker, sector=sector, year=year,
                mscore_result=mscore_result,
                cfq_result=cfq_result,
                red_flags=red_flags,
            )
        )

    def analyze_streaming(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: MScoreResult,
        cfq_result: CashFlowQualityResult,
        red_flags: List[str],
    ) -> Generator[str, None, None]:
        """
        Generate the Risk Thesis report, yielding text chunks as they arrive.

        Usage
        -----
        >>> for chunk in analyst.analyze_streaming(...):
        ...     print(chunk, end="", flush=True)
        """
        anthropic = _get_anthropic()

        if not self.api_key:
            raise ValueError(
                "No Anthropic API key found. Set ANTHROPIC_API_KEY or pass api_key=."
            )

        grade, grade_label = compute_grade(
            mscore_result.m_score, cfq_result.accrual_ratio
        )
        user_prompt = self._builder.build(
            ticker=ticker, sector=sector, year=year,
            mscore_result=mscore_result, cfq_result=cfq_result,
            red_flags=red_flags, grade=grade, grade_label=grade_label,
        )

        client = anthropic.Anthropic(api_key=self.api_key)

        # Header prepended before streaming content arrives
        yield self._report_header(ticker, year, grade, grade_label, mscore_result, cfq_result)

        with client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

        yield self._report_footer(ticker, year, self.model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _report_header(
        ticker: str,
        year: int,
        grade: str,
        grade_label: str,
        mscore: MScoreResult,
        cfq: CashFlowQualityResult,
    ) -> str:
        alert_emoji = {
            "Crítico": "🔴", "Alto Risco": "🟠",
            "Atenção": "🟡", "Normal": "🟢",
        }.get(cfq.alert_level.value, "⚪")

        return (
            f"# Tese de Risco — {ticker} ({year})\n\n"
            f"> **Modelo:** Beneish M-Score (1999) + CFA Level 2 Accruals Quality  \n"
            f"> **M-Score:** `{mscore.m_score:+.4f}` (limiar: −1.78) — "
            f"**{mscore.classification}**  \n"
            f"> **Nível de Alerta:** {alert_emoji} {cfq.alert_level.value}  \n"
            f"> **Nota Qualitativa:** **{grade}** — {grade_label}  \n\n"
            "---\n\n"
        )

    @staticmethod
    def _report_footer(ticker: str, year: int, model: str) -> str:
        return (
            "\n\n---\n"
            f"*Análise gerada pelo Advisor-Brain-FSA · "
            f"Ticker: {ticker} · Exercício: {year} · "
            f"Modelo IA: {model} · "
            'Framework: CFA Level 2 — "Evaluating Quality of Financial Reports"*\n'
        )
