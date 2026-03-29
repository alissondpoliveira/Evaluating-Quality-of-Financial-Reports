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

import hashlib
import os
import textwrap
from pathlib import Path
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


def compute_grade_financial(risk_score: float, alert_level_value: str) -> tuple[str, str]:
    """
    Return (letter_grade, label) for banking / insurance scorers.
    Uses the normalised 0-10 risk_score from SectorRiskResult.
    """
    if risk_score <= 2.5:
        return "A", "Qualidade exemplar — indicadores financeiros saudáveis"
    if risk_score <= 4.5:
        return "B", "Boa qualidade — monitorar indicadores secundários"
    if risk_score <= 6.5:
        return "C", "Qualidade questionável — desvios precisam de explicação"
    if risk_score <= 8.0:
        return "D", "Alto risco — múltiplos indicadores comprometidos"
    return "F", "Risco crítico — indicadores em zona de alerta máximo"


# ---------------------------------------------------------------------------
# System prompt (persona + output contract)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    Você é um AUDITOR FORENSE DE DEMONSTRAÇÕES FINANCEIRAS com a designação CFA Charterholder
    e experiência em litígios contábeis (forensic accounting). Sua função é distinta da de um
    analista de investimentos convencional: você não opina sobre preço-alvo ou valuation —
    você investiga se os números reportados refletem a realidade econômica da empresa ou se
    foram manipulados intencionalmente (earnings manipulation).

    ENQUADRAMENTO METODOLÓGICO (CFA Level 2):
    - Beneish M-Score (1999): modelo probit de 8 fatores para detectar gerenciamento de
      resultados. Limiar: M > −1.78 ⟹ "zona de manipulador provável".
    - Accrual Ratio — Sloan (1996): lucros "accrual-driven" revertem; lucros "cash-driven"
      são persistentes. Accruals elevados sinalizam baixa qualidade de earnings.
    - IFRS 15 / CPC 47 (Receitas): critérios de reconhecimento de receita — performance
      obligations, variable consideration, bill-and-hold, channel stuffing.
    - IAS 36 / CPC 01 (Impairment): teste de recuperabilidade e subjetividade nas premissas.
    - IAS 16 / CPC 27 (Ativo Fixo): vida útil estimada e impacto na depreciação (DEPI).
    - Cookie jar reserves: provisionamento excessivo em anos bons para liberação futura.

    SUA TAREFA CENTRAL — DISTINGUIR:
    A. MANIPULATION HYPOTHESIS: os desvios indicam alteração intencional de políticas
       contábeis, reconhecimento agressivo de receitas, ou omissão de obrigações.
    B. OPERATIONAL VOLATILITY HYPOTHESIS: os desvios têm explicação operacional legítima
       (expansão de capacidade, mudança de mix de produtos, ciclo setorial, M&A).

    VOCÊ DEVE SEMPRE tomar uma posição clara sobre qual hipótese os dados favorecem,
    citando os índices numéricos como evidência. Não seja neutro onde os dados falam.

    REGRAS ABSOLUTAS:
    1. Responda SOMENTE em português brasileiro. Use terminologia técnica de CFA/IFRS/IFAC.
    2. Seja direto e incisivo — escreva como perito em tribunal, não como relações públicas.
    3. Siga EXATAMENTE o formato especificado abaixo — não adicione nem omita seções.
    4. Cite SEMPRE o valor numérico do índice ao referenciá-lo (ex: DSRI = 1.24).
    5. As perguntas para o RI devem ser impossíveis de responder com evasivas genéricas.

    FORMATO DE SAÍDA OBRIGATÓRIO:
    ---
    ## 1. Veredito Forense

    **Nota de Qualidade: [LETRA]** · **Hipótese dominante: [MANIPULAÇÃO | VOLATILIDADE OPERACIONAL | INCONCLUSIVO]**

    [2–3 frases diretas com o veredito. Identifique os 1–2 índices mais críticos com seus
    valores numéricos. Declare se o lucro é "accrual-driven" ou "cash-driven" e o que isso
    implica para a persistência dos resultados. Seja assertivo.]

    ---
    ## 2. Cadeia de Evidências

    **Índice-gatilho: [NOME DO ÍNDICE] = [VALOR] (limiar: [LIMIAR])**

    [1 parágrafo descrevendo o MECANISMO contábil suspeito ou a explicação operacional
    mais provável. Cite a norma IFRS/CPC relevante. Para cada índice anômalo, classifique
    como (a) evidência de manipulação, (b) ruído operacional, ou (c) ambíguo, com justificativa
    de 1 frase. Se múltiplos índices convergem, isso reforça a hipótese de manipulação.]

    ---
    ## 3. Perguntas Forenses para o Diretor de RI

    1. [Pergunta sobre o índice mais crítico — cite a conta exata e o desvio numérico. Ex:
       "Contas a receber cresceram X% acima da receita (DSRI = Y). Qual a política de
       reconhecimento de receita para contratos de longo prazo sob IFRS 15 art. 35(c)?"]
    2. [Pergunta sobre a divergência caixa vs. accrual — cite o Accrual Ratio e peça
       reconciliação entre EBITDA reportado e FCO com detalhamento por linha.]
    3. [Pergunta sobre política contábil específica alterada no período — vida útil,
       provisões, capitalização de P&D, ou critério de impairment sob IAS 36.]
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

    def build_financial(
        self,
        ticker: str,
        sector: str,
        year: int,
        scorer_type: str,
        metrics: dict,
        risk_score: float,
        classification: str,
        red_flags: list,
        grade: str,
        grade_label: str,
    ) -> str:
        """Return the user-turn prompt for banking or insurance scorers."""
        import textwrap as _tw

        flags_block = "\n".join(
            f"  - {f}" for f in red_flags
        ) if red_flags else "  - Nenhum red flag acima do limiar detectado"

        def _fmt(v):
            if v != v:  # NaN
                return "N/D"
            return f"{v:.4f}"

        if scorer_type == "banking":
            metrics_block = _tw.dedent(f"""                ## Indicadores Bancários
                | Métrica             | Valor              | Referência Prudencial |
                |---------------------|--------------------|-----------------------|
                | ROA                 | {_fmt(metrics.get('roa', float('nan')))} | > 1,0% (BACEN bom) |
                | Cost-to-Income      | {_fmt(metrics.get('cost_income', float('nan')))} | < 50% (eficiente) |
                | CFO Quality         | {_fmt(metrics.get('cfo_quality', float('nan')))} | > 0 (caixa saudável) |
                | Crescimento Receita | {_fmt(metrics.get('rev_growth', float('nan')))} | referência setorial |
                | Spread Bancário     | {_fmt(metrics.get('spread', float('nan')))} | referência setorial |
                | Alavancagem         | {_fmt(metrics.get('leverage', float('nan')))} | < 12x (Basileia III) |

                **Score de Risco: {risk_score:.2f}/10**
                **Classificação: {classification}**

                > Analise ROA, eficiência operacional e qualidade do CFO sob a ótica do
                > Acordo de Basileia III (capital mínimo, índice de cobertura de liquidez).
                > Verifique se o crescimento de receita é sustentado por spread saudável ou
                > por alavancagem excessiva (risco de crédito sistêmico).""")
        else:  # insurance
            metrics_block = _tw.dedent(f"""                ## Indicadores de Seguros
                | Métrica              | Valor              | Referência SUSEP  |
                |----------------------|--------------------|-------------------|
                | Sinistralidade       | {_fmt(metrics.get('loss_ratio', float('nan')))} | < 70% (aceitável) |
                | Índice de Despesas   | {_fmt(metrics.get('expense_ratio', float('nan')))} | < 30% (eficiente) |
                | Índice Combinado     | {_fmt(metrics.get('combined_ratio', float('nan')))} | < 100% (lucrativo) |
                | Crescimento Receita  | {_fmt(metrics.get('rev_growth', float('nan')))} | referência setorial |
                | ROA                  | {_fmt(metrics.get('roa', float('nan')))} | > 2% (seguros) |
                | Qualidade CFO        | {_fmt(metrics.get('cfo_quality', float('nan')))} | > 0 (caixa positivo)|

                **Score de Risco: {risk_score:.2f}/10**
                **Classificação: {classification}**

                > Analise sinistralidade, índice combinado e qualidade do CFO sob a ótica
                > das normas SUSEP (Circular 667/2022) e IFRS 17 (Insurance Contracts).
                > Verifique se provisões técnicas são suficientes (adequacy test) e se o
                > índice combinado reflete gerenciamento agressivo de sinistros.""")

        return _tw.dedent(f"""            # Dados para Análise — {ticker} | Setor: {sector} | Exercício: {year}

            **Tipo de Scorer:** {scorer_type.upper()} (especializado — não aplica Beneish M-Score)

            {metrics_block}

            ## Red Flags / Alertas Detectados
            {flags_block}

            ## Nota Qualitativa Pré-calculada
            **Letra: {grade}** — {grade_label}

            ---
            Com base nesses dados, redija a Tese de Risco seguindo EXATAMENTE o formato
            especificado no seu prompt de sistema. Use a Nota Qualitativa acima ({grade}).
            Adapte as referências normativas ao setor ({sector}): para bancos, use Basileia III
            e BACEN; para seguros, use SUSEP e IFRS 17. Substitua as perguntas sobre DSRI/GMI
            por perguntas específicas sobre os indicadores acima que mais se desviam do padrão.
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


# ---------------------------------------------------------------------------
# Gemini Analyst (Google Generative AI backend)
# ---------------------------------------------------------------------------

def _get_genai():
    try:
        import google.generativeai as genai
        return genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-generativeai' package is required for Gemini analysis.\n"
            "Install it with:  pip install google-generativeai"
        ) from exc


class GeminiAnalyst:
    """
    Calls Google Gemini API to generate a structured Risk Thesis report.
    Drop-in replacement for MDAnalyst — same public interface.

    Parameters
    ----------
    api_key : str | None
        Google API key. Falls back to GOOGLE_API_KEY environment variable.
    model : str
        Gemini model ID. Defaults to gemini-2.5-flash.
    cache_dir : Path | str | None
        Directory for persisted AI report cache.
        Defaults to ``data/ai_reports/`` relative to the project root.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    _DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "ai_reports"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.api_key   = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model     = model
        self.cache_dir = Path(cache_dir) if cache_dir else self._DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._builder  = PromptBuilder()

    # ------------------------------------------------------------------
    # Tarefa 1 — cache helpers
    # ------------------------------------------------------------------

    def _sig_hash(
        self,
        ticker: str,
        year: int,
        mscore_result: Optional[MScoreResult],
        cfq_result: Optional[CashFlowQualityResult],
        sector_risk=None,
    ) -> str:
        """
        Short hash that changes when financial data is restated (CVM VERSAO).
        For beneish: uses M-Score + accrual_ratio.
        For banking/insurance: uses risk_score + scorer_type + classification.
        """
        if mscore_result is not None and cfq_result is not None:
            sig = (
                f"{ticker}|{year}|{mscore_result.m_score:.6f}"
                f"|{cfq_result.accrual_ratio:.6f}|{cfq_result.earnings_quality}"
            )
        elif sector_risk is not None:
            sig = (
                f"{ticker}|{year}|{sector_risk.risk_score:.6f}"
                f"|{sector_risk.scorer_type}|{sector_risk.classification}"
            )
        else:
            sig = f"{ticker}|{year}"
        return hashlib.md5(sig.encode()).hexdigest()[:10]

    def _cache_path(
        self,
        ticker: str,
        year: int,
        mscore_result: Optional[MScoreResult],
        cfq_result: Optional[CashFlowQualityResult],
        sector_risk=None,
    ) -> Path:
        h = self._sig_hash(ticker, year, mscore_result, cfq_result, sector_risk)
        return self.cache_dir / f"{ticker}_{year}_{h}.md"

    def _load_cache(
        self,
        ticker: str,
        year: int,
        mscore_result: Optional[MScoreResult],
        cfq_result: Optional[CashFlowQualityResult],
        sector_risk=None,
    ) -> Optional[str]:
        """Return cached report text, or None on miss."""
        p = self._cache_path(ticker, year, mscore_result, cfq_result, sector_risk)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def _save_cache(
        self,
        ticker: str,
        year: int,
        mscore_result: Optional[MScoreResult],
        cfq_result: Optional[CashFlowQualityResult],
        content: str,
        sector_risk=None,
    ) -> None:
        p = self._cache_path(ticker, year, mscore_result, cfq_result, sector_risk)
        p.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: Optional[MScoreResult] = None,
        cfq_result: Optional[CashFlowQualityResult] = None,
        red_flags: Optional[List[str]] = None,
        sector_risk=None,
    ) -> str:
        """
        Generate the full Risk Thesis report (non-streaming).
        Returns cached result if data signature matches; calls API otherwise.
        Supports beneish (mscore_result + cfq_result) and banking/insurance (sector_risk).
        """
        red_flags = red_flags or []
        cached = self._load_cache(ticker, year, mscore_result, cfq_result, sector_risk)
        if cached is not None:
            return cached
        return "".join(
            self._stream_api(
                ticker=ticker, sector=sector, year=year,
                mscore_result=mscore_result,
                cfq_result=cfq_result,
                red_flags=red_flags,
                sector_risk=sector_risk,
            )
        )

    def analyze_streaming(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: Optional[MScoreResult] = None,
        cfq_result: Optional[CashFlowQualityResult] = None,
        red_flags: Optional[List[str]] = None,
        sector_risk=None,
    ) -> Generator[str, None, None]:
        """
        Yield text chunks.  On cache hit: yields the full cached text as one
        chunk (instant).  On cache miss: streams from Gemini and persists.
        Supports beneish (mscore_result + cfq_result) and banking/insurance (sector_risk).
        """
        red_flags = red_flags or []
        cached = self._load_cache(ticker, year, mscore_result, cfq_result, sector_risk)
        if cached is not None:
            yield cached
            return

        parts: List[str] = []
        for chunk in self._stream_api(
            ticker=ticker, sector=sector, year=year,
            mscore_result=mscore_result,
            cfq_result=cfq_result,
            red_flags=red_flags,
            sector_risk=sector_risk,
        ):
            parts.append(chunk)
            yield chunk

        self._save_cache(ticker, year, mscore_result, cfq_result, "".join(parts), sector_risk)

    @staticmethod
    def _report_header_financial(
        ticker: str,
        year: int,
        grade: str,
        grade_label: str,
        scorer_type: str,
        risk_score: float,
        classification: str,
        alert_level_value: str,
    ) -> str:
        """Report header for banking / insurance scorers."""
        alert_emoji = {
            "Critico": "🔴", "Crítico": "🔴",
            "Alto Risco": "🟠",
            "Atencao": "🟡", "Atenção": "🟡",
            "Normal": "🟢",
        }.get(alert_level_value, "⚪")
        scorer_label = {
            "banking": "Modelo Bancário (ROA + Cost-to-Income + CFO Quality)",
            "insurance": "Índice Combinado SUSEP + IFRS 17",
        }.get(scorer_type, scorer_type.upper())
        return (
            f"# Tese de Risco — {ticker} ({year})\n\n"
            f"> **Scorer:** {scorer_label}  \n"
            f"> **Score de Risco:** `{risk_score:.2f}/10` — **{classification}**  \n"
            f"> **Nível de Alerta:** {alert_emoji} {alert_level_value}  \n"
            f"> **Nota Qualitativa:** **{grade}** — {grade_label}  \n\n"
            "---\n\n"
        )

    def _stream_api(
        self,
        ticker: str,
        sector: str,
        year: int,
        mscore_result: Optional[MScoreResult] = None,
        cfq_result: Optional[CashFlowQualityResult] = None,
        red_flags: Optional[List[str]] = None,
        sector_risk=None,
    ) -> Generator[str, None, None]:
        """Internal: always calls the Gemini API (no cache check).
        Routes to beneish or financial prompt based on available data."""
        genai = _get_genai()

        if not self.api_key:
            raise ValueError(
                "No Google API key found. Set GOOGLE_API_KEY or pass api_key=."
            )

        red_flags = red_flags or []
        is_beneish = (mscore_result is not None and cfq_result is not None)

        if is_beneish:
            grade, grade_label = compute_grade(
                mscore_result.m_score, cfq_result.accrual_ratio
            )
            user_prompt = self._builder.build(
                ticker=ticker, sector=sector, year=year,
                mscore_result=mscore_result, cfq_result=cfq_result,
                red_flags=red_flags, grade=grade, grade_label=grade_label,
            )
            header = MDAnalyst._report_header(
                ticker, year, grade, grade_label, mscore_result, cfq_result
            )
        elif sector_risk is not None:
            grade, grade_label = compute_grade_financial(
                sector_risk.risk_score, sector_risk.alert_level.value
            )
            user_prompt = self._builder.build_financial(
                ticker=ticker, sector=sector, year=year,
                scorer_type=sector_risk.scorer_type,
                metrics=sector_risk.metrics,
                risk_score=sector_risk.risk_score,
                classification=sector_risk.classification,
                red_flags=red_flags,
                grade=grade, grade_label=grade_label,
            )
            header = GeminiAnalyst._report_header_financial(
                ticker, year, grade, grade_label,
                sector_risk.scorer_type,
                sector_risk.risk_score,
                sector_risk.classification,
                sector_risk.alert_level.value,
            )
        else:
            raise ValueError(
                "Provide either mscore_result+cfq_result (beneish) or sector_risk (banking/insurance)."
            )

        genai.configure(api_key=self.api_key)
        genai_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=_SYSTEM_PROMPT,
        )

        yield header

        response = genai_model.generate_content(user_prompt, stream=True)
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

        yield MDAnalyst._report_footer(ticker, year, self.model)
