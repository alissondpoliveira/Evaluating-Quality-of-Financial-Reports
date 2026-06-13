# Advisor-Brain-FSA — Instruções para o Agente

## Regras Comportamentais (Karpathy Guidelines)

### 1. Think Before Coding
Antes de qualquer implementação, declare explicitamente:
- Suposições feitas sobre o que o código deve fazer
- Módulos de cálculo que serão afetados
- Tradeoffs entre abordagens alternativas

Se houver ambiguidade sobre uma fórmula financeira (M-Score, accruals ratio, sector scoring),
pergunte antes de implementar. Apresente interpretações alternativas quando a tarefa for ambígua.

### 2. Simplicity First
- Código mínimo que atende ao pedido — sem features especulativas
- Sem abstrações para código de uso único
- Sem tratamento de erro para cenários impossíveis
- Se 200 linhas podem ser 50 sem perda de clareza, reescreva

Não adicione parâmetros, flags, ou opções não solicitadas. Não generalize para casos hipotéticos futuros.

### 3. Surgical Changes
- Toque apenas o que a tarefa requer
- Não "melhore" código adjacente que não foi solicitado
- Não renomeie variáveis, não reorganize imports, não reformate código fora do escopo
- Remova apenas imports/variáveis que ficaram sem uso **por causa da sua própria mudança**
- Mantenha o estilo e padrão de nomenclatura existente no arquivo

### 4. Goal-Driven Execution
- Converta tarefas em critérios verificáveis antes de começar
- Para mudanças em cálculos financeiros: o critério de sucesso deve incluir testes passando
  E validação manual do resultado numérico contra um caso conhecido
- Tarefas com múltiplos passos recebem um plano explícito com verificação em cada etapa
- Pare e pergunte quando bloqueado, em vez de adivinhar

---

## Módulos Críticos de Cálculo

Estes arquivos implementam modelos quantitativos com thresholds e fórmulas específicas.
**Não refatorar, não renomear, não reorganizar sem solicitação explícita.**
Erros aqui são silenciosos e só aparecem nos resultados numéricos finais.

| Arquivo | Modelo | Risco de erro silencioso |
|---------|--------|--------------------------|
| `src/advisor_brain_fsa/beneish_mscore.py` | Beneish M-Score (1999) — 8 índices, limiar −1.78 | Alto: erro de precedência ou sinal inverte classificação |
| `src/advisor_brain_fsa/accruals.py` | CFA Level 2 Accruals Quality — accrual ratio | Alto: divisor errado muda sinal do ratio |
| `src/advisor_brain_fsa/sector_scorer.py` | Strategy Pattern por setor B3 | Médio: setor errado aplica scorer errado |
| `src/advisor_brain_fsa/cvm_accounts.py` | Mapeamento contas CVM DFP → campos Beneish | Alto: conta errada propaga erro para todos os índices |
| `src/advisor_brain_fsa/data_fetcher.py` | Download e parse CVM DFP open-data | Médio: parse incorreto de ZIP/CSV silencioso |

---

## Contexto do Projeto

**Stack:** Python 3.11, numpy, pandas, streamlit, plotly, anthropic (Claude), google-generativeai (Gemini)

**Fontes de dados:**
- CVM DFP (Portal de Dados Abertos da CVM) — demonstrações financeiras padronizadas B3
- B3 ticker universe via `ticker_map.py` (estático) e `cvm_registry.py` (dinâmico)

**Testes:** `pytest tests/` — todos os testes mockam chamadas de rede. Rodar antes de qualquer commit
que afete módulos de cálculo.

**Deploy:** Streamlit Community Cloud — `app.py` é o entry point. Regra crítica: `st.set_page_config()`
deve ser a PRIMEIRA chamada Streamlit. Todos os imports de `advisor_brain_fsa` são lazy
via `@st.cache_resource`.

**Branch de desenvolvimento:** `claude/beneish-mscore-module-bjmig` (feature) / `main` (produção)
