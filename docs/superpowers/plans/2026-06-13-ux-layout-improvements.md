# UX/Layout Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Melhorar clareza visual e usabilidade das três abas do app sem tocar em nenhum módulo de cálculo.

**Architecture:** Todas as mudanças ficam em `app.py` — textos de UI, layout, HTML de empty-states. Nenhum módulo de cálculo (`beneish_mscore.py`, `accruals.py`, `sector_scorer.py`, `cvm_accounts.py`, `data_fetcher.py`) é modificado. Cada tarefa é uma edição cirúrgica em uma seção isolada do `app.py`.

**Tech Stack:** Python 3.11, Streamlit 1.45.1, Plotly, tema dark (`#000000` bg, `#FFA500` primary)

---

## Mapa de arquivos

| Arquivo | Mudança |
|---------|---------|
| `app.py:617–648` | Task 1 — Dashboard empty state |
| `app.py:1044–1077` | Task 2 — Header: label ANO + tooltip Gemini |
| `app.py:717–759` | Task 3 — Análise Individual: labels + hierarquia |
| `app.py:512–516` | Task 4 — Ranking: labels do multiselect |
| `app.py:1101–1109` | Task 5 — Footer: separação visual |

---

### Task 1: Dashboard — Estado vazio amigável

**Problema observado:** O Dashboard mostra "Nenhum dado disponível." e uma instrução de terminal (`build_market_cache.py`) que não faz sentido para o usuário final. Não há caminho claro para o que fazer.

**Files:**
- Modify: `app.py:617–648` (função `_tab_dashboard`)

- [ ] **Step 1: Screenshot "antes" para referência**

```bash
cd /home/user/Evaluating-Quality-of-Financial-Reports
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.screenshot(path='/tmp/before_task1.png')
    b.close()
print('screenshot salvo em /tmp/before_task1.png')
"
```

Expected: vê caption com `build_market_cache.py` e `st.info("Nenhum dado disponível.")`

- [ ] **Step 2: Substituir caption e empty state em `_tab_dashboard()`**

No `app.py`, função `_tab_dashboard` (linha ~617), substitua:

```python
    st.caption(
        "Empresas não-financeiras B3 · Beneish M-Score · "
        "execute `build_market_cache.py` para dados completos"
    )
```

por:

```python
    st.caption("Empresas não-financeiras B3 · Beneish M-Score · dados via Portal CVM")
```

E substitua:

```python
    if ok.empty:
        st.info("Nenhum dado disponível.")
        return
```

por:

```python
    if ok.empty:
        st.markdown(
            f'<div style="text-align:center;padding:48px 20px">'
            f'<div style="font-size:2.5rem;margin-bottom:14px">📊</div>'
            f'<div style="font-family:{_B["mono"]};color:{_B["text"]};font-size:0.92rem;'
            f'font-weight:700;margin-bottom:10px">Cache de mercado ainda não disponível</div>'
            f'<div style="font-family:{_B["mono"]};color:{_B["muted"]};font-size:0.80rem;'
            f'line-height:1.7;margin-bottom:24px">'
            f'O Dashboard exibe o ranking completo após o cache ser gerado.<br>'
            f'Para analisar uma empresa agora, use a aba '
            f'<strong style="color:{_B["orange"]}">🔍 Análise Individual</strong>.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return
```

- [ ] **Step 3: Screenshot "depois" e verificar**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.screenshot(path='/tmp/after_task1.png')
    b.close()
print('done')
"
```

Expected: ícone 📊 centralizado, texto explicativo, referência à aba "Análise Individual". Sem `build_market_cache.py`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "ux: dashboard empty state amigável — remove instrução de terminal"
```

---

### Task 2: Header — Label "ANO" + tooltip no indicador Gemini

**Problema observado:** O seletor de ano aparece sem contexto (label colapsado). O indicador "SEM API KEY" não explica o que fazer para corrigir.

**Files:**
- Modify: `app.py:1044–1077` (função `main()`)

- [ ] **Step 1: Localizar os blocos exatos**

```bash
sed -n '1044,1077p' /home/user/Evaluating-Quality-of-Financial-Reports/app.py
```

Expected: vê `with y_col:` com `label_visibility="collapsed"` e `with api_col:` com `"■ SEM API KEY"`.

- [ ] **Step 2: Adicionar label "ANO" acima do seletor de ano**

No bloco `with y_col:` (linha ~1053), substitua:

```python
    with y_col:
        year_t = st.selectbox(
            "Ano", _YEAR_OPTS, index=0,
            label_visibility="collapsed", key="year_sel",
        )
```

por:

```python
    with y_col:
        st.markdown(
            f'<div style="font-family:{_B["mono"]};font-size:0.60rem;color:{_B["muted"]};'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:-14px">ANO</div>',
            unsafe_allow_html=True,
        )
        year_t = st.selectbox(
            "Ano", _YEAR_OPTS, index=0,
            label_visibility="collapsed", key="year_sel",
        )
```

- [ ] **Step 3: Substituir API key indicator por versão com tooltip**

No bloco `with api_col:` (linha ~1058), substitua:

```python
    with api_col:
        ok_key = bool(_api_key())
        st.markdown(
            f'<span style="color:{"#00FF00" if ok_key else "#FF3E3E"};'
            f'font-family:{_B["mono"]};font-size:0.72rem">'
            f'{"■ GEMINI OK" if ok_key else "■ SEM API KEY"}</span>',
            unsafe_allow_html=True,
        )
```

por:

```python
    with api_col:
        ok_key = bool(_api_key())
        label  = "■ Gemini: ativo" if ok_key else "■ Gemini: inativo"
        color  = "#00FF00" if ok_key else "#FF3E3E"
        tip    = "" if ok_key else ' title="Configure GOOGLE_API_KEY nos Secrets do Streamlit"'
        st.markdown(
            f'<span{tip} style="color:{color};font-family:{_B["mono"]};font-size:0.72rem;'
            f'cursor:{"default" if ok_key else "help"}">{label}</span>',
            unsafe_allow_html=True,
        )
```

- [ ] **Step 4: Screenshot e verificar header**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.screenshot(path='/tmp/after_task2.png')
    b.close()
print('done')
"
```

Expected: label "ANO" visível acima do seletor, indicador mostra "■ Gemini: inativo" (sem API key) ou "■ Gemini: ativo".

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "ux: header — label ANO no seletor de ano, tooltip no indicador Gemini"
```

---

### Task 3: Análise Individual — Labels visíveis + hierarquia de inputs

**Problema observado:** O dropdown de ticker aparece sem label visível. "Carregar todas as empresas B3" (ação secundária) tem o mesmo peso visual que "Calcular" (ação primária).

**Files:**
- Modify: `app.py:717–759` (função `_tab_analise()`)

- [ ] **Step 1: Screenshot "antes" da aba**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.click('text=Análise Individual'); pg.wait_for_timeout(2000)
    pg.screenshot(path='/tmp/before_task3.png')
    b.close()
print('done')
"
```

Expected: dropdown sem label, "Carregar todas as empresas B3" como botão largo.

- [ ] **Step 2: Mostrar labels nos inputs e rebaixar botão secundário**

No bloco de colunas em `_tab_analise()` (linha ~735), substitua:

```python
    col_dd, col_cnpj, col_btn = st.columns([3, 2, 1])
    with col_dd:
        selected_label = st.selectbox(
            "Ticker", [""] + all_labels,
            index=default_idx,
            label_visibility="collapsed",
            placeholder="Selecione um ticker B3...",
            key="analise_dd",
        )
    with col_cnpj:
        cnpj_input = st.text_input(
            "CNPJ", value="",
            placeholder="ou CNPJ / nome livre...",
            label_visibility="collapsed",
            key="analise_cnpj",
        )
    with col_btn:
        calc_clicked = st.button("Calcular", use_container_width=True, key="analise_btn")

    # Lazy-load full CVM list on demand
    if not st.session_state.get("use_full_opts", False):
        if st.button("+ Carregar todas as empresas B3", key="load_full_opts"):
            st.session_state["use_full_opts"] = True
            st.rerun()
```

por:

```python
    col_dd, col_cnpj, col_btn = st.columns([3, 2, 1])
    with col_dd:
        selected_label = st.selectbox(
            "Ticker B3", [""] + all_labels,
            index=default_idx,
            placeholder="Selecione um ticker B3...",
            key="analise_dd",
        )
    with col_cnpj:
        cnpj_input = st.text_input(
            "CNPJ / busca livre",
            value="",
            placeholder="ex: 33.000.167/0001-01",
            key="analise_cnpj",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        calc_clicked = st.button("Calcular", use_container_width=True, key="analise_btn")

    # Lazy-load full CVM list on demand — ação secundária, menor destaque
    if not st.session_state.get("use_full_opts", False):
        _, col_load, _ = st.columns([3, 2, 3])
        with col_load:
            if st.button("⬇ Ampliar lista (universo CVM completo)", key="load_full_opts"):
                st.session_state["use_full_opts"] = True
                st.rerun()
```

- [ ] **Step 3: Screenshot "depois" da aba**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.click('text=Análise Individual'); pg.wait_for_timeout(2000)
    pg.screenshot(path='/tmp/after_task3.png')
    b.close()
print('done')
"
```

Expected: labels "Ticker B3" e "CNPJ / busca livre" visíveis acima dos inputs. Botão "Ampliar lista" menor e centralizado.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "ux: analise individual — labels visíveis, botão carregar lista rebaixado"
```

---

### Task 4: Ranking — Labels do multiselect sem truncamento

**Problema observado:** Tags selecionadas mostram "MGLU3 — MAGA...", "RENT3 — LOCALI..." — texto cortado pelo componente. O ticker é o identificador principal no contexto de ranking.

**Files:**
- Modify: `app.py:512–516` (função `_ranking_opts()`)

- [ ] **Step 1: Localizar a função**

```bash
sed -n '512,517p' /home/user/Evaluating-Quality-of-Financial-Reports/app.py
```

Expected:
```python
@st.cache_resource(show_spinner=False)
def _ranking_opts() -> tuple:
    TICKER_TO_KEYWORD, _, _ = _ticker_map()
    labels = [f"{t} — {TICKER_TO_KEYWORD.get(t, t)}" for t in sorted(TICKER_TO_KEYWORD.keys())]
    return labels, dict(zip(labels, sorted(TICKER_TO_KEYWORD.keys())))
```

- [ ] **Step 2: Encurtar labels para ticker + nome truncado (máx 14 chars)**

Substitua:

```python
@st.cache_resource(show_spinner=False)
def _ranking_opts() -> tuple:
    TICKER_TO_KEYWORD, _, _ = _ticker_map()
    labels = [f"{t} — {TICKER_TO_KEYWORD.get(t, t)}" for t in sorted(TICKER_TO_KEYWORD.keys())]
    return labels, dict(zip(labels, sorted(TICKER_TO_KEYWORD.keys())))
```

por:

```python
@st.cache_resource(show_spinner=False)
def _ranking_opts() -> tuple:
    TICKER_TO_KEYWORD, _, _ = _ticker_map()
    tickers = sorted(TICKER_TO_KEYWORD.keys())
    labels  = [
        f"{t} — {TICKER_TO_KEYWORD.get(t, t)[:14]}"
        for t in tickers
    ]
    return labels, dict(zip(labels, tickers))
```

Atenção: o `default_lbl` em `_tab_ranking()` usa `lbl.startswith(t)`, que continua funcionando porque "PETR4 — PETROBRA".startswith("PETR4") é True.

- [ ] **Step 3: Screenshot da aba Ranking para verificar tags**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.click('text=Ranking'); pg.wait_for_timeout(2000)
    pg.screenshot(path='/tmp/after_task4.png')
    b.close()
print('done')
"
```

Expected: tags mostram "PETR4 — PETROBRA" (sem "..."), "MGLU3 — MAGAZINE" — truncado por nós de forma previsível, não pela UI.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "ux: ranking — truncar nomes de empresa em 14 chars nas tags do multiselect"
```

---

### Task 5: Footer — Separação visual clara

**Problema observado:** O disclaimer legal tem a mesma fonte/tamanho que o conteúdo principal, sem distinção visual clara.

**Files:**
- Modify: `app.py:1101–1109` (função `main()`)

- [ ] **Step 1: Localizar o footer**

```bash
sed -n '1100,1110p' /home/user/Evaluating-Quality-of-Financial-Reports/app.py
```

Expected:
```python
    st.markdown(
        f'<hr style="border-top:1px solid {_B["border"]};margin:32px 0 8px 0">',
        unsafe_allow_html=True,
    )
    st.caption(
        "CFA Institute: indicadores quantitativos são sinais de alerta, não prova de fraude. "
        "IA: narrativas geradas por Gemini 2.5 Flash — uso educacional. "
        "Não é recomendação de investimento."
    )
```

- [ ] **Step 2: Substituir `st.caption` por markdown centralizado e menor**

Substitua:

```python
    st.caption(
        "CFA Institute: indicadores quantitativos são sinais de alerta, não prova de fraude. "
        "IA: narrativas geradas por Gemini 2.5 Flash — uso educacional. "
        "Não é recomendação de investimento."
    )
```

por:

```python
    st.markdown(
        f'<div style="text-align:center;font-family:{_B["mono"]};font-size:0.65rem;'
        f'color:{_B["muted"]};line-height:1.6;padding:4px 0 12px">'
        f'CFA Institute: indicadores quantitativos são sinais de alerta, não prova de fraude. '
        f'IA: narrativas geradas por Gemini 2.5 Flash — uso educacional. '
        f'Não é recomendação de investimento.'
        f'</div>',
        unsafe_allow_html=True,
    )
```

- [ ] **Step 3: Screenshot do rodapé**

```bash
pkill -f "streamlit run" 2>/dev/null; sleep 1
streamlit run app.py --server.port 8502 --server.headless true &
sleep 9
python3 -c "
import os; os.environ['PLAYWRIGHT_BROWSERS_PATH'] = '/opt/pw-browsers'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
    pg = b.new_page(viewport={'width': 1280, 'height': 800})
    pg.goto('http://localhost:8502'); pg.wait_for_timeout(8000)
    pg.screenshot(path='/tmp/after_task5.png')
    b.close()
print('done')
"
```

Expected: disclaimer centralizado, menor e claramente separado do conteúdo de negócio acima.

- [ ] **Step 4: Commit final + push**

```bash
git add app.py
git commit -m "ux: footer — disclaimer centralizado e menor para separação visual clara"
git push -u origin claude/beneish-mscore-module-bjmig
```

---

## Self-Review

**1. Spec coverage:**
- ✅ Dashboard vazio sem orientação → Task 1
- ✅ "SEM API KEY" sem contexto → Task 2
- ✅ Label ANO ausente no seletor → Task 2
- ✅ Dropdown sem label, botão secundário proeminente → Task 3
- ✅ Tags de ranking truncadas → Task 4
- ✅ Footer sem separação visual → Task 5

**2. Placeholder scan:** Nenhum "TBD", "TODO" ou "handle edge cases" encontrado.

**3. Consistência de nomes:**
- `_ranking_opts()` retorna `(labels, dict)` — mesma assinatura, mesma forma de uso em `_tab_ranking()`. ✓
- `default_lbl` usa `lbl.startswith(t)` — compatível com labels truncados. ✓
- `_tab_analise()` usa `lbl_to_val.get(selected_label, selected_label)` — label com nome visível continua funcionando. ✓
