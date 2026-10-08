#!/usr/bin/env python3
"""
site/gerar.py — página estática do M-Score de Beneish (beneish.alissonprata.io)
================================================================================
Gera site/public/index.html e site/public/vercel.json. Os números vêm de
site/public/dados.json, produzido por build_static_site.py; a página só lê e mostra.
Mesma identidade visual do alissonprata.io (Inter + Newsreader, tokens claro/escuro).
"""

from pathlib import Path
import json

PUBLICO = Path(__file__).resolve().parent / "public"

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
       "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
       "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; img-src 'self' data:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'self'; form-action 'none'; object-src 'none'; upgrade-insecure-requests")

CSS = """
:root{color-scheme:light;--bg:#f7f6f3;--surface:#fcfcfb;--surface-2:#f0efec;--ink:#0b0b0b;--ink-2:#52514e;--ink-3:#6b6a65;--line:#e2e0da;--line-2:#d3d1ca;--accent:#1c5cab;
 --neg-2:#1c5cab;--neg-1:#86b6ef;--neu:#d9d7d1;--pos-1:#f0a3a2;--pos-2:#c7302f;--aviso:#b26a00}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--bg:#121211;--surface:#1a1a19;--surface-2:#232321;--ink:#fff;--ink-2:#c3c2b7;--ink-3:#9b9a94;--line:#2c2c2a;--line-2:#3a3a37;--accent:#6da7ec;
 --neg-2:#3987e5;--neg-1:#1c4f8f;--neu:#4a4a46;--pos-1:#8f3534;--pos-2:#e66767;--aviso:#e0a040}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--surface-2:#232321;--ink:#fff;--ink-2:#c3c2b7;--ink-3:#9b9a94;--line:#2c2c2a;--line-2:#3a3a37;--accent:#6da7ec;
 --neg-2:#3987e5;--neg-1:#1c4f8f;--neu:#4a4a46;--pos-1:#8f3534;--pos-2:#e66767;--aviso:#e0a040}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
h1,h2,.serif{font-family:Newsreader,Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:2.5rem;line-height:1.1;margin:.25rem 0 .7rem}h2{font-size:1.5rem;margin:0 0 .4rem}h3{font-size:.9375rem;margin:1.2rem 0 .4rem}
p{margin:.4rem 0 .9rem}a{color:var(--accent)}
.kicker{font-size:.75rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
.muted{color:var(--ink-2)}.fraco{color:var(--ink-3)}.pequeno{font-size:.8125rem}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.barra{position:sticky;top:0;z-index:30;background:color-mix(in srgb,var(--bg) 90%,transparent);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.barra-in{max-width:1180px;margin:0 auto;padding:10px 16px;display:flex;align-items:center;gap:18px}
.marca{display:flex;flex-direction:column;text-decoration:none;color:var(--ink);line-height:1.15}
.marca b{font-family:Newsreader,Georgia,serif;font-size:1.25rem}.marca span{font-size:.75rem;color:var(--ink-3)}
.nav{display:flex;gap:16px;margin-left:auto;align-items:center;font-size:.875rem}
.nav a{color:var(--ink-2);text-decoration:none}.nav a:hover{color:var(--ink)}
.icone{width:34px;height:34px;display:inline-grid;place-items:center;border-radius:8px;border:1px solid var(--line-2);background:var(--surface);padding:0;color:var(--ink-2);cursor:pointer;font:inherit}
main{max-width:1180px;margin:0 auto;padding:26px 16px 64px}
.educ{display:flex;gap:10px;align-items:flex-start;background:var(--surface);border:1px solid var(--line-2);border-left:3px solid var(--aviso);border-radius:8px;padding:10px 14px;font-size:.875rem;color:var(--ink-2);margin:4px 0 22px}
.dek{color:var(--ink-2);font-size:1.05rem;max-width:760px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:20px 0}
.tile{padding:14px 16px;box-shadow:inset -1px -1px 0 var(--line)}.tile span{display:block;font-size:.8125rem;color:var(--ink-2)}
.tile b{display:block;font-size:1.55rem;font-weight:600;font-variant-numeric:tabular-nums}.tile small{color:var(--ink-3);font-size:.75rem}
.painel{background:var(--surface);border:1px solid var(--line);border-radius:10px}
section{margin-top:34px}
#hist{width:100%;display:block}
.filtros{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 12px;border-bottom:1px solid var(--line)}
.filtros input[type=search]{flex:1;min-width:200px;max-width:340px}.espaco{flex:1}
select,input,button{font:inherit;font-size:.875rem;color:var(--ink);background:var(--surface);border:1px solid var(--line-2);border-radius:6px;padding:.35rem .55rem}
button{cursor:pointer;background:var(--surface-2)}
.chk{display:inline-flex;gap:6px;align-items:center;font-size:.8125rem;color:var(--ink-2);cursor:pointer}
.tabela{overflow:auto;max-height:calc(100vh - 200px)}
table{border-collapse:collapse;width:100%;font-size:.8125rem;font-variant-numeric:tabular-nums}
th,td{padding:.45rem .6rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
td.t,th.t{text-align:left}
th{font-weight:600;color:var(--ink-2);background:var(--surface-2);position:sticky;top:0;cursor:pointer;user-select:none;z-index:1}
tbody tr[data-cnpj]{cursor:pointer}tbody tr:hover{background:var(--surface-2)}
.nome{max-width:300px;overflow:hidden;text-overflow:ellipsis;display:inline-block;vertical-align:bottom}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:.75rem}.badge i{width:9px;height:9px;border-radius:50%;display:inline-block}
.av{color:var(--aviso);font-weight:700}
.vazio{padding:16px 12px;margin:0;color:var(--ink-2)}
.metodo li{margin-bottom:.5rem;color:var(--ink-2)}.metodo b{color:var(--ink)}
footer{margin-top:48px;font-size:.8125rem;color:var(--ink-3);border-top:1px solid var(--line);padding-top:16px}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--line-2);border-radius:8px;padding:8px 10px;font-size:.8125rem;display:none;z-index:60;box-shadow:0 6px 24px rgba(0,0,0,.14)}
/* ficha lateral */
.fundo{position:fixed;inset:0;background:rgba(0,0,0,.32);z-index:40;opacity:0;pointer-events:none;transition:opacity .2s}
body.aberta .fundo{opacity:1;pointer-events:auto}
.gaveta{position:fixed;top:0;right:0;bottom:0;width:min(560px,100vw);background:var(--surface);border-left:1px solid var(--line-2);z-index:41;display:flex;flex-direction:column;transform:translateX(102%);transition:transform .25s ease;box-shadow:-14px 0 36px rgba(0,0,0,.14)}
.gaveta[hidden]{display:none}body.aberta .gaveta{transform:none}
.gav-topo{display:flex;gap:10px;align-items:flex-start;padding:16px 18px 10px;border-bottom:1px solid var(--line)}
.gav-topo h2{font-size:1.4rem;line-height:1.2;margin:.1rem 0}.gav-corpo{overflow-y:auto;padding:12px 18px 40px;flex:1}
.grade{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.grade div{background:var(--surface);padding:10px 12px}.grade small{display:block;color:var(--ink-2);font-size:.75rem}.grade b{font-size:1.15rem;font-variant-numeric:tabular-nums}
.grade em{display:block;font-style:normal;font-size:.75rem;color:var(--ink-3)}
.avisos{border:1px solid var(--aviso);border-radius:8px;padding:10px 12px;margin:12px 0;font-size:.8125rem;color:var(--ink-2)}
.avisos b{color:var(--aviso)}
#contrib{width:100%;display:block}
.gav-corpo td.t{white-space:normal}.gav-corpo th,.gav-corpo td{padding:.4rem .5rem}
/* botão i do glossário */
.info{display:inline-grid;place-items:center;position:relative;width:15px;height:15px;margin:0 0 0 5px;padding:0;border:1px solid var(--ink-3);border-radius:50%;background:none;color:var(--ink-3);font:italic 600 10px/1 Georgia,serif;text-transform:none;vertical-align:1px;cursor:help}
.info::after{content:'';position:absolute;inset:-7px}.info:hover,.info:focus-visible{color:var(--accent);border-color:var(--accent)}
.info-pop{position:fixed;z-index:70;display:none;max-height:calc(100vh - 16px);overflow:auto;background:var(--surface);color:var(--ink);border:1px solid var(--line-2);border-radius:10px;padding:10px 12px;font-size:.8125rem;line-height:1.5;box-shadow:0 10px 30px rgba(0,0,0,.18)}
.info-pop b{display:block;font-size:.875rem}.info-pop p{margin:.35rem 0 0}
.eqs{background:var(--surface-2);border-radius:8px;padding:4px 10px;margin:.5rem 0;overflow-x:auto}.eqs .katex-display{margin:.45rem 0}
.passos{margin:.4rem 0;padding-left:1.2rem}.passos li{margin:.15rem 0}
.formula{display:block;font-family:ui-monospace,Consolas,monospace;font-size:.75rem;background:var(--surface-2);border-radius:6px;padding:6px 8px;margin:.4rem 0}
.glossario{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:4px 28px;margin:0}
.glossario div{padding:.6rem 0;border-bottom:1px solid var(--line)}.glossario dt{font-weight:600;font-size:.875rem}.glossario dd{margin:.15rem 0 0;font-size:.8125rem;color:var(--ink-2)}
@media (max-width:760px){h1{font-size:2rem}.barra-in{flex-wrap:wrap}.nav{margin-left:0;width:100%}.gaveta{width:100vw}.nome{max-width:170px}}
@media (prefers-reduced-motion:reduce){*{transition:none !important}}
"""

HTML = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>M-Score de Beneish</title>
<meta name="description" content="Projeto educacional: M-Score de Beneish (1999) calculado com as demonstrações anuais das companhias abertas não financeiras da CVM.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap" rel="stylesheet">
<style>__CSS__</style>
<script>try{var t=localStorage.getItem('tema');if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
</head>
<body>
<header class="barra"><div class="barra-in">
<a class="marca" href="#"><b>M-Score de Beneish</b><span id="topoData">carregando…</span></a>
<nav class="nav" aria-label="Seções"><a href="#empresas">Empresas</a><a href="#metodo">Método</a><a href="https://www.alissonprata.io/">Alisson Prata</a>
<button id="tema" class="icone" aria-label="Alternar tema claro ou escuro">◐</button></nav>
</div></header>
<main>
<div class="kicker">Qualidade da informação contábil · companhias abertas não financeiras</div>
<h1>M-Score de Beneish</h1>
<p class="dek">Modelo de Beneish (1999) que estima a probabilidade de manipulação de resultados a partir de oito índices contábeis, calculado com as demonstrações financeiras padronizadas (DFP) que as companhias entregam à CVM. <span id="dekAnos"></span></p>
<div class="educ" role="note"><span aria-hidden="true">⚠</span><span>Projeto educacional e pessoal, sem fim comercial. O modelo é uma triagem estatística, não uma acusação: M-Score acima do limiar não significa fraude, e o cálculo pode conter erros de dado ou de mapeamento de contas. Nada aqui é recomendação de investimento.</span></div>

<div class="tiles" id="tiles"></div>

<section id="distribuicao">
<h2>Distribuição do M-Score <button type="button" class="info" data-info="mscore">i</button></h2>
<p class="muted pequeno">Cada barra conta as empresas numa faixa de 0,25 de M-Score (com o filtro "só com dado completo" da tabela). À direita da linha tracejada (limiar de −1,78), o modelo classifica como possível manipuladora. Clique numa barra para filtrar a tabela.</p>
<div class="painel" style="padding:8px 6px"><svg id="hist" role="img" aria-label="Histograma do M-Score das empresas"></svg></div>
</section>

<section id="empresas">
<h2>Empresas</h2>
<p class="muted pequeno">Clique numa linha para ver os oito índices, a formação do M-Score e os números de base. Por padrão a tabela esconde as empresas com dado a conferir; desmarque "só com dado completo" para vê-las. O símbolo ⚠ indica dado a conferir: índice fora da faixa usual, índice neutralizado ou exercício substituído.</p>
<div class="painel">
<div class="filtros">
<input id="fBusca" type="search" placeholder="Buscar empresa, ticker ou CNPJ" aria-label="Buscar empresa">
<select id="fSetor" aria-label="Setor"><option value="">Todos os setores</option></select>
<select id="fAlerta" aria-label="Nível de alerta"><option value="">Todos os alertas</option><option>Crítico</option><option>Alto Risco</option><option>Atenção</option><option>Normal</option></select>
<label class="chk"><input type="checkbox" id="fAcima"> só acima do limiar</label>
<label class="chk"><input type="checkbox" id="fCompleto" checked> só com dado completo</label>
<label class="chk" id="fEventoRot" hidden><input type="checkbox" id="fEvento"> com evento nos documentos</label>
<span class="espaco"></span><span id="fInfo" class="pequeno fraco" aria-live="polite"></span><button id="fLimpar" hidden>Limpar filtros</button>
</div>
<div class="tabela"><table id="tab"><thead><tr>
<th class="t" data-k="nome">Empresa</th><th class="t" data-k="setor">Setor</th>
<th data-k="mscore">M-Score <button type="button" class="info" data-info="mscore">i</button></th>
<th class="t" data-k="alerta">Alerta <button type="button" class="info" data-info="alerta">i</button></th>
<th data-k="accrual">Accrual ratio <button type="button" class="info" data-info="accrual">i</button></th>
<th class="t" data-k="qualidade">Qualidade dos lucros <button type="button" class="info" data-info="qualidade">i</button></th>
<th class="t" data-k="flags">Principal sinal</th><th data-k="nev">Eventos <button type="button" class="info" data-info="jev">i</button></th></tr></thead><tbody></tbody></table>
<p id="vazio" class="vazio" hidden>Nenhuma empresa com esses filtros.</p></div>
</div>
<p class="pequeno fraco" id="semDados"></p>
</section>

<section id="metodo">
<h2>Método e limites</h2>
<ul class="metodo pequeno">
<li><b>Dados.</b> Demonstrações financeiras padronizadas (DFP) anuais e consolidadas do Portal de Dados Abertos da CVM, para as companhias abertas ativas classificadas como não financeiras no cadastro da CVM (bancos, seguradoras e similares ficam de fora, porque o modelo foi estimado para empresas industriais e comerciais).</li>
<li><b>Modelo.</b> M-Score de oito variáveis de Beneish (1999), com os coeficientes do artigo original e limiar de −1,78: acima dele, a empresa é classificada como possível manipuladora.</li>
<li><b>Qualidade dos lucros.</b> Accrual ratio = (lucro líquido − fluxo de caixa operacional) ÷ ativo total médio. O nível de alerta combina essa leitura com o M-Score.</li>
<li><b>Dado ausente.</b> Quando uma conta não é encontrada, o coletor usa zero; quando um índice não pode ser calculado ou sai negativo, ele vira 1 (neutro). Quando o exercício pedido não existe, o coletor usa o mais próximo. Índices fora de uma faixa larga (acima de 10 ou abaixo de 0,1; TATA além de ±1) também são marcados, porque costumam indicar conta mapeada errado ou ano atípico. As fichas mostram esses casos com ⚠, e o filtro "só com dado completo" os exclui.</li>
<li><b>O que o modelo não mede.</b> Mudança de norma contábil, reorganização societária, setores com ciclo próprio e o contexto de cada empresa. O modelo foi estimado com empresas americanas dos anos 1980 e 1990; aplicado ao Brasil, é uma triagem, não um diagnóstico.</li>
<li><b>Eventos nos documentos (JEV).</b> O modelo System One da TypeSafe faz uma triagem pelo título dos fatos relevantes e comunicados ao mercado dos últimos 12 meses e, nos que o título aponta algum evento, lê o texto do documento para indicar se tratam de reapresentação de demonstrações, auditoria, saída de diretor financeiro ou de RI, investigação, reestruturação de dívida ou mudança de política contábil. Só entram classificações com confiança de pelo menos 0,8; títulos genéricos não são enviados. Documento só com imagem, sem texto extraível, fica com a classificação pelo título. É informação complementar: não altera o M-Score nem o nível de alerta.</li>
<li><b>Referência.</b> Beneish, M. D. (1999). The Detection of Earnings Manipulation. Financial Analysts Journal, 55(5), 24 a 36.</li>
</ul>
<h3>Glossário</h3>
<dl id="glossario" class="glossario"></dl>
</section>

<footer><p id="rodape"></p>Fonte: CVM, Portal de Dados Abertos (cadastro de companhias abertas e DFP). Projeto educacional de <a href="https://www.alissonprata.io/">Alisson Prata</a>; código em <a href="https://github.com/alissondpoliveira/Evaluating-Quality-of-Financial-Reports">GitHub</a>. Não constitui recomendação de investimento.</footer>
</main>
<div id="fundo" class="fundo"></div>
<aside id="gaveta" class="gaveta" role="dialog" aria-labelledby="gTitulo" hidden>
<div class="gav-topo"><div id="gCab" style="flex:1;min-width:0"></div><button id="gFechar" class="icone" aria-label="Fechar a ficha">✕</button></div>
<div id="gCorpo" class="gav-corpo"></div>
</aside>
<div id="tip" class="tip"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>__JS__</script>
</body>
</html>
"""

JS = r"""
const fmt = (v, c=2) => v==null||isNaN(v) ? '–' : Number(v).toLocaleString('pt-BR',{minimumFractionDigits:c,maximumFractionDigits:c}).replace('-', '−');
const esc = t => String(t ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
const semA = t => String(t||'').normalize('NFD').replace(/[̀-ͯ]/g,'').toLowerCase();
const tip = document.getElementById('tip');
const mostrar = (e, h) => { tip.innerHTML = h; tip.style.display = 'block'; mover(e); };
const mover = e => { tip.style.left = Math.min(e.clientX+14, innerWidth-tip.offsetWidth-8)+'px'; tip.style.top = (e.clientY+14)+'px'; };
const esconder = () => tip.style.display = 'none';
const COR_ALERTA = {'Crítico':'var(--pos-2)','Alto Risco':'var(--pos-1)','Atenção':'var(--aviso)','Normal':'var(--neu)'};
const ORDEM_ALERTA = {'Crítico':3,'Alto Risco':2,'Atenção':1,'Normal':0};
const NOMES = {dsri:'Dias de vendas em recebíveis', gmi:'Margem bruta', aqi:'Qualidade dos ativos', sgi:'Crescimento das vendas', depi:'Depreciação', sgai:'Despesas de vendas, gerais e administrativas', lvgi:'Alavancagem', tata:'Accruals totais sobre ativos'};
const SINAL = {dsri:'recebíveis crescem mais rápido que a receita', gmi:'margem bruta em queda', aqi:'mais ativos de difícil realização', sgi:'crescimento de vendas acelerado', depi:'depreciação desacelerando', sgai:'despesas de vendas e administrativas crescendo acima da receita', lvgi:'alavancagem em alta', tata:'lucro acima do caixa operacional'};
const CAMPOS = [['revenues','Receita líquida'],['cost_of_goods_sold','Custo dos produtos vendidos'],['sales_general_admin_expenses','Despesas de vendas, gerais e administrativas'],['receivables','Contas a receber'],['total_assets','Ativo total'],['current_assets','Ativo circulante'],['pp_and_e','Imobilizado'],['securities','Títulos e aplicações'],['total_long_term_debt','Dívida de longo prazo'],['current_liabilities','Passivo circulante'],['depreciation','Depreciação e amortização'],['net_income','Lucro líquido'],['cash_from_operations','Caixa das operações']];

// glossário: o "i" ao lado de cada número
const GLOSS = {
  mscore:{t:'M-Score de Beneish', d:'Combinação linear de oito índices contábeis, estimada por Beneish (1999) com empresas que manipularam resultados e empresas de controle. Quanto maior, mais o perfil contábil da empresa se parece com o das manipuladoras.', fx:['M = -4{,}84 + 0{,}920\\,DSRI + 0{,}528\\,GMI + 0{,}404\\,AQI + 0{,}892\\,SGI + 0{,}115\\,DEPI - 0{,}172\\,SGAI + 4{,}679\\,TATA - 0{,}327\\,LVGI'], l:'Acima de −1,78, o modelo classifica como possível manipuladora. É triagem estatística: aponta onde olhar, não prova nada.'},
  limiar:{t:'Limiar de −1,78', d:'Ponto de corte usado por Beneish para separar possíveis manipuladoras das demais, que equilibra os erros de classificação na amostra original.', l:'Empresas logo acima ou logo abaixo do limiar têm perfil parecido; a fronteira não é nítida.'},
  dsri:{t:'DSRI: índice de dias de vendas em recebíveis', d:'Compara o peso das contas a receber sobre a receita neste ano com o do ano anterior.', fx:['DSRI = \\frac{Receb_t / Receita_t}{Receb_{t-1} / Receita_{t-1}}'], l:'Acima de 1, os recebíveis cresceram mais que a receita: pode ser venda antecipada ou crédito mais frouxo.'},
  gmi:{t:'GMI: índice de margem bruta', d:'Margem bruta do ano anterior dividida pela deste ano.', fx:['GMI = \\frac{MB_{t-1}}{MB_t}, \\quad MB = \\frac{Receita - CPV}{Receita}'], l:'Acima de 1, a margem caiu: empresa sob pressão tem mais incentivo para ajustar resultado.'},
  aqi:{t:'AQI: índice de qualidade dos ativos', d:'Peso dos ativos que não são circulantes nem imobilizado (por exemplo, intangíveis e custos diferidos) neste ano contra o anterior.', fx:['AQI = \\frac{1 - (AC_t + Imob_t)/AT_t}{1 - (AC_{t-1} + Imob_{t-1})/AT_{t-1}}'], l:'Acima de 1, cresceu a parte do ativo de realização mais incerta, onde é mais fácil capitalizar custos.'},
  sgi:{t:'SGI: índice de crescimento das vendas', d:'Receita deste ano dividida pela do ano anterior.', fx:['SGI = \\frac{Receita_t}{Receita_{t-1}}'], l:'Crescimento não é manipulação, mas empresas em expansão acelerada sofrem mais pressão para manter o ritmo.'},
  depi:{t:'DEPI: índice de depreciação', d:'Taxa de depreciação do ano anterior dividida pela deste ano.', fx:['DEPI = \\frac{d_{t-1}}{d_t}, \\quad d = \\frac{Deprec}{Deprec + Imob}'], l:'Acima de 1, a empresa está depreciando mais devagar, o que aumenta o lucro.'},
  sgai:{t:'SGAI: índice de despesas de vendas, gerais e administrativas', d:'Peso dessas despesas sobre a receita neste ano contra o anterior.', fx:['SGAI = \\frac{SGA_t / Receita_t}{SGA_{t-1} / Receita_{t-1}}'], l:'No modelo original o coeficiente é negativo; o projeto marca como sinal quando o índice sobe acima da referência.'},
  lvgi:{t:'LVGI: índice de alavancagem', d:'Dívida total (passivo circulante mais dívida de longo prazo) sobre o ativo, neste ano contra o anterior.', fx:['LVGI = \\frac{(PC_t + DLP_t)/AT_t}{(PC_{t-1} + DLP_{t-1})/AT_{t-1}}'], l:'Alavancagem crescente aproxima a empresa de covenants de dívida. No modelo original o coeficiente é negativo.'},
  tata:{t:'TATA: accruals totais sobre ativos', d:'Parte do lucro que não virou caixa operacional, em relação ao ativo.', fx:['TATA = \\frac{LL_t - FCO_t}{AT_t}'], l:'É o índice de maior peso no modelo: lucro muito acima do caixa é o sinal mais forte de resultado contábil agressivo.'},
  accrual:{t:'Accrual ratio', d:'Quanto do lucro do ano não passou pelo caixa das operações, medido contra o tamanho da empresa (ativo total médio dos dois anos). A diferença entre lucro e caixa são os accruals: receitas e despesas reconhecidas pelo regime de competência antes de virarem dinheiro, como vendas a prazo, aumento de estoque ou provisões.', fx:['\text{Accruals} = LL_t - FCO_t','\text{Accrual ratio} = \frac{LL_t - FCO_t}{(AT_t + AT_{t-1})/2}'], p:['LL: lucro ou prejuízo consolidado do período na DRE (conta 3.11).','FCO: caixa líquido das atividades operacionais na DFC (conta 6.01).','AT: ativo total no balanço (conta 1), média do exercício e do anterior.'], l:'Negativo: o caixa operacional superou o lucro. Perto de zero: lucro e caixa andam juntos. Positivo e alto: parte relevante do lucro ainda está em contas a receber, estoques ou outros lançamentos. A ficha de cada empresa mostra a conta com os números dela.'},
  setor:{t:'Posição no setor', d:'Percentil da empresa entre as companhias do mesmo setor com dado completo neste exercício: p50 é a mediana do setor, p90 quer dizer que 90% das empresas do setor têm valor menor. Setor com menos de 10 empresas usa a amostra toda. Só exibição: o sinal, a qualidade dos lucros e o nível de alerta seguem as réguas fixas.', l:'As réguas fixas vêm da amostra de Beneish (empresas americanas, décadas de 1980 e 1990) e valem igual para todos os setores. O percentil mostra se o valor é comum no setor no Brasil: incorporadoras, por exemplo, reconhecem lucro antes de receber, e accrual alto é frequente entre elas.'},
  qualidade:{t:'Qualidade dos lucros', d:'Classe atribuída ao accrual ratio. A ideia vem de Sloan (1996): o componente do lucro que não é caixa tende a se reverter nos anos seguintes, então lucro sustentado por caixa costuma persistir mais que lucro sustentado por lançamentos contábeis.', p:['Calcula os accruals: lucro líquido menos caixa das operações.','Divide pelo ativo total médio, para comparar empresas de tamanhos diferentes.','Classifica: abaixo de 0,01 é alta; de 0,01 até 0,05 é moderada; a partir de 0,05 é baixa.','Junta com o M-Score no nível de alerta: qualidade baixa leva a Atenção ou, acima do limiar, a Crítico.'], l:'É uma régua simples e fixa, igual para todos os setores. Empresa crescendo rápido, com mais capital de giro, ou ano com evento não recorrente pode cair em baixa sem nada de errado; a classe indica onde olhar, não conclui.'},
  alerta:{t:'Nível de alerta', d:'Combina o M-Score com a qualidade dos lucros.', l:'Crítico: acima do limiar e com qualidade baixa. Alto risco: acima do limiar. Atenção: abaixo do limiar, mas com qualidade baixa. Normal: nenhum dos dois.'},
  jev:{t:'Eventos nos documentos (JEV)', d:'Fatos relevantes e comunicados ao mercado dos últimos 12 meses que o JEV (modelo System One da TypeSafe) classificou, pelo texto do documento, como um evento que a análise forense costuma cruzar com o M-Score: reapresentação de demonstrações, auditoria, saída de diretor financeiro ou de RI, investigação, reestruturação de dívida ou mudança de política contábil.', l:'Só aparecem classificações com confiança de pelo menos 0,8. Não altera o M-Score. É um apontamento para ler o documento original, não uma conclusão.'},
  contrib:{t:'Formação do M-Score', d:'Cada barra é o coeficiente do índice multiplicado pelo valor do índice. Somadas à constante (−4,84), dão o M-Score.', fx:['M = -4{,}84 + \\sum_i \\beta_i \\, I_i'], l:'Mostra qual índice puxou o resultado. Um índice igual a 1 contribui com o próprio coeficiente.'},
};
let _katex = null;
const carregarKatex = () => _katex || (_katex = new Promise(res => {
  const l = document.createElement('link'); l.rel = 'stylesheet'; l.href = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css'; document.head.appendChild(l);
  const s = document.createElement('script'); s.src = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js'; s.onload = () => { res(true); montarGlossario(); }; s.onerror = () => res(false); document.head.appendChild(s); }));
const formula = g => g.fx ? (window.katex ? '<div class="eqs">'+g.fx.map(x => katex.renderToString(x, {displayMode:true, throwOnError:false})).join('')+'</div>' : g.fx.map(x => '<span class="formula">'+esc(x)+'</span>').join('')) : '';
const passos = g => g.p ? '<ol class="passos">'+g.p.map(x => '<li>'+x+'</li>').join('')+'</ol>' : '';
const infoBtn = k => '<button type="button" class="info" data-info="'+k+'" aria-label="O que é: '+GLOSS[k].t+'">i</button>';
function montarGlossario(){ document.getElementById('glossario').innerHTML = Object.values(GLOSS).map(g => '<div><dt>'+g.t+'</dt><dd>'+g.d+formula(g)+passos(g)+(g.l?'<span class="fraco">Como ler: '+g.l+'</span>':'')+'</dd></div>').join(''); }
(function(){
  const pop = document.createElement('div'); pop.className = 'info-pop'; pop.id = 'infoPop'; pop.setAttribute('role','tooltip'); document.body.appendChild(pop);
  let fixo = null, atual = null;
  function abrir(b){ const g = GLOSS[b.dataset.info]; if (!g) return; atual = b;
    pop.innerHTML = '<b>'+g.t+'</b><p>'+g.d+'</p>'+formula(g)+passos(g)+(g.l?'<p><span class="fraco">Como ler:</span> '+g.l+'</p>':'');
    pop.style.display = 'block'; const r = b.getBoundingClientRect(), w = Math.min(360, innerWidth-16); pop.style.width = w+'px';
    pop.style.left = Math.max(8, Math.min(r.left + r.width/2 - w/2, innerWidth - w - 8))+'px';
    const h = pop.offsetHeight; pop.style.top = (r.bottom+8+h <= innerHeight-8 ? r.bottom+8 : r.top-8-h >= 8 ? r.top-8-h : Math.max(8, innerHeight-h-8))+'px';
    if (g.fx && !window.katex) carregarKatex().then(ok => { if (ok && atual===b && pop.style.display==='block') abrir(b); }); }
  const fechar = () => { pop.style.display = 'none'; fixo = null; };
  document.addEventListener('mouseover', e => { const b = e.target.closest('.info'); if (b && !fixo) abrir(b); });
  document.addEventListener('mouseout', e => { const b = e.target.closest('.info'); if (b && !fixo && !b.contains(e.relatedTarget)) fechar(); });
  document.addEventListener('click', e => { const b = e.target.closest('.info'); if (b) { e.preventDefault(); e.stopPropagation(); if (fixo===b) fechar(); else { abrir(b); fixo = b; } return; } if (fixo && !e.target.closest('.info-pop')) fechar(); }, true);
  document.addEventListener('keydown', e => { if (e.key==='Escape' && pop.style.display==='block') { fechar(); e.stopPropagation(); } }, true);
  montarGlossario();
  if (location.hash==='#metodo') carregarKatex();
  document.querySelector('a[href="#metodo"]').addEventListener('click', () => carregarKatex());
})();
document.getElementById('tema').onclick = () => { const r = document.documentElement, escuro = r.dataset.theme ? r.dataset.theme==='dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  r.dataset.theme = escuro ? 'light' : 'dark'; try { localStorage.setItem('tema', r.dataset.theme); } catch(e) {} };

Promise.all([fetch('/dados.json', {cache:'no-cache'}).then(r => r.json()), fetch('/eventos.json', {cache:'no-cache'}).then(r => r.ok ? r.json() : null).catch(() => null)]).then(([d, ev]) => iniciar(d, ev)).catch(() => { document.getElementById('topoData').textContent = 'falha ao carregar os dados; recarregue a página'; });

function iniciar(D, EV){
  const EVE = (EV && EV.empresas) || {}, NOME_EV = {reapresentacao:'Reapresentação de demonstrações', auditoria:'Auditoria', saida_executivo:'Saída de executivo', investigacao:'Investigação ou sanção', reestruturacao_divida:'Reestruturação de dívida', politica_contabil:'Política contábil ou baixa'};
  const E = D.empresas, ok = E.filter(e => !e.erro && e.mscore!=null), lim = D.limiar;
  ok.forEach(e => e.nev = (EVE[e.cnpj]||[]).length);
  if (EV && EV.classificados) document.getElementById('fEventoRot').hidden = false;
  const completo = e => !e.avisos;
  // Posição no setor (só exibição): percentil entre as empresas com dado completo do mesmo setor;
  // setor com menos de 10 empresas usa a amostra toda.
  const MIN_SETOR = 10, base = ok.filter(e => completo(e) && e.indices && e.accrual != null);
  const porSetor = d3.group(base, e => e.setor);
  const grupoDe = e => (porSetor.get(e.setor) || []).length >= MIN_SETOR ? {nome: e.setor, lista: porSetor.get(e.setor)} : {nome: 'amostra toda', lista: base};
  const valor = (e, k) => k === 'accrual' ? e.accrual : e.indices[k];
  function posicao(e, k){ const g = grupoDe(e), v = valor(e, k); if (v == null) return null;
    const vs = g.lista.map(x => valor(x, k)).filter(x => x != null).sort((a, b) => a - b);
    const abaixo = vs.filter(x => x < v).length, iguais = vs.filter(x => x === v).length;
    return {p: Math.round(100 * (abaixo + iguais / 2) / vs.length), med: d3.median(vs), n: vs.length, grupo: g.nome}; }
  const pTxt = q => q ? 'p'+q.p+' <span class="fraco">('+esc(q.grupo)+', '+q.n+')</span>' : '–';
  document.getElementById('topoData').textContent = 'exercício '+D.exercicio+' contra '+(D.exercicio-1)+' · atualizado '+D.gerado_em;
  document.getElementById('dekAnos').textContent = 'Exercício '+D.exercicio+' contra '+(D.exercicio-1)+'.';
  document.getElementById('rodape').textContent = 'Exercício '+D.exercicio+' contra '+(D.exercicio-1)+'. Dados gerados em '+D.gerado_em+' (horário de Brasília).';
  const acima = ok.filter(e => e.mscore > lim);
  document.getElementById('tiles').innerHTML = [
    ['Empresas calculadas', ok.length, 'de '+E.length+' não financeiras ativas'],
    ['Acima do limiar', acima.length, fmt(100*acima.length/Math.max(ok.length,1),0)+'% das calculadas'],
    ['Alerta crítico', ok.filter(e => e.alerta==='Crítico').length, 'acima do limiar e lucro pouco sustentado por caixa'],
    ['Com dado a conferir', ok.filter(e => !completo(e)).length, 'índice extremo, neutralizado ou exercício substituído'],
    ['M-Score mediano', fmt(d3.median(ok, e => e.mscore),2), 'limiar: '+fmt(lim,2)],
  ].map(x => '<div class="tile"><span>'+x[0]+'</span><b>'+x[1]+'</b><small>'+x[2]+'</small></div>').join('');
  document.getElementById('semDados').textContent = (E.length-ok.length)+' empresas ficaram sem M-Score por falta de demonstração nos dois exercícios ou de contas essenciais.';
  [...new Set(ok.map(e => e.setor))].sort().forEach(s => { const o = document.createElement('option'); o.value = o.textContent = s; document.getElementById('fSetor').appendChild(o); });

  // ---------- histograma ----------
  let faixa = null;
  function desenharHist(){
    const svg = d3.select('#hist'), larg = document.getElementById('hist').parentNode.clientWidth - 12; svg.selectAll('*').remove();
    const W = Math.max(320, larg), H = 220, M = {t:16, r:12, b:30, l:36}; svg.attr('viewBox', `0 0 ${W} ${H}`);
    const lo = -5, hi = 2, passo = .25, bins = d3.range(lo, hi, passo).map(a => ({a, b:a+passo, n:0}));
    const base = document.getElementById('fCompleto').checked ? ok.filter(completo) : ok;
    base.forEach(e => { const v = Math.max(lo, Math.min(hi - 1e-9, e.mscore)); bins[Math.floor((v-lo)/passo)].n++; });
    const x = d3.scaleLinear().domain([lo, hi]).range([M.l, W-M.r]), y = d3.scaleLinear().domain([0, d3.max(bins, b => b.n)||1]).nice().range([H-M.b, M.t]);
    y.ticks(4).forEach(t => { svg.append('line').attr('x1',M.l).attr('x2',W-M.r).attr('y1',y(t)).attr('y2',y(t)).attr('stroke','var(--line)');
      svg.append('text').attr('x',M.l-6).attr('y',y(t)+3).attr('text-anchor','end').attr('font-size',10).attr('fill','var(--ink-3)').text(t); });
    d3.range(lo, hi+.01, 1).forEach(t => svg.append('text').attr('x',x(t)).attr('y',H-12).attr('text-anchor','middle').attr('font-size',10).attr('fill','var(--ink-3)').text((t===lo?'≤ ':t===hi?'≥ ':'')+fmt(t,0)));
    svg.selectAll('rect.b').data(bins).join('rect').attr('class','b').attr('x', b => x(b.a)+1).attr('width', b => Math.max(1, x(b.b)-x(b.a)-2)).attr('y', b => y(b.n)).attr('height', b => y(0)-y(b.n)).attr('rx',2)
      .attr('fill', b => b.a >= lim - 1e-9 || b.b > lim + 1e-9 ? 'var(--pos-2)' : 'var(--neu)').attr('opacity', b => faixa && faixa.a!==b.a ? .35 : 1).style('cursor','pointer')
      .on('mouseenter', (e,b) => mostrar(e, '<b>'+b.n+' empresas</b><br>M-Score de '+fmt(b.a,2)+' a '+fmt(b.b,2)+'<br><span class="fraco">clique para filtrar</span>')).on('mousemove', mover).on('mouseleave', esconder)
      .on('click', (e,b) => { faixa = faixa && faixa.a===b.a ? null : b; desenharHist(); filtrar(); });
    svg.append('line').attr('x1',x(lim)).attr('x2',x(lim)).attr('y1',M.t-6).attr('y2',H-M.b).attr('stroke','var(--ink)').attr('stroke-dasharray','4 3');
    svg.append('text').attr('x',x(lim)+5).attr('y',M.t+4).attr('font-size',10.5).attr('fill','var(--ink)').text('limiar −1,78');
    svg.append('text').attr('x',(M.l+W-M.r)/2).attr('y',H-1).attr('text-anchor','middle').attr('font-size',10).attr('fill','var(--ink-3)').text('M-Score');
  }
  desenharHist(); addEventListener('resize', () => { clearTimeout(window._rh); window._rh = setTimeout(desenharHist, 150); });

  // ---------- tabela ----------
  const corpo = document.querySelector('#tab tbody');
  const tk = e => /\d$/.test(e.ticker||'') ? e.ticker : '';
  const primeiroSinal = e => { const k = Object.keys(SINAL).find(k => (e.flags||[]).some(f => f.toLowerCase().startsWith(k))); return k ? k.toUpperCase()+': '+SINAL[k] : ''; };
  let ordem = {k:'mscore', asc:false};
  function linhas(){
    const q = semA(document.getElementById('fBusca').value).trim(), s = document.getElementById('fSetor').value, al = document.getElementById('fAlerta').value;
    const so = document.getElementById('fAcima').checked, comp = document.getElementById('fCompleto').checked, cev = document.getElementById('fEvento').checked;
    let r = ok.filter(e => (!q || semA(e.nome+' '+e.ticker+' '+e.cnpj).includes(q)) && (!s || e.setor===s) && (!al || e.alerta===al) && (!so || e.mscore > lim) && (!comp || completo(e)) && (!cev || e.nev > 0) && (!faixa || (e.mscore >= faixa.a && e.mscore < faixa.b) || (faixa.a===-5 && e.mscore < -5) || (faixa.b===2 && e.mscore >= 2)));
    const v = e => ordem.k==='alerta' ? ORDEM_ALERTA[e.alerta] : ordem.k==='flags' ? primeiroSinal(e) : e[ordem.k];
    r.sort((a,b) => { const x = v(a), y = v(b); return (x==null) - (y==null) || (x>y?1:x<y?-1:0)*(ordem.asc?1:-1); });
    document.getElementById('fLimpar').hidden = !(q || s || al || so || !comp || cev || faixa);
    document.getElementById('fInfo').textContent = r.length===ok.length ? r.length+' empresas' : r.length+' de '+ok.length+' empresas';
    document.getElementById('vazio').hidden = r.length > 0;
    return r;
  }
  function filtrar(){
    corpo.innerHTML = linhas().map(e => '<tr data-cnpj="'+e.cnpj+'" tabindex="0"><td class="t"><span class="nome" title="'+esc(e.nome)+'">'+esc(e.nome)+'</span> <span class="fraco">'+esc(tk(e))+'</span>'+(completo(e)?'':' <span class="av" title="dado incompleto">⚠</span>')+'</td><td class="t">'+esc(e.setor)+'</td>'
      +'<td>'+fmt(e.mscore,2)+'</td><td class="t"><span class="badge"><i style="background:'+COR_ALERTA[e.alerta]+'"></i>'+e.alerta+'</span></td><td>'+fmt(e.accrual,3)+'</td><td class="t">'+(e.qualidade||'–')+'</td><td class="t fraco">'+esc(primeiroSinal(e))+'</td><td>'+(e.nev ? '<span title="eventos nos documentos dos últimos 12 meses">⚑ '+e.nev+'</span>' : '<span class="fraco">–</span>')+'</td></tr>').join('');
  }
  ['fBusca','fSetor','fAlerta','fAcima','fCompleto','fEvento'].forEach(id => document.getElementById(id).addEventListener(id==='fBusca'?'input':'change', filtrar));
  document.getElementById('fCompleto').addEventListener('change', desenharHist);
  document.getElementById('fLimpar').onclick = () => { ['fBusca','fSetor','fAlerta'].forEach(id => document.getElementById(id).value=''); document.getElementById('fAcima').checked = false; document.getElementById('fCompleto').checked = true; document.getElementById('fEvento').checked = false; faixa = null; desenharHist(); filtrar(); };
  document.querySelectorAll('#tab th[data-k]').forEach(th => th.addEventListener('click', () => { ordem = {k:th.dataset.k, asc: ordem.k===th.dataset.k ? !ordem.asc : th.classList.contains('t')}; filtrar(); }));
  corpo.addEventListener('click', e => { const tr = e.target.closest('tr[data-cnpj]'); if (tr) abrirFicha(tr.dataset.cnpj); });
  corpo.addEventListener('keydown', e => { const tr = e.target.closest('tr[data-cnpj]'); if (tr && e.key==='Enter') abrirFicha(tr.dataset.cnpj); });
  filtrar();

  // ---------- ficha ----------
  const gav = document.getElementById('gaveta'), porCnpj = new Map(E.map(e => [e.cnpj, e]));
  let empilhou = false;
  function abrirFicha(cnpj, daUrl){
    const e = porCnpj.get(cnpj); if (!e) return;
    const u = new URL(location.href); u.searchParams.set('empresa', cnpj);
    if (daUrl || document.body.classList.contains('aberta')) history.replaceState(history.state, '', u); else { history.pushState({ficha:1}, '', u); empilhou = true; }
    const cnpjF = cnpj.replace(/^(\d{2})(\d{3})(\d{3})(\d{4})(\d{2})$/,'$1.$2.$3/$4-$5');
    document.getElementById('gCab').innerHTML = '<div class="kicker">'+esc(e.setor)+' · CNPJ '+cnpjF+'</div><h2 id="gTitulo">'+esc(e.nome)+'</h2><span class="fraco pequeno">'+(tk(e)?esc(tk(e))+' · ':'')+'exercício '+D.exercicio+' contra '+(D.exercicio-1)+'</span>';
    const av = e.avisos, alerta = av ? '<div class="avisos"><b>⚠ Dado a conferir.</b> '+[
        av.exercicio_substituido && av.exercicio_substituido.length ? 'O coletor não achou o exercício pedido e usou '+[...new Set(av.exercicio_substituido)].join(' e ')+'.' : '',
        av.contas_ausentes && Object.keys(av.contas_ausentes).length ? 'Contas não encontradas (calculadas como zero): '+Object.entries(av.contas_ausentes).map(([a, cs]) => a+': '+cs.map(c => (CAMPOS.find(x => x[0]===c)||[c,c])[1].toLowerCase()).join(', ')).join('; ')+'.' : '',
        av.indices_neutralizados && av.indices_neutralizados.length ? 'Índices que saíram negativos e foram trocados por 1: '+av.indices_neutralizados.join(', ')+'.' : '',
        av.indices_extremos && av.indices_extremos.length ? 'Índice fora da faixa usual ('+av.indices_extremos.map(k => k.toUpperCase()+' = '+fmt(e.indices[k], 2)).join(', ')+'): pode ser conta mapeada errado ou um ano atípico.' : ''
      ].filter(Boolean).join(' ')+' O M-Score desta empresa deve ser lido com cuidado.</div>' : '';
    const ix = Object.keys(NOMES), ref = D.referencias_alerta;
    document.getElementById('gCorpo').innerHTML = alerta
      + '<div class="grade">'
      + '<div><small>M-Score '+infoBtn('mscore')+'</small><b>'+fmt(e.mscore,2)+'</b><em>'+(e.mscore > lim ? 'acima' : 'abaixo')+' do limiar de −1,78</em></div>'
      + '<div><small>Nível de alerta '+infoBtn('alerta')+'</small><b><span class="badge" style="font-size:1rem"><i style="background:'+COR_ALERTA[e.alerta]+';width:11px;height:11px"></i>'+e.alerta+'</span></b><em>'+(e.classificacao==='Potential Manipulator'?'possível manipuladora':'não manipuladora')+' pelo modelo</em></div>'
      + '<div><small>Accrual ratio '+infoBtn('accrual')+'</small><b>'+fmt(e.accrual,3)+'</b><em>qualidade dos lucros: '+(e.qualidade||'–').toLowerCase()+'</em></div>'
      + '<div><small>Risco normalizado</small><b>'+fmt(e.risco,1)+'</b><em>de 0 a 10 (M-Score de −5 a +2)</em></div></div>'
      + (() => { const t = e.fin && e.fin.t, t1 = e.fin && e.fin.t1; if (!t || t.net_income==null || t.cash_from_operations==null || !t.total_assets) return '';
          const acc = t.net_income - t.cash_from_operations, at = (t.total_assets + (t1.total_assets||0))/2, r = acc/at;
          const faixa = r < 0.01 ? 'abaixo de 0,01: qualidade alta' : r < 0.05 ? 'entre 0,01 e 0,05: qualidade moderada' : 'a partir de 0,05: qualidade baixa';
          return '<h3>Como chegamos na qualidade dos lucros '+infoBtn('qualidade')+'</h3><div class="tabela" style="max-height:none;border:1px solid var(--line);border-radius:8px"><table><tbody>'
            + '<tr><td class="t">Lucro líquido '+D.exercicio+' (DRE 3.11)</td><td>'+fmt(t.net_income,0)+'</td></tr>'
            + '<tr><td class="t">(−) Caixa das operações '+D.exercicio+' (DFC 6.01)</td><td>'+fmt(t.cash_from_operations,0)+'</td></tr>'
            + '<tr><td class="t"><b>(=) Accruals</b>: lucro que não passou pelo caixa</td><td><b>'+fmt(acc,0)+'</b></td></tr>'
            + '<tr><td class="t">(÷) Ativo total médio ('+fmt(t.total_assets,0)+' e '+fmt(t1.total_assets,0)+')</td><td>'+fmt(at,0)+'</td></tr>'
            + '<tr><td class="t"><b>(=) Accrual ratio</b></td><td><b>'+fmt(r,3)+'</b></td></tr></tbody></table></div>'
            + (() => { const q = posicao(e, 'accrual'); return q ? '<p class="pequeno">No setor '+infoBtn('setor')+': percentil '+q.p+' em '+esc(q.grupo)+' ('+q.n+' empresas, mediana '+fmt(q.med,3)+'). '+(q.p >= 75 ? 'Está entre os 25% com mais accruals do grupo.' : q.p <= 25 ? 'Está entre os 25% com menos accruals do grupo.' : 'Fica na faixa central do grupo.')+'</p>' : ''; })()
            + '<p class="pequeno fraco">'+(acc < 0 ? 'O caixa das operações superou o lucro. ' : '')+'Resultado '+faixa+'. Valores na escala publicada pela companhia (em geral, R$ mil).</p>'; })()
      + '<h3>Formação do M-Score '+infoBtn('contrib')+'</h3><svg id="contrib" role="img" aria-label="Contribuição de cada índice para o M-Score"></svg>'
      + '<h3>Os oito índices</h3><div class="tabela" style="max-height:none;border:1px solid var(--line);border-radius:8px"><table><thead><tr><th class="t">Índice</th><th>Valor</th><th>Referência de alerta</th><th>No setor '+infoBtn('setor')+'</th><th>Contribuição</th></tr></thead><tbody>'
      + ix.map(k => { const v = e.indices[k], sinal = v!=null && v > ref[k]; return '<tr><td class="t"><b>'+k.toUpperCase()+'</b> '+infoBtn(k)+'<br><span class="fraco">'+NOMES[k]+'</span></td><td>'+fmt(v, k==='tata'?4:3)+(sinal?' <span class="av" title="acima da referência">▲</span>':'')+'</td><td>'+fmt(ref[k], k==='tata'?3:3)+'</td><td>'+pTxt(posicao(e, k))+'</td><td>'+(e.contrib[k]>0?'+':'')+fmt(e.contrib[k],3)+'</td></tr>'; }).join('')
      + '</tbody></table></div>'
      + ((e.flags||[]).length ? '<h3>Sinais mais fortes</h3><ul class="pequeno" style="padding-left:1.1rem;margin:.2rem 0">'+ix.filter(k => (e.flags||[]).some(f => f.toLowerCase().startsWith(k))).map(k => '<li><b>'+k.toUpperCase()+'</b> em '+fmt(e.indices[k], k==='tata'?4:2)+': '+SINAL[k]+'</li>').join('')+'</ul>' : '<p class="pequeno muted">Nenhum índice acima da referência de alerta.</p>')
      + (() => { const ev = EVE[e.cnpj] || []; if (!EV || !EV.classificados) return '';
          return '<h3>Eventos nos documentos (JEV) '+infoBtn('jev')+'</h3>' + (ev.length ? '<ul class="pequeno" style="padding-left:1.1rem;margin:.2rem 0">'+ev.map(x => '<li><span class="fraco">'+x.data.split('-').reverse().join('/')+'</span> <b>'+NOME_EV[x.evento]+'</b>: <a href="'+esc(x.url)+'" target="_blank" rel="noopener">'+esc(x.titulo)+'</a> <span class="fraco">('+x.categoria.toLowerCase()+', confiança '+fmt(x.confianca,2)+(x.base==='titulo'?', classificado pelo título':'')+')</span></li>').join('')+'</ul>' : '<p class="pequeno muted">Nenhum evento desse tipo nos fatos relevantes e comunicados dos últimos 12 meses.</p>')
            + '<p class="pequeno fraco">Classificação automática pelo JEV; leia o documento original. Não altera o M-Score.</p>'; })()
      + '<h3>Números de base</h3><p class="pequeno fraco" style="margin-top:0">'+(e.individual ? 'Demonstrações individuais da DFP'+(e.individual.length===1 ? ' em '+e.individual[0] : '')+': a companhia não publica consolidado com valores (em geral, por não ter controladas). ' : 'Demonstrações consolidadas da DFP. ')+'Escala publicada pela companhia (em geral, R$ mil).</p>'
      + '<div class="tabela" style="max-height:none;border:1px solid var(--line);border-radius:8px"><table><thead><tr><th class="t">Conta</th><th>'+D.exercicio+'</th><th>'+(D.exercicio-1)+'</th><th>Variação</th></tr></thead><tbody>'
      + CAMPOS.map(([k, n]) => { const a = e.fin.t[k], b = e.fin.t1[k], vr = b ? (a/b-1)*100 : null; return '<tr><td class="t">'+n+'</td><td>'+fmt(a,0)+'</td><td>'+fmt(b,0)+'</td><td>'+(vr==null||!isFinite(vr)?'–':(vr>0?'+':'')+fmt(vr,1)+'%')+'</td></tr>'; }).join('')
      + '</tbody></table></div>';
    desenharContrib(e);
    gav.hidden = false; requestAnimationFrame(() => document.body.classList.add('aberta')); document.getElementById('gFechar').focus({preventScroll:true});
  }
  function desenharContrib(e){
    const svg = d3.select('#contrib'), W = 500, lin = 24, M = {t:6, r:64, b:6, l:58};
    const itens = [{k:'constante', v:D.coeficientes.intercept}, ...Object.keys(NOMES).map(k => ({k, v:e.contrib[k]}))];
    const H = M.t + (itens.length+1)*lin + M.b; svg.attr('viewBox', `0 0 ${W} ${H}`);
    let acc = 0; itens.forEach(it => { it.a = acc; acc += it.v; it.b = acc; });
    const vals = itens.flatMap(it => [it.a, it.b]).concat([lim]), x = d3.scaleLinear().domain([Math.min(0, d3.min(vals)), Math.max(0, d3.max(vals))]).nice().range([M.l, W-M.r]);
    svg.append('line').attr('x1',x(lim)).attr('x2',x(lim)).attr('y1',M.t).attr('y2',H-M.b).attr('stroke','var(--ink)').attr('stroke-dasharray','4 3');
    svg.append('line').attr('x1',x(0)).attr('x2',x(0)).attr('y1',M.t).attr('y2',H-M.b).attr('stroke','var(--line-2)');
    itens.forEach((it, i) => { const y = M.t + i*lin;
      svg.append('text').attr('x',0).attr('y',y+lin/2+4).attr('font-size',11).attr('fill','var(--ink-2)').text(it.k==='constante'?'constante':it.k.toUpperCase());
      svg.append('rect').attr('x',x(Math.min(it.a,it.b))).attr('y',y+5).attr('width',Math.max(1.5,Math.abs(x(it.b)-x(it.a)))).attr('height',lin-10).attr('rx',2).attr('fill', it.k==='constante' ? 'var(--neu)' : it.v > 0 ? 'var(--pos-2)' : 'var(--neg-2)');
      svg.append('text').attr('x',W-M.r+6).attr('y',y+lin/2+4).attr('font-size',11).attr('fill','var(--ink)').style('font-variant-numeric','tabular-nums').text((it.v>0?'+':'')+fmt(it.v,2)); });
    const yf = M.t + itens.length*lin;
    svg.append('text').attr('x',0).attr('y',yf+lin/2+4).attr('font-size',11).attr('font-weight',700).attr('fill','var(--ink)').text('M-Score');
    svg.append('circle').attr('cx',x(acc)).attr('cy',yf+lin/2).attr('r',6).attr('fill', acc > lim ? 'var(--pos-2)' : 'var(--neg-2)').attr('stroke','var(--surface)').attr('stroke-width',2);
    svg.append('text').attr('x',W-M.r+6).attr('y',yf+lin/2+4).attr('font-size',11).attr('font-weight',700).attr('fill','var(--ink)').text(fmt(acc,2));
    svg.append('text').attr('x',x(lim)+4).attr('y',H-M.b-2).attr('font-size',9.5).attr('fill','var(--ink-3)').text('limiar');
  }
  function fecharFicha(viaHistorico){
    if (!document.body.classList.contains('aberta')) return;
    if (!viaHistorico && empilhou && history.state && history.state.ficha) { empilhou = false; history.back(); return; }
    document.body.classList.remove('aberta'); setTimeout(() => { if (!document.body.classList.contains('aberta')) gav.hidden = true; }, 280);
    const u = new URL(location.href); if (u.searchParams.has('empresa')) { u.searchParams.delete('empresa'); history.replaceState(null, '', u); }
  }
  document.getElementById('gFechar').onclick = () => fecharFicha(); document.getElementById('fundo').onclick = () => fecharFicha();
  document.addEventListener('keydown', ev => { if (ev.key==='Escape') fecharFicha(); });
  addEventListener('popstate', () => { const c = new URLSearchParams(location.search).get('empresa'); c ? abrirFicha(c, true) : fecharFicha(true); });
  const inicial = new URLSearchParams(location.search).get('empresa'); if (inicial) abrirFicha(inicial, true);
}
"""


def main() -> None:
    PUBLICO.mkdir(parents=True, exist_ok=True)
    (PUBLICO / "index.html").write_text(HTML.replace("__CSS__", CSS).replace("__JS__", JS), encoding="utf-8")
    cabecalhos = [{"key": "Content-Security-Policy", "value": CSP},
                  {"key": "Strict-Transport-Security", "value": "max-age=63072000; includeSubDomains"},
                  {"key": "X-Content-Type-Options", "value": "nosniff"}, {"key": "X-Frame-Options", "value": "DENY"},
                  {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
                  {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=(), payment=(), usb=()"}]
    (PUBLICO / "vercel.json").write_text(json.dumps({"headers": [{"source": "/(.*)", "headers": cabecalhos}]}, indent=2) + "\n", encoding="utf-8")
    # vercel.json da raiz do repositório: sem framework e sem build, publica site/public.
    # Sem ele a Vercel detecta o requirements.txt do app Streamlit e tenta montar um projeto Python.
    raiz = {"framework": None, "installCommand": "echo sem dependencias", "buildCommand": "echo pagina estatica",
            "outputDirectory": "site/public", "headers": [{"source": "/(.*)", "headers": cabecalhos}]}
    (PUBLICO.parent.parent / "vercel.json").write_text(json.dumps(raiz, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print("site/public gerado")


if __name__ == "__main__":
    main()
