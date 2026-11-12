"""Versão em inglês do site, traduzida a partir do HTML gerado em português (pedido do Alisson, 12/11/2026).

O português continua sendo a única fonte: os geradores montam a página em português e este módulo produz a cópia
em inglês trocando cada trecho de texto pela tradução guardada em site/i18n/en.json (memória de tradução).

Unidade de tradução: o texto corrido entre duas fronteiras de bloco (p, li, td, h2, button...), com as tags
de dentro (b, a, span...) trocadas por marcadores ⟦n⟧ e os números por ⟨n⟩. Assim a mesma frase com o número
do dia ("947 séries de 300 emissores") bate com a tradução já guardada, e o número volta no formato americano.
Atributos de texto (title, aria-label, placeholder, alt, descrição) e o JSON-LD também são traduzidos.

No JavaScript embutido, os textos entre aspas são traduzidos pela mesma memória, menos os que a lógica usa:
comparações, chaves de objeto, argumentos de seletor e de atributo, e constantes. Valores que vêm dos dados
(segmento, garantia...) são trocados na tela por script_dados().

Trecho sem tradução fica em português e entra em site/i18n/pendentes.json; site/traduzir.py traduz os pendentes
com o Claude e grava na memória. Elemento com o atributo data-i18n-skip não é traduzido (nomes de empresas,
códigos de papéis, artigos já traduzidos à parte).
"""

from __future__ import annotations

import html as _html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

BLOCOS = {"html", "head", "body", "title", "div", "p", "li", "ul", "ol", "dl", "dt", "dd", "h1", "h2", "h3", "h4", "h5",
          "h6", "table", "thead", "tbody", "tfoot", "tr", "td", "th", "section", "article", "header", "footer", "nav",
          "main", "aside", "details", "summary", "figure", "figcaption", "blockquote", "form", "label", "select",
          "option", "button", "fieldset", "legend", "caption", "hr", "pre", "svg", "textarea", "iframe", "noscript",
          "meta", "link", "input", "img"}
VAZIOS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
CRUS = {"script", "style"}
ATRIBUTOS = ("title", "aria-label", "placeholder", "alt")
META_TEXTO = {"description", "og:title", "og:description", "twitter:title", "twitter:description"}
MESES_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

_NUMERO = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|(?<![\w⟦⟨])\d+(?:[.,]\d+)*")
_MARCADOR = re.compile(r"⟦\d+⟧|⟨\d+⟩")
_LETRAS = re.compile(r"[A-Za-zÀ-ÿ]{2,}")


# ------------------------------------------------------------------ memória

class Memoria:
    """Memória de tradução (pt -> en) e lista do que ainda falta traduzir."""

    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.arq = pasta / "en.json"
        self.trad: dict[str, str] = json.loads(self.arq.read_text(encoding="utf-8")) if self.arq.exists() else {}
        self.pendentes: set[str] = set()
        self.usadas: set[str] = set()

    def get(self, chave: str) -> str | None:
        en = self.trad.get(chave)
        if en is None:
            self.pendentes.add(chave)
        else:
            self.usadas.add(chave)
        return en

    def salvar_pendentes(self) -> int:
        self.pasta.mkdir(parents=True, exist_ok=True)
        arq = self.pasta / "pendentes.json"
        if self.pendentes:
            arq.write_text(json.dumps(sorted(self.pendentes), ensure_ascii=False, indent=1), encoding="utf-8")
        elif arq.exists():
            arq.unlink()
        return len(self.pendentes)


# ------------------------------------------------------------------ números e marcadores

def numero_en(token: str) -> str:
    """Número ou data no formato brasileiro -> formato americano (1.234,5 -> 1,234.5; 03/11/2026 -> Nov 3, 2026)."""
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", token)
    if m:
        d, mes, a = int(m.group(1)), int(m.group(2)), m.group(3)
        return f"{MESES_EN[mes - 1]} {d}, {a}" if 1 <= mes <= 12 else token
    if "," in token and "." in token:
        return token.replace(".", "§").replace(",", ".").replace("§", ",")
    if "," in token:
        return token.replace(",", ".")
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", token):
        return token.replace(".", ",")
    return token


def mascarar(texto: str, tags: list[str]) -> tuple[str, list[str]]:
    """Troca números por ⟨n⟩ (as tags já vêm como ⟦n⟧). Devolve o modelo e os números."""
    nums: list[str] = []

    def troca(m):
        nums.append(m.group(0))
        return f"⟨{len(nums) - 1}⟩"
    return _NUMERO.sub(troca, texto), nums


def restaurar(modelo: str, tags: list[str], nums: list[str]) -> str:
    def troca(m):
        s = m.group(0)
        i = int(s[1:-1])
        return tags[i] if s[0] == "⟦" else numero_en(nums[i])
    return _MARCADOR.sub(troca, modelo)


def marcadores_ok(pt: str, en: str) -> bool:
    return sorted(_MARCADOR.findall(pt)) == sorted(_MARCADOR.findall(en))


def traduzir_texto(texto: str, mem: Memoria, tags: list[str] | None = None) -> str | None:
    """Texto com tags já mascaradas -> tradução, ou None se não há o que traduzir ou falta na memória."""
    tags = tags or []
    nucleo = texto.strip()
    if not _LETRAS.search(re.sub(r"&#?\w+;", " ", _MARCADOR.sub(" ", nucleo))):
        # só números (células de tabela, valores): vão para o formato americano, sem passar pela memória
        if not _NUMERO.search(_MARCADOR.sub(" ", texto)):
            return None
        modelo, nums = mascarar(texto, tags)
        return restaurar(modelo, tags, nums)
    modelo, nums = mascarar(nucleo, tags)
    en = mem.get(modelo)
    if en is None or not marcadores_ok(modelo, en):
        return None
    ini, fim = texto[:len(texto) - len(texto.lstrip())], texto[len(texto.rstrip()):]
    return ini + restaurar(en, tags, nums) + fim


# ------------------------------------------------------------------ HTML

class _Fichas(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.f: list[tuple] = []

    def handle_starttag(self, tag, attrs):
        self.f.append(("ini", tag, self.get_starttag_text(), dict(attrs)))

    def handle_startendtag(self, tag, attrs):
        self.f.append(("vazio", tag, self.get_starttag_text(), dict(attrs)))

    def handle_endtag(self, tag):
        self.f.append(("fim", tag, f"</{tag}>", {}))

    def handle_data(self, d):
        self.f.append(("txt", "", d, {}))

    def handle_entityref(self, name):
        self.f.append(("txt", "", f"&{name};", {}))

    def handle_charref(self, name):
        self.f.append(("txt", "", f"&#{name};", {}))

    def handle_comment(self, d):
        self.f.append(("cru", "", f"<!--{d}-->", {}))

    def handle_decl(self, d):
        self.f.append(("cru", "", f"<!{d}>", {}))

    def handle_pi(self, d):
        self.f.append(("cru", "", f"<?{d}>", {}))

    def unknown_decl(self, d):
        self.f.append(("cru", "", f"<![{d}]>", {}))


def _attr_texto(valor_esc: str, mem: Memoria) -> str:
    txt = _html.unescape(valor_esc)
    en = traduzir_texto(txt, mem)
    return valor_esc if en is None else _html.escape(en, quote=True)


def _tag_en(raw: str, tag: str, attrs: dict, mem: Memoria | None, link) -> str:
    """Traduz os atributos de texto de uma tag de abertura (sem mem: não traduz) e reescreve o href, menos o do
    botão de idioma (data-idioma), que aponta de propósito para a outra versão."""
    def troca(m):
        nome, aspas, valor = m.group(1).lower(), m.group(2), m.group(3)
        if mem and (nome in ATRIBUTOS or (nome == "content" and tag == "meta" and (attrs.get("name") or attrs.get("property")) in META_TEXTO)):
            valor = _attr_texto(valor, mem)
        elif nome == "href" and tag == "a" and link and "data-idioma" not in attrs:
            valor = _html.escape(link(_html.unescape(valor)), quote=True)
        return f" {m.group(1)}={aspas}{valor}{aspas}"
    return re.sub(r'\s([\w:-]+)=(["\'])(.*?)\2', troca, raw, flags=re.S)


def _jsonld_en(texto: str, mem: Memoria, url_en) -> str:
    try:
        dados = json.loads(texto)
    except ValueError:
        return texto

    def anda(x):
        if isinstance(x, dict):
            for k, v in list(x.items()):
                if isinstance(v, str) and k in ("description", "headline", "name"):
                    x[k] = traduzir_texto(v, mem) or v
                elif isinstance(v, str) and k in ("url", "mainEntityOfPage") and url_en:
                    x[k] = url_en(v)
                elif k == "inLanguage":
                    x[k] = "en"
                else:
                    anda(v)
        elif isinstance(x, list):
            for v in x:
                anda(v)
    anda(dados)
    return json.dumps(dados, ensure_ascii=False)


def traduzir_html(pagina: str, mem: Memoria, link=None, url_en=None) -> str:
    """HTML em português -> HTML em inglês. link(href) reescreve os links internos; url_en(url) as URLs canônicas
    do JSON-LD."""
    p = _Fichas()
    p.feed(pagina)
    p.close()
    f = p.f
    saida: list[str] = []
    unidade: list[tuple] = []
    pilha: list[tuple[str, bool]] = []
    cru = None  # dentro de <script>/<style>: (tag, attrs)

    def pulando() -> bool:
        return any(s for _, s in pilha)

    def descarrega():
        if not unidade:
            return
        tags, partes = [], []
        for tipo, tag, raw, attrs in unidade:
            if tipo == "txt":
                partes.append(raw)
            else:
                tags.append(_tag_en(raw, tag, attrs, mem, link) if tipo in ("ini", "vazio") else raw)
                partes.append(f"⟦{len(tags) - 1}⟧")
        texto = "".join(partes)
        en = traduzir_texto(texto, mem, tags)
        saida.append(en if en is not None else restaurar_cru(texto, tags))
        unidade.clear()

    def restaurar_cru(texto, tags):
        return re.sub(r"⟦(\d+)⟧", lambda m: tags[int(m.group(1))], texto)

    for i, (tipo, tag, raw, attrs) in enumerate(f):
        if cru:
            if tipo == "fim" and tag == cru[0]:
                cru = None
                saida.append(raw)
            elif tipo == "txt" and cru[0] == "script":
                if (cru[1].get("type") or "") == "application/ld+json":
                    saida.append(_jsonld_en(raw, mem, url_en))
                elif not cru[1].get("src"):
                    saida.append(traduzir_js(raw, mem))
                else:
                    saida.append(raw)
            else:
                saida.append(raw)
            continue
        if tipo == "ini" and tag not in VAZIOS and "data-i18n-skip" in attrs and not pulando():
            descarrega()
            saida.append(_tag_en(raw, tag, attrs, None, link))
            pilha.append((tag, True))
            continue
        if tipo == "txt" or (tipo in ("ini", "vazio", "fim") and tag not in BLOCOS and tag not in CRUS):
            if pulando():
                saida.append(_tag_en(raw, tag, attrs, None, link) if tipo in ("ini", "vazio") else raw)
            else:
                unidade.append((tipo, tag, raw, attrs))
            if tipo == "ini" and tag not in VAZIOS:
                pilha.append((tag, False))
            elif tipo == "fim":
                _fecha(pilha, tag)
            continue
        descarrega()
        if tipo in ("ini", "vazio"):
            if tag == "option" and "value" not in attrs and not pulando():
                seguintes = []
                for x in f[i + 1:]:
                    if x[0] != "txt":
                        break
                    seguintes.append(x[2])
                rotulo = "".join(seguintes).strip()
                raw = raw[:-1].rstrip("/") + f' value="{_html.escape(_html.unescape(rotulo), quote=True)}">'
            saida.append(_tag_en(raw, tag, attrs, None if pulando() else mem, link))
            if tag in CRUS and tipo == "ini":
                cru = (tag, attrs)
            elif tipo == "ini" and tag not in VAZIOS:
                pilha.append((tag, "data-i18n-skip" in attrs))
        elif tipo == "fim":
            saida.append(raw)
            _fecha(pilha, tag)
        else:
            saida.append(raw)
    descarrega()
    return "".join(saida)


def _fecha(pilha: list, tag: str) -> None:
    for j in range(len(pilha) - 1, -1, -1):
        if pilha[j][0] == tag:
            del pilha[j:]
            return


# ------------------------------------------------------------------ JavaScript

_ANTES_PROIBE = re.compile(r"(===|!==|==|!=|\bcase|\b(?:getElementById|querySelector|querySelectorAll|closest|contains|add|remove|toggle|"
                           r"setAttribute|getAttribute|removeAttribute|hasAttribute|attr|style|on|addEventListener|startsWith|endsWith|"
                           r"includes|split|indexOf|localeCompare|createElement|fetch|getItem|setItem|removeItem|select|selectAll|append|"
                           r"insert|classed|property|join|has|get|set|delete|replace|replaceAll|matches|toLocaleString|"
                           r"toLocaleDateString|Intl\.\w+)\(\s*)$")
_ATTR_2O = re.compile(r"\.(?:(?:attr|setAttribute)\(\s*'(?!aria-label'|title'|alt'|placeholder')[^']*'"
                      r"|(?:style|classed|property|on)\(\s*'[^']*')\s*,\s*$")
_LATEX = re.compile(r"(\\\\text\{)([^{}]*)(\})")


def _literais(js: str):
    """(início, fim, aspas) de cada texto entre aspas simples ou duplas, pulando comentários, regex e crases."""
    i, n = 0, len(js)
    ultimo = ""
    while i < n:
        c = js[i]
        if c == "/" and js[i + 1:i + 2] == "/":
            i = js.find("\n", i)
            i = n if i < 0 else i
            continue
        if c == "/" and js[i + 1:i + 2] == "*":
            i = js.find("*/", i + 2)
            i = n if i < 0 else i + 2
            continue
        if c == "/" and (ultimo == "" or ultimo in "(,=:[!&|?{};+-*%<>~^"):
            j, classe = i + 1, False
            while j < n and js[j] != "\n":
                if js[j] == "\\":
                    j += 2
                    continue
                if js[j] == "[":
                    classe = True
                elif js[j] == "]":
                    classe = False
                elif js[j] == "/" and not classe:
                    break
                j += 1
            i = j + 1
            ultimo = "/"
            continue
        if c in "'\"`":
            j = i + 1
            while j < n and js[j] != c:
                j += 2 if js[j] == "\\" else 1
            if c != "`":
                yield i, j + 1, c
            i = j + 1
            ultimo = c
            continue
        if not c.isspace():
            ultimo = c
        i += 1


def _js_unesc(s: str) -> str:
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


def _js_esc(s: str, aspas: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace(aspas, "\\" + aspas)


def _parece_codigo(v: str) -> bool:
    if "\\" in v and "\\\\text{" not in v:
        return True
    if re.fullmatch(r"[#.\[]?[\w:-]+(\([^)]*\))?", v) and not re.search(r"[À-ÿ]", v) and not v[:1].isupper():
        return True  # identificador, classe, seletor, var(--x)
    if v.startswith(("var(", "color-mix(", "data:")) or re.fullmatch(r"[\d\s.,%+\-−]*", v):
        return True
    if not re.search(r"\s", v) and v.startswith(("http", "/", "#", ".")):
        return True
    return False


def _contexto_de_codigo(antes: str, depois: str) -> bool:
    """Texto que a lógica usa: comparação, chave de objeto, argumento de seletor ou de atributo, constante."""
    # constante (const ISOL = 'Isolada (a identificar)') costuma ser comparada com os dados: fica como está
    return bool(_ANTES_PROIBE.search(antes) or _ATTR_2O.search(antes) or depois.startswith(("===", "!==", "==", "!="))
                or (depois.startswith(":") and antes.rstrip()[-1:] in "{,") or re.search(r"\b(const|let|var)\s+\w+\s*=\s*$", antes))


def traduzir_js(js: str, mem: Memoria) -> str:
    achados = [(ini, fim, aspas, js[ini + 1:fim - 1], js[max(0, ini - 80):ini], js[fim:fim + 6].lstrip())
               for ini, fim, aspas in _literais(js)]
    # palavras que a página usa como código (comparação, classe, id, data-*): não se traduzem nem soltas
    codigo = {v for _, _, _, v, antes, depois in achados if _contexto_de_codigo(antes, depois)}
    codigo |= {w for c in re.findall(r"(?:class|id|data-[\w-]+)=[\"']([^\"']*)", js) for w in c.split()}
    partes, pos = [], 0
    for ini, fim, aspas, v, antes, depois in achados:
        # palavra solta como resultado de ternário concatenado em texto ('<em>'+(x > lim ? 'acima' : 'abaixo')+' do
        # limiar') é texto de tela; fora disso ('light', 'change', 'true') é código
        ternario = bool(re.fullmatch(r"[a-zà-ÿ]{3,}", v) and v not in codigo
                        and re.search(r"\+\s*\((?:[^()]|\([^()]*\))*[?:]\s*$", antes) and depois.startswith((":", ")+", ") +"))
                        and not re.search(r"=[\"']'\s*\+\s*\([^()]*$", antes))  # class="'+(i===foco?'foco':'')
        if not _LETRAS.search(v) or (_parece_codigo(v) and not ternario) or _contexto_de_codigo(antes, depois):
            continue
        if "\\\\text{" in v:
            novo = _LATEX.sub(lambda m: m.group(1) + (traduzir_texto(m.group(2), mem) or m.group(2)) + m.group(3), v)
        else:
            texto = _js_unesc(v)
            tags = []

            def marca(m):
                nome = re.match(r"</?(\w+)", m.group(0))
                tags.append(_tag_en(m.group(0), nome.group(1).lower() if nome else "", {}, mem, None))
                return f"⟦{len(tags) - 1}⟧"
            mascarado = re.sub(r"<[^<>]+>", marca, texto)
            aberto = re.fullmatch(r'(.*<\w[^<>]*\s(?:title|aria-label|alt|placeholder)=")([^"<>]*)', mascarado, re.S)
            if aberto and _LETRAS.search(aberto.group(2)):
                # termina dentro de um atributo ('<span title="confiança '+x+'">'): traduz só o texto do atributo
                en = traduzir_texto(aberto.group(2), mem)
                if en is None:
                    continue
                novo = _js_esc(re.sub(r"⟦(\d+)⟧", lambda m: tags[int(m.group(1))], aberto.group(1)) + en, aspas)
                partes.append(js[pos:ini + 1] + novo)
                pos = fim - 1
                continue
            if re.search(r"[<>]|=\"|='", mascarado):
                continue  # pedaço de tag montada por concatenação
            en = traduzir_texto(mascarado, mem, tags)
            if en is None:
                # sem texto para traduzir fora das tags, mas talvez um title ou aria-label dentro delas
                en = re.sub(r"⟦(\d+)⟧", lambda m: tags[int(m.group(1))], mascarado)
                if en == texto:
                    continue
            novo = _js_esc(en, aspas)
        partes.append(js[pos:ini + 1] + novo)
        pos = fim - 1
    partes.append(js[pos:])
    return "".join(partes)


# ------------------------------------------------------------------ página inteira

MARCADOR_IDIOMA = "<!--idioma-->"
CSS_IDIOMA = (".idioma{position:fixed;top:12px;right:14px;z-index:60;font:600 .75rem/1 Inter,system-ui,sans-serif;letter-spacing:.04em;"
              "padding:6px 9px;border:1px solid var(--line-2);border-radius:999px;background:var(--surface);color:var(--ink-2);text-decoration:none}"
              ".idioma:hover{color:var(--accent);border-color:var(--accent)}.barra-acoes .idioma,.nav .idioma{position:static;margin-left:.9rem}")

# Valores que vêm dos dados (segmento, garantia, nível de alerta...) são trocados na tela, depois que o JavaScript
# os escreve: o nó de texto inteiro tem de ser igual ao valor. Opção sem value guarda o original no value, para os
# filtros continuarem comparando com o dado em português.
_JS_DADOS = """(function(){const M=__MAPA__;const k=s=>s.trim();
function tr(n){const t=n.nodeValue,v=M[k(t)];if(v===undefined)return;const p=n.parentNode;
if(p&&p.tagName==='OPTION'&&!p.hasAttribute('value'))p.setAttribute('value',k(t));n.nodeValue=t.replace(k(t),v);}
function anda(r){if(r.nodeType===3){tr(r);return;}const w=document.createTreeWalker(r,NodeFilter.SHOW_TEXT);let n;while((n=w.nextNode()))tr(n);}
new MutationObserver(ms=>ms.forEach(m=>{if(m.type==='characterData')tr(m.target);else m.addedNodes.forEach(anda);}))
.observe(document.documentElement,{childList:true,subtree:true,characterData:true});
document.addEventListener('DOMContentLoaded',()=>anda(document.body));})();"""


def script_dados(valores, mem: Memoria) -> str:
    mapa = {}
    for v in sorted({str(x).strip() for x in valores if x}):
        en = traduzir_texto(v, mem)
        if en is not None and en != v:
            mapa[v] = en
    # window.TRD: o mesmo mapa, para o JavaScript da página traduzir o que monta junto com outro texto
    js = "window.TRD=__MAPA__;" + _JS_DADOS.replace("__MAPA__", "window.TRD")
    return "<script>" + js.replace("__MAPA__", json.dumps(mapa, ensure_ascii=False).replace("</", "<\\/")) + "</script>" if mapa else ""


def finalizar(pagina_pt: str, en: bool, mem: Memoria | None = None, link=None, valores_dados=()) -> str:
    """Põe o botão de idioma (no lugar de MARCADOR_IDIOMA) e, para o inglês, traduz a página, troca o idioma do
    documento e aponta canônico e og:url para a versão em inglês. O caminho da página vem de <meta name="caminho">."""
    m = re.search(r'<meta name="caminho" content="([^"]*)">', pagina_pt)
    caminho = m.group(1) if m else "/"
    cookie = "document.cookie='lang=%s;path=/;max-age=31536000;SameSite=Lax';this.href=this.getAttribute('href')+location.hash"
    destino_en = "/en" + caminho if caminho.startswith("/") else caminho
    botao = (f'<a class="idioma" data-i18n-skip data-idioma href="{caminho}" hreflang="pt-BR" lang="pt-BR" onclick="{cookie % "pt"}" title="Ver em português">PT</a>' if en
             else f'<a class="idioma" data-i18n-skip data-idioma href="{destino_en}" hreflang="en" lang="en" onclick="{cookie % "en"}" title="View in English">EN</a>')
    saida = pagina_pt.replace(MARCADOR_IDIOMA, botao)
    if not en:
        return saida
    saida = traduzir_html(saida, mem, link=link, url_en=link)
    saida = saida.replace('<html lang="pt-BR">', '<html lang="en">', 1).replace("'pt-BR'", "'en-US'")
    if link:
        saida = re.sub(r'(<link rel="canonical" href=")([^"]+)', lambda x: x.group(1) + link(x.group(2)), saida)
        saida = re.sub(r'(<meta property="og:url" content=")([^"]+)', lambda x: x.group(1) + link(x.group(2)), saida)
    if valores_dados:
        saida = saida.replace("</head>", script_dados(valores_dados, mem) + "</head>", 1)
    return saida


def link_en_para(*bases: str):
    """Regra de link da versão em inglês: endereços destas bases (e caminhos relativos à raiz) ganham /en; arquivos
    de dados e imagens não."""
    def link(h: str) -> str:
        for base in bases:
            if h.startswith(base + "/") or h == base:
                resto = h[len(base):] or "/"
                return h if resto.startswith("/en/") or resto == "/en" else base + ("/en/" if resto == "/" else "/en" + resto)
        if h.startswith("/") and not h.startswith(("/en/", "//")) and h != "/en" and not re.search(r"\.(json|png|jpe?g|gif|webp|svg|xml|txt|csv)$", h):
            return "/en/" if h == "/" else "/en" + h
        return h
    return link
