"""Extrai os textos visíveis (pt-BR) do painel global para a tabela de tradução.

Uso: python i18n/extrair_textos.py   (dentro de utils/fire_global_monitor/)
Saída: i18n/textos_novos.csv, só com os textos que ainda não estão em i18n/traducoes.csv nem em
i18n/extra.csv. Na primeira execução, gera i18n/traducoes.csv completo.
Traduza as colunas de idioma (en, es, fr, ind, zh) e junte as linhas ao traducoes.csv.
"""
import csv
import html as H
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "i18n" / "traducoes.csv"
OUT = ROOT / "i18n" / "textos_novos.csv"
KNOWN = [BASE, ROOT / "i18n" / "extra.csv"]
LANG_COLS = ["en", "es", "fr", "ind", "zh"]
WORD = re.compile(r"[A-Za-zÀ-ÿ]{2,}")
NOT_TEXT = re.compile(r"^(#|var\(|rgba?\(|hsl|https?:|data:|\d|[.#\[]?[\w-]+[\]=]|[\w-]+:\s|M\d)")
CODEY = re.compile(r"(=>|\$\(|\bconst\b|\bfunction\b|querySelector|getElementById|\bcss\(|\bnf\(|\.format\(|\bS\.\w|\[\w+\]|\bk\b =)")


def js_strings(src):
    """Literais de string de um código JS (aspas simples, duplas e crases), sem comentários e regex."""
    i, n, out, prev = 0, len(src), [], "("
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "'\"`":
            j, buf = i + 1, []
            while j < n and src[j] != c:
                if src[j] == "\\":
                    nxt = src[j + 1:j + 2]
                    buf.append({"n": " ", "t": " "}.get(nxt, nxt))
                    j += 2
                    continue
                if c == "`" and src.startswith("${", j):
                    depth, j = 1, j + 2
                    while j < n and depth:
                        depth += {"{": 1, "}": -1}.get(src[j], 0)
                        j += 1
                    buf.append("{…}")
                    continue
                buf.append(src[j])
                j += 1
            out.append((i, "".join(buf)))
            i, prev = j + 1, "x"
            continue
        if c == "/" and prev in "(,=:[!&|?{};+-*%<>~^":
            j, cls = i + 1, False
            while j < n and (src[j] != "/" or cls):
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    cls = True
                elif src[j] == "]":
                    cls = False
                elif src[j] == "\n":
                    break
                j += 1
            i, prev = j + 1, "x"
            continue
        if c.isalpha() or c in "_$":
            k = i
            while k < n and (src[k].isalnum() or src[k] in "_$"):
                k += 1
            prev = "(" if src[i:k] in ("return", "typeof", "case", "in", "of") else "x"
            i = k
            continue
        if not c.isspace():
            prev = c
        i += 1
    return out


class Collector:
    def __init__(self, old):
        self.rows, self.seen, self.old = [], {}, old

    def add(self, page, kind, ctx, text):
        t = re.sub(r"\s+", " ", text)
        if kind != "js":
            t = t.strip()
        if not WORD.search(t):
            return
        if t in self.seen:
            r = self.rows[self.seen[t]]
            if page not in r["pagina"]:
                r["pagina"] += " + " + page
            return
        self.seen[t] = len(self.rows)
        self.rows.append({"id": f"T{len(self.rows) + 1:04d}", "pagina": page, "tipo": kind,
                          "contexto": re.sub(r"\s+", " ", ctx)[:110], "pt": t, "en": self.old.get(t, "")})


class Page(HTMLParser):
    def __init__(self, page, col):
        super().__init__(convert_charrefs=True)
        self.page, self.col, self.skip, self.stack = page, col, 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag not in ("br", "img", "meta", "link", "input", "hr"):
            self.stack.append(tag)
        if self.skip:
            return
        for k, v in attrs:
            if k in ("title", "aria-label", "alt", "placeholder") or (k == "content" and tag == "meta" and v and " " in v and "width=" not in v):
                if v and WORD.search(v):
                    self.col.add(self.page, "atributo", f"{tag}[{k}]", v)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip -= 1
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.col.add(self.page, "html", "<" + (self.stack[-1] if self.stack else "") + ">", d)


def main():
    col = Collector({})
    src = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    Page("painel", col).feed(src)
    for sc in re.findall(r"<script>([\s\S]*?)</script>", src):
        for pos, t in js_strings(sc):
            t = H.unescape(t)
            if NOT_TEXT.match(t) or CODEY.search(t):
                continue
            if re.fullmatch(r"[a-z]+[A-Z]\w*|[\w-]+", t) and not re.search(r"[À-ÿ]", t) and (t.islower() or len(t) < 4 or not t.istitle()):
                continue  # ids, chaves e classes
            if " " not in t.strip() and re.search(r"[/=&?;]", t):
                continue  # parâmetros, tipos de arquivo
            line = sc[max(0, pos - 80):pos].split("\n")[-1].strip()
            if "<" in t or ">" in t:  # trecho de HTML montado no JS: só os textos entre as tags
                frag = re.sub(r"<[^>]*$", "", re.sub(r"^[^<]*>", "", t)) if not t.lstrip().startswith("<") else re.sub(r"<[^>]*$", "", t)
                for seg in re.split(r"<[^>]*>", frag):
                    if WORD.search(seg) and not re.search(r"[=\"]", seg):
                        col.add("painel", "js", line, seg.strip())
                continue
            if WORD.search(t) and not re.match(r'^\s*[".]|.*="', t):
                col.add("painel", "js", line, t)
    known = set()
    for k in KNOWN:
        if k.exists():
            known |= {r["pt"] for r in csv.DictReader(k.open(encoding="utf-8-sig"))}
    out = OUT if BASE.exists() else BASE
    col.rows = [dict(r, **{c: "" for c in LANG_COLS}) for r in col.rows if r["pt"] not in known]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "pagina", "tipo", "contexto", "pt", *LANG_COLS])
        w.writeheader()
        w.writerows(col.rows)
    print(out, len(col.rows), dict(Counter(r["tipo"] for r in col.rows)))


if __name__ == "__main__":
    main()
