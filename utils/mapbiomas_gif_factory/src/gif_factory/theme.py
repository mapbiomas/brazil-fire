"""Tema claro terroso da Fábrica (mesma paleta da vitrine e do mapa do repositório brazil-fire).

As cores ficam como variáveis CSS em `.gf-root`; os estilos inline da interface usam
`var(--...)`, então tudo muda num lugar só. Cada cor tem uma função:
terracota = ação principal e passo atual, sálvia = escolhido/pronto, mostarda = atenção,
areia = fundo, guias e botões secundários.
"""
import ipywidgets as w

TOKENS = {
    "bg": "#f6f1e7", "surface": "#fffbf4", "ink": "#3b3228", "muted": "#7a6c5c", "line": "#e6dccb",
    "accent": "#c0643f", "accent-soft": "#f5e2d5", "accent-ink": "#9c4a2b",
    "green": "#5f8160", "green-soft": "#e2eadb", "warn": "#96702a", "warn-soft": "#f3e6c6",
    "danger": "#a33b2b", "danger-soft": "#f6dcd5", "sel": "#f0e7d8", "sky": "#7f9cab", "sky-soft": "#e3ebee",
    "shadow": "rgba(92, 70, 45, .12)",
}

BUTTON = ".gf-root .jupyter-button, .gf-root .widget-button, .gf-root .widget-toggle-button"

CSS = """
.gf-root { %(vars)s
  background: var(--bg); color: var(--ink); padding: 14px 16px 16px; border-radius: 14px;
  font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; box-sizing: border-box; }
.gf-root .widget-html, .gf-root .widget-label, .gf-root .widget-html-content, .gf-root .widget-label-basic
  { color: var(--ink); }
.gf-root h2, .gf-root h3, .gf-root h4 { color: var(--ink); font-weight: 650; letter-spacing: .1px; }
.gf-title h2 { margin: 2px 0 2px; font-size: 22px; }
.gf-title .gf-sub { color: var(--muted); margin: 0 0 4px; }
.gf-section h3 { margin: 10px 0 4px; padding-left: 10px; border-left: 4px solid var(--accent); font-size: 17px; }
.gf-section p { color: var(--muted); margin: 2px 0 8px; }
.gf-label { font-size: 12px; font-weight: 700; color: var(--muted); text-transform: uppercase;
  letter-spacing: .6px; margin-top: 8px; }

%(button)s { background: var(--sel) !important; color: var(--ink) !important; border: 1px solid var(--line) !important;
  border-radius: 8px !important; box-shadow: none !important; }
.gf-root .jupyter-button:hover:enabled, .gf-root .widget-toggle-button:hover:enabled
  { box-shadow: 0 1px 4px var(--shadow) !important; border-color: var(--muted) !important; }
.gf-root .jupyter-button.mod-success { background: var(--accent) !important; color: #fff !important;
  border-color: var(--accent) !important; font-weight: 600; }
.gf-root .jupyter-button.mod-primary { background: var(--green) !important; color: #fff !important;
  border-color: var(--green) !important; font-weight: 600; }
.gf-root .widget-toggle-button.mod-active { background: var(--green-soft) !important;
  border-color: var(--green) !important; color: var(--ink) !important; font-weight: 600; }
.gf-root .jupyter-button:disabled { opacity: .55; }

.gf-steps .widget-toggle-button { background: var(--surface) !important; border-radius: 999px !important;
  margin: 2px 6px 2px 0; }
.gf-steps .widget-toggle-button.mod-active { background: var(--accent) !important; color: #fff !important;
  border-color: var(--accent) !important; }

.gf-tabs .widget-toggle-button { border-radius: 8px 8px 0 0 !important; border-bottom: 2px solid var(--line) !important;
  background: var(--bg) !important; }
.gf-tabs .widget-toggle-button.mod-active { background: var(--surface) !important;
  border-bottom: 2px solid var(--accent) !important; color: var(--accent-ink) !important; }

.gf-option .widget-toggle-button, .gf-root .gf-option.widget-toggle-button { background: var(--surface) !important;
  text-align: left; padding-left: 12px; }
.gf-root .gf-option.widget-toggle-button.mod-active { background: var(--green-soft) !important;
  border-left: 4px solid var(--green) !important; }

.gf-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px;
  box-shadow: 0 1px 3px var(--shadow); }
.gf-card { background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 12px;
  box-shadow: 0 1px 3px var(--shadow); }
.gf-card:hover { box-shadow: 0 3px 10px var(--shadow); }

.gf-root input, .gf-root select, .gf-root textarea { background: var(--surface) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 7px !important; }
.gf-root input:focus, .gf-root select:focus, .gf-root textarea:focus { border-color: var(--accent) !important;
  outline: none; box-shadow: 0 0 0 2px var(--accent-soft); }
.gf-root .progress-bar, .gf-root .widget-progress .progress-bar { background-color: var(--accent) !important; }
.gf-root .progress, .gf-root .widget-progress .progress { background: var(--sel) !important; border-radius: 6px; }
.gf-root .jupyter-widget-Accordion, .gf-root .p-Accordion, .gf-root .lm-Accordion { border-radius: 8px; }
.gf-root code { background: var(--sel); border-radius: 4px; padding: 0 4px; }
.gf-root details summary { cursor: pointer; }
"""


def css():
    variables = " ".join(f"--{k}: {v};" for k, v in TOKENS.items())
    return CSS % {"vars": variables, "button": BUTTON}


def style_widget():
    """Bloco <style> a ser colocado dentro da raiz da interface."""
    return w.HTML(f"<style>{css()}</style>", layout=w.Layout(height="0px", margin="0"))


def classed(widget, *names):
    for name in names:
        widget.add_class(name)
    return widget


def section(title, text=""):
    body = f"<h3>{title}</h3>" + (f"<p>{text}</p>" if text else "")
    return classed(w.HTML(body), "gf-section")


def label(text):
    return classed(w.HTML(f"<div class='gf-label'>{text}</div>"), "gf-label-box")
