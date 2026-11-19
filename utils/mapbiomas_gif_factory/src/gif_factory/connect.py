"""Conexão com o Earth Engine, com mensagens claras e escolha de outro projeto.

Uso na célula 1 do notebook:

    from gif_factory.connect import conectar
    conexao = conectar(GEE_PROJECT)

Se a conta não puder usar o projeto informado, em vez de um erro longo aparece
uma explicação curta e um campo para digitar outro projeto, testado na hora.
A célula 2 usa o projeto que conectou (`conexao.require()`).
"""
from html import escape

import ipywidgets as w
from IPython.display import display

from .i18n import _

def _ee():
    import ee
    return ee


def explain(error, project):
    """(título, explicação) amigáveis para um erro de ee.Initialize."""
    text = str(error)
    low = text.lower()
    project = escape(str(project or ""))
    if "please authorize" in low or "earthengine authenticate" in low:
        return (_("Falta autorizar o Earth Engine"),
                _("Rode a célula 1 de novo e aceite o pedido de acesso do Google."))
    if "user_project_denied" in low or "does not have required permission" in low or "permission_denied" in low:
        return (_("Sua conta não tem acesso ao projeto “{projeto}”", projeto=project),
                _("Digite abaixo outro projeto do Earth Engine que a sua conta usa, ou peça acesso a quem "
                  "administra esse projeto."))
    if "not registered" in low or "not been used" in low or "is disabled" in low or "not enabled" in low:
        return (_("O projeto “{projeto}” não está habilitado para o Earth Engine", projeto=project),
                _("Digite outro projeto abaixo ou registre este em code.earthengine.google.com/register."))
    if "not found" in low or "does not exist" in low:
        return (_("O projeto “{projeto}” não foi encontrado", projeto=project),
                _("Confira o nome ou digite outro projeto abaixo."))
    if "no project" in low or "project is required" in low:
        return (_("Falta informar o projeto"), _("Digite abaixo um projeto do Earth Engine que a sua conta usa."))
    return (_("Não foi possível conectar ao Earth Engine"), escape(text.splitlines()[0][:300]) if text else "")


def needs_auth(error):
    low = str(error).lower()
    return "please authorize" in low or "earthengine authenticate" in low or "credentials" in low and "not found" in low


class Connection:
    """Estado da conexão: `ok`, `project` e o painel mostrado no notebook."""

    def __init__(self, project, initialize=None, authenticate=None, show=True):
        self.project = (project or "").strip()
        self.ok = False
        self._initialize = initialize or (lambda p: _ee().Initialize(project=p))
        self._authenticate = authenticate or (lambda: _ee().Authenticate())
        self.message = w.HTML()
        self.typed = w.Text(placeholder=_("ID do projeto (ex.: meu-projeto-gee)"), description=_("Projeto:"),
                            style={"description_width": "70px"}, layout=w.Layout(width="420px", display="none"))
        self.use = w.Button(description=_("Usar este projeto"), icon="check", button_style="success",
                            layout=w.Layout(display="none", width="180px"))
        self.use.on_click(lambda _b: self.try_project(self.typed.value))
        self.typed.on_submit(lambda _t: self.try_project(self.typed.value))
        self.widget = w.VBox([self.message, w.HBox([self.typed, self.use])])
        if show:
            display(self.widget)

    def _say(self, color, title, text=""):
        self.message.value = (f"<div style='border-left:4px solid {color};padding:6px 12px;margin:4px 0;"
                              f"background:#fffbf4;font-size:14px'><b>{title}</b>"
                              + (f"<div style='font-size:13px;color:#5c5044'>{text}</div>" if text else "")
                              + "</div>")

    def connect(self):
        if not self.project:
            self._say("#96702a", _("Falta informar o projeto"),
                      _("Digite abaixo um projeto do Earth Engine que a sua conta usa."))
            self.offer_project_field()
            return self
        try:
            self._initialize(self.project)
        except Exception as first:
            if needs_auth(first):
                try:
                    self._authenticate()
                    self._initialize(self.project)
                except Exception as second:
                    return self._failed(second)
            else:
                return self._failed(first)
        return self._connected()

    def _connected(self):
        self.ok = True
        self._say("#5f8160", "✓ " + _("Conectado ao Earth Engine · projeto {projeto}", projeto=escape(self.project)))
        for widget in (self.typed, self.use):
            widget.layout.display = "none"
        return self

    def _failed(self, error):
        self.ok = False
        title, text = explain(error, self.project)
        if needs_auth(error):
            self._say("#a33b2b", title, text)
            return self
        self._say("#a33b2b", title, text)
        self.offer_project_field()
        return self

    def offer_project_field(self):
        self.typed.layout.display = ""
        self.use.layout.display = ""
        self.message.value += ("<div style='font-size:12px;color:#7a6c5c;margin:2px 0 4px'>"
                               + _("O ID aparece no topo de console.cloud.google.com ou em code.earthengine.google.com. "
                                   "Depois de conectar, rode a célula 2.") + "</div>")

    def try_project(self, project):
        project = (project or "").strip()
        if not project:
            return self
        self.project = project
        try:
            self._initialize(project)
        except Exception as exc:
            return self._failed(exc)
        self._connected()
        self.message.value += ("<div style='font-size:12px;color:#7a6c5c'>"
                               + _("Pronto: agora rode a célula 2. Para não escolher de novo, escreva {projeto} em "
                                   "GEE_PROJECT.", projeto=f"<code>{escape(project)}</code>") + "</div>")
        return self

    def require(self):
        """Projeto conectado, ou uma mensagem clara de que falta conectar."""
        if not self.ok:
            raise RuntimeError(_("O Earth Engine ainda não está conectado. Digite um projeto na célula 1 "
                                 "e clique em “Usar este projeto”."))
        return self.project


def conectar(project, **kwargs):
    """Conecta ao Earth Engine sem estourar erros; se precisar, pede outro projeto."""
    return Connection(project, **kwargs).connect()
