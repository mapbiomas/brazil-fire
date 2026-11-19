"""Conexão com o Earth Engine, com mensagens claras e escolha de outro projeto.

Uso na célula 1 do notebook:

    from gif_factory.connect import conectar
    conexao = conectar(GEE_PROJECT)

Se a conta não puder usar o projeto informado, em vez de um erro longo aparece
uma explicação curta e uma lista com os projetos do Google Cloud da pessoa. Ao
escolher um, a conexão é testada na hora. A célula 2 usa `conexao.project`.
"""
from html import escape

import ipywidgets as w
from IPython.display import display

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _ee():
    import ee
    return ee


def explain(error, project):
    """(título, explicação) amigáveis para um erro de ee.Initialize."""
    text = str(error)
    low = text.lower()
    project = escape(str(project or ""))
    if "please authorize" in low or "earthengine authenticate" in low:
        return ("Falta autorizar o Earth Engine",
                "Rode a célula 1 de novo e aceite o pedido de acesso do Google.")
    if "user_project_denied" in low or "does not have required permission" in low or "permission_denied" in low:
        return (f"Sua conta não tem acesso ao projeto “{project}”",
                "Escolha abaixo um projeto da sua conta ou peça acesso a quem administra esse projeto.")
    if "not registered" in low or "not been used" in low or "is disabled" in low or "not enabled" in low:
        return (f"O projeto “{project}” não está habilitado para o Earth Engine",
                "Escolha outro projeto abaixo ou registre este em code.earthengine.google.com/register.")
    if "not found" in low or "does not exist" in low:
        return (f"O projeto “{project}” não foi encontrado",
                "Confira o nome ou escolha um projeto da sua conta abaixo.")
    if "no project" in low or "project is required" in low:
        return ("Falta informar o projeto", "Escolha abaixo um projeto da sua conta.")
    return ("Não foi possível conectar ao Earth Engine", escape(text.splitlines()[0][:300]) if text else "")


def needs_auth(error):
    low = str(error).lower()
    return "please authorize" in low or "earthengine authenticate" in low or "credentials" in low and "not found" in low


def list_projects():
    """[(rótulo, id)] dos projetos ativos do Google Cloud da conta conectada (vazio se não der)."""
    import google.auth
    from googleapiclient.discovery import build
    credentials, _ = google.auth.default(scopes=[CLOUD_SCOPE])
    service = build("cloudresourcemanager", "v1", credentials=credentials, cache_discovery=False)
    found = []
    request = service.projects().list(filter="lifecycleState:ACTIVE", pageSize=500)
    while request is not None:
        response = request.execute()
        for item in response.get("projects", []):
            pid, name = item.get("projectId"), item.get("name") or ""
            if pid:
                found.append((f"{name} ({pid})" if name and name != pid else pid, pid))
        request = service.projects().list_next(request, response)
    return sorted(found, key=lambda pair: pair[0].casefold())


class Connection:
    """Estado da conexão: `ok`, `project` e o painel mostrado no notebook."""

    def __init__(self, project, initialize=None, authenticate=None, projects=None, show=True):
        self.project = (project or "").strip()
        self.ok = False
        self._initialize = initialize or (lambda p: _ee().Initialize(project=p))
        self._authenticate = authenticate or (lambda: _ee().Authenticate())
        self._projects = projects or list_projects
        self.message = w.HTML()
        self.choice = w.Dropdown(description="Projeto:", layout=w.Layout(width="460px", display="none"),
                                 style={"description_width": "70px"})
        self.typed = w.Text(placeholder="ou digite o ID de um projeto", layout=w.Layout(width="260px", display="none"))
        self.use = w.Button(description="Usar este projeto", icon="check", button_style="success",
                            layout=w.Layout(display="none", width="180px"))
        self.use.on_click(lambda _b: self.try_project(self.typed.value.strip() or self.choice.value))
        self.widget = w.VBox([self.message, w.HBox([self.choice, self.typed]), self.use])
        if show:
            display(self.widget)

    def _say(self, color, title, text=""):
        self.message.value = (f"<div style='border-left:4px solid {color};padding:6px 12px;margin:4px 0;"
                              f"background:#fffbf4;font-size:14px'><b>{title}</b>"
                              + (f"<div style='font-size:13px;color:#5c5044'>{text}</div>" if text else "")
                              + "</div>")

    def connect(self):
        if not self.project:
            self._say("#96702a", "Falta informar o projeto", "Escolha abaixo um projeto da sua conta.")
            self.offer_projects()
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
        self._say("#5f8160", f"✓ Conectado ao Earth Engine · projeto {escape(self.project)}")
        for widget in (self.choice, self.typed, self.use):
            widget.layout.display = "none"
        return self

    def _failed(self, error):
        self.ok = False
        title, text = explain(error, self.project)
        if needs_auth(error):
            self._say("#a33b2b", title, text)
            return self
        self._say("#a33b2b", title, text)
        self.offer_projects()
        return self

    def offer_projects(self):
        try:
            projects = [p for p in self._projects() if p[1] != self.project]
        except Exception:
            projects = []
        self.typed.layout.display = ""
        self.use.layout.display = ""
        if projects:
            self.choice.options = projects
            self.choice.value = projects[0][1]
            self.choice.layout.display = ""
            hint = (f"{len(projects)} projeto(s) encontrados na sua conta. Depois de conectar, rode a célula 2. "
                    "Na próxima vez, deixe esse projeto escrito em GEE_PROJECT.")
        else:
            hint = ("Não consegui listar os seus projetos. Digite o ID de um projeto com Earth Engine "
                    "(aparece em console.cloud.google.com) e clique em “Usar este projeto”.")
        self.message.value += f"<div style='font-size:12px;color:#7a6c5c;margin:2px 0 4px'>{hint}</div>"

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
        self.message.value += ("<div style='font-size:12px;color:#7a6c5c'>Pronto: agora rode a célula 2. "
                               f"Para não escolher de novo, escreva <code>{escape(project)}</code> em GEE_PROJECT.</div>")
        return self

    def require(self):
        """Projeto conectado, ou uma mensagem clara de que falta conectar."""
        if not self.ok:
            raise RuntimeError("O Earth Engine ainda não está conectado. Escolha um projeto na célula 1 "
                               "e clique em “Usar este projeto”.")
        return self.project


def conectar(project, **kwargs):
    """Conecta ao Earth Engine sem estourar erros; mostra um seletor de projetos se precisar."""
    return Connection(project, **kwargs).connect()
