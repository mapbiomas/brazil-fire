"""UI ipywidgets do mapbiomas_cog (M4 inicial).

Navegacao pais -> tema -> colecao -> produto -> unidades, com acoes ligadas ao
``Engine``: Export (submit tiled), Sync (poll), Assemble (COG). Versao enxuta;
a paridade completa com a UI do version_01 vem depois.
"""

from __future__ import annotations

import datetime
import ipywidgets as widgets
from IPython.display import display

from ..domain import config as cog_config
from ..domain import paths
from ..domain.models import Product, TaskSpec, Unit
from ..engine import Engine
from ..engine import images as ee_images
from ..assemble import assemble_unit


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


class App:
    def __init__(self, countries=None, themes=None, runner: str = "ee_batch_tiled",
                 config_path: str | None = None):
        self.obj = cog_config.load_obj(config_path)
        self.countries = cog_config.resolve_countries(self.obj, countries, themes)
        if not self.countries:
            raise ValueError("Nenhum pais disponivel no catalogo.")
        self.engine = Engine(runner=runner)
        self._units: list = []
        self._build()

    # -- widgets -------------------------------------------------------------
    def _build(self):
        self.w_country = widgets.Dropdown(options=self.countries, description="Pais")
        self.w_theme = widgets.Dropdown(options=[], description="Tema")
        self.w_collection = widgets.Dropdown(options=[], description="Colecao")
        self.w_product = widgets.Dropdown(options=[], description="Produto")
        self.w_units = widgets.SelectMultiple(options=[], description="Unidades",
                                              layout=widgets.Layout(width="420px", height="180px"))
        self.w_info = widgets.HTML()

        self.btn_load = widgets.Button(description="Load units", button_style="info")
        self.btn_export = widgets.Button(description="Export", button_style="danger")
        self.btn_sync = widgets.Button(description="Sync", button_style="warning")
        self.btn_assemble = widgets.Button(description="Assemble", button_style="success")
        self.log = widgets.HTML(layout=widgets.Layout(width="100%"))

        self.w_country.observe(lambda c: self._refresh_themes(), names="value")
        self.w_theme.observe(lambda c: self._refresh_collections(), names="value")
        self.w_collection.observe(lambda c: self._refresh_products(), names="value")
        self.w_product.observe(lambda c: self._refresh_info(), names="value")

        self.btn_load.on_click(lambda _b: self.load_units())
        self.btn_export.on_click(lambda _b: self.export())
        self.btn_sync.on_click(lambda _b: self.sync())
        self.btn_assemble.on_click(lambda _b: self.assemble())

        self.container = widgets.VBox([
            widgets.HBox([self.w_country, self.w_theme, self.w_collection, self.w_product]),
            self.w_info,
            widgets.HBox([self.btn_load, self.btn_export, self.btn_sync, self.btn_assemble]),
            self.w_units,
            self.log,
        ])
        self._refresh_themes()

    def display(self):
        display(self.container)

    # -- log -----------------------------------------------------------------
    def _log(self, message: str):
        line = f"<div style='font-family:monospace;font-size:12px'>[{_now()}] {message}</div>"
        self.log.value = line + self.log.value

    # -- cascata -------------------------------------------------------------
    def _country(self) -> str:
        return self.w_country.value

    def _refresh_themes(self):
        themes = list(self.obj.get(self._country(), {}))
        self.w_theme.options = themes
        if themes:
            self.w_theme.value = themes[0]
        self._refresh_collections()

    def _refresh_collections(self):
        colls = sorted(self.obj.get(self._country(), {}).get(self.w_theme.value, {}))
        self.w_collection.options = colls
        if colls:
            self.w_collection.value = colls[0]
        self._refresh_products()

    def _refresh_products(self):
        prods = cog_config.products(
            self.obj, self._country(), self.w_theme.value, self.w_collection.value
        )
        names = [p["product"] for p in prods]
        self.w_product.options = names
        if names:
            self.w_product.value = names[0]
        self._refresh_info()

    def _refresh_info(self):
        try:
            p = self._product()
        except Exception:
            self.w_info.value = ""
            return
        self.w_info.value = (
            f"<code>{p.assetid}</code><br>tipo={p.dtype} | scale={p.scale} | "
            f"kind={p.kind} | vectorize={p.vectorize}"
        )

    def _product(self) -> Product:
        return cog_config.make_product(
            self.obj, self._country(), self.w_theme.value,
            self.w_collection.value, self.w_product.value,
        )

    # -- acoes ---------------------------------------------------------------
    def load_units(self):
        try:
            p = self._product()
            self._units = ee_images.list_units(p.assetid, p.kind)
            self.w_units.options = self._units
            self._log(f"[LOAD] {len(self._units)} unidade(s) em {p.product}.")
        except Exception as exc:  # noqa: BLE001
            self._log(f"[ERROR] load_units: {exc}")

    def _selected_units(self) -> list:
        selected = list(self.w_units.value) or self._units
        return [Unit(key=u, kind=self._product().kind) for u in selected]

    def _tasks(self):
        product = self._product()
        for unit in self._selected_units():
            yield TaskSpec(product=product, unit=unit, runner=self.engine.runner_name)

    def export(self):
        for task in self._tasks():
            try:
                result = self.engine.run(task)
                self._log(f"[EXPORT] {task.unit_key}: submitted={result.submitted} "
                          f"({result.detail})")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] export {task.unit_key}: {exc}")

    def sync(self):
        for task in self._tasks():
            try:
                result = self.engine.sync(task)
                self._log(f"[SYNC] {task.unit_key}: {result.detail}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] sync {task.unit_key}: {exc}")

    def assemble(self):
        for task in self._tasks():
            try:
                cog = assemble_unit(task, log=self._log)
                self._log(f"[ASSEMBLE] {task.unit_key}: {cog}")
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] assemble {task.unit_key}: {exc}")


def run_ui(countries=None, themes=None, runner: str = "ee_batch_tiled",
           config_path: str | None = None) -> App:
    app = App(countries=countries, themes=themes, runner=runner, config_path=config_path)
    app.display()
    return app
