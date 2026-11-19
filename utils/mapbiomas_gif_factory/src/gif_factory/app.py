"""Fábrica de GIFs MapBiomas: interface em 4 passos para o Google Colab.

1. Mapa        — escolher produtos do catálogo (busca, tema, coleção)
2. Território  — buscar nas bases oficiais, desenhar no mapa, enviar arquivo ou colar asset
3. Período e saídas — anos, tipos de arquivo, destino e prévia
4. Gerar       — conferir, gerar e ver os resultados

Ninguém precisa editar código: tudo é escolhido na interface.
"""
from html import escape
from pathlib import Path
from threading import Thread

import ipywidgets as w
from IPython.display import display

from . import analyst, preview, territories
from .catalog import catalog_layers
from .engine import validate
from .legends import INITIATIVES, theme_label
from .runner import BatchRunner, drive_mounted
from .user_library import UserLibrary

STEPS = ["1 · Mapa", "2 · Território", "3 · Período e saídas", "4 · Gerar"]
KINDS = [("gif", "GIF animado", True), ("colagem", "Colagem (anos lado a lado)", False),
         ("finais", "Imagens de cada ano (mapa completo)", False)]
ADVANCED_KINDS = [("brutos", "Imagens de cada ano sem moldura"), ("preparados", "Imagens de cada ano com ano e escala")]
DEFAULT_DRIVE_FOLDER = "/content/drive/MyDrive/MapBiomas_GIF_Factory"
TERRITORY_BASES = [("Brasil", "brasil"), ("Perú", "peru"), ("Paraguay", "paraguay"), ("RD Congo", "drc"),
                   ("Meus territórios", "meus")]
MY_MAPS = "Meus mapas"


def _ee():
    import ee
    return ee


def _swatches(legend, limit=30):
    items = []
    for item in (legend or [])[:limit]:
        color = escape(str(item.get("color", "")).lstrip("#"))
        items.append(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px">'
                     f'<span style="width:14px;height:14px;border:1px solid #888;background:#{color}"></span>'
                     f'{escape(str(item.get("label", "")))}</div>')
    more = f'<div style="font-size:11px;color:#777">+ {len(legend) - limit} classes</div>' if len(legend or []) > limit else ""
    return "".join(items) + more


def _card(html):
    return w.HTML(f'<div style="border:1px solid #ddd;border-radius:6px;padding:8px;background:#fafafa">{html}</div>')


class FactoryApp:
    def __init__(self, project, layers=(), territories=(), output_root="/content/mapbiomas_gif_factory_output",
                 temp_root="/content/temp", show=True, library=None):
        self.library = library if isinstance(library, UserLibrary) else UserLibrary(library)
        saved = {layer["id"]: layer for layer in self.library.maps}
        saved.update({layer["id"]: layer for layer in layers or ()})
        layers = [dict(layer, initiative=MY_MAPS, status=layer.get("status", "experimental")) for layer in saved.values()]
        validate(layers, list(territories or ()))
        self.project = project
        self.catalog = catalog_layers(("validado", "beta")) + layers
        self.by_id = {layer["id"]: layer for layer in self.catalog}
        self.selected_layers = []        # ids, na ordem de escolha
        self.selected_territories = []   # dicionários de território
        self.year_ranges = {}            # id da camada -> (ano inicial, ano final) ou None
        self._territories_started = False  # a busca no GEE só começa ao abrir o passo 2
        self.runner = BatchRunner(output_root, temp_root, project)
        self.runner.on_finished = self._show_results
        self.summary = w.HTML()
        self.stepper = w.ToggleButtons(options=STEPS, value=STEPS[0], style={"button_width": "190px"})
        self.back = w.Button(description="Voltar", icon="arrow-left")
        self.next = w.Button(description="Avançar", icon="arrow-right", button_style="primary")
        self.pages = [self._build_maps(), self._build_territories(), self._build_outputs(), self._build_generate()]
        self.stepper.observe(self._go, names="value")
        self.back.on_click(lambda _: self._move(-1))
        self.next.on_click(lambda _: self._move(1))
        title = w.HTML("<h2 style='margin:4px 0'>Fábrica de GIFs MapBiomas</h2>"
                       "<p style='margin:0 0 6px;color:#555'>Escolha o mapa, o território e o período. "
                       "Nada aqui exige programação.</p>")
        self.root = w.VBox([title, self.stepper, self.summary, *self.pages, w.HBox([self.back, self.next])])
        self._render_products()
        for territory in territories or ():
            self.selected_territories.append(dict(territory))
        self._render_chosen()
        self._update_summary()
        self._go()
        if show:
            display(self.root)

    # ------------------------------------------------------------------ navegação
    def _move(self, delta):
        index = STEPS.index(self.stepper.value) + delta
        if 0 <= index < len(STEPS):
            self.stepper.value = STEPS[index]

    def _go(self, *_):
        index = STEPS.index(self.stepper.value)
        for i, page in enumerate(self.pages):
            page.layout.display = "" if i == index else "none"
        self.back.disabled = index == 0
        self.next.layout.display = "none" if index == len(STEPS) - 1 else ""
        if index == 1 and not self._territories_started:
            self._territories_started = True
            self._start_search()
        if index == 2:
            self._load_years()
        if index == 3:
            self._render_plan()

    def _update_summary(self):
        maps = [self.by_id[i]["name"] for i in self.selected_layers]
        places = [t["name"] for t in self.selected_territories]
        kinds = [label for key, label, _ in KINDS if self.kind_checks[key].value] + \
                [label for key, label in ADVANCED_KINDS if self.kind_checks[key].value]
        def fmt(items):
            if not items:
                return "<i style='color:#999'>nenhum</i>"
            text = ", ".join(escape(i) for i in items[:3])
            return text + (f" e mais {len(items) - 3}" if len(items) > 3 else "")
        period = "todos os anos" if self.all_years.value else f"{self.period.value[0]}–{self.period.value[1]}"
        total = len(maps) * len(places)
        self.summary.value = (
            "<div style='border-left:4px solid #2e7d32;padding:4px 10px;margin:6px 0;background:#f3f8f3;font-size:13px'>"
            f"<b>Mapas:</b> {fmt(maps)} &nbsp;·&nbsp; <b>Territórios:</b> {fmt(places)} &nbsp;·&nbsp; "
            f"<b>Período:</b> {period} &nbsp;·&nbsp; <b>Arquivos:</b> {fmt(kinds)}"
            f" &nbsp;·&nbsp; <b>{total} combinação(ões)</b></div>")

    # ------------------------------------------------------------------ 1. mapa
    def _build_maps(self):
        initiatives = list(dict.fromkeys(layer["initiative"] for layer in self.catalog))
        initiatives.sort(key=lambda name: (name != "Brasil", name == MY_MAPS, name))
        self.initiative = w.Dropdown(options=initiatives, value=initiatives[0], description="Iniciativa:")
        self.map_search = w.Text(placeholder="Buscar mapa (ex.: frequência, cobertura nível 2)",
                                 continuous_update=True, layout=w.Layout(width="420px"))
        self.theme = w.Dropdown(description="Tema:")
        self.collection = w.Dropdown(description="Coleção:")
        self.show_beta = w.Checkbox(value=False, description="Mostrar mapas beta (em revisão)", indent=False)
        self.product_list = w.VBox(layout=w.Layout(max_height="360px", overflow_y="auto", width="100%"))
        self.product_detail = _card("<i>Clique em um mapa para ver a legenda.</i>")
        self.product_buttons = {}
        for widget in (self.initiative, self.theme, self.collection, self.show_beta):
            widget.observe(self._render_products, names="value")
        self.map_search.observe(self._render_products, names="value")
        self.initiative.observe(self._refresh_filters, names="value")
        self.show_beta.observe(self._refresh_filters, names="value")
        self._refresh_filters()
        self.new_map = w.ToggleButton(description="Novo mapa a partir de um asset", icon="plus",
                                      tooltip="Para mapas beta ou fora do catálogo", layout=w.Layout(width="280px"))
        self.beta_panel = self._build_beta_panel()
        self.my_maps_panel = w.VBox()
        self.proposal = w.Textarea(layout=w.Layout(width="98%", height="220px", display="none"))
        self.new_map.observe(self._toggle_beta_panel, names="value")
        self.initiative.observe(self._render_my_maps, names="value")
        left = w.VBox([self.initiative, self.map_search, w.HBox([self.theme, self.collection]), self.show_beta,
                       self.product_list, self.my_maps_panel], layout=w.Layout(width="62%"))
        right = w.VBox([w.HTML("<b>Legenda</b>"), self.product_detail], layout=w.Layout(width="36%"))
        self._render_my_maps()
        return w.VBox([w.HTML("<h3>Qual mapa?</h3><p>Escolha um ou mais mapas. "
                              "Eles serão gerados para cada território escolhido no passo 2.</p>"),
                       w.HBox([left, right]), self.new_map, self.beta_panel])

    # ------------------------------------------------------------------ 1b. modo analista
    def _build_beta_panel(self):
        self.beta_asset = w.Text(placeholder="projects/.../assets/minha_imagem", description="Asset:",
                                 layout=w.Layout(width="560px"))
        self.beta_read = w.Button(description="Ler asset", icon="search")
        self.beta_info = w.HTML()
        self.beta_name = w.Text(placeholder="Nome que aparece no título do mapa", description="Nome:",
                                layout=w.Layout(width="560px"))
        self.beta_theme = w.Dropdown(options=analyst.THEMES, value="fire", description="Tema:")
        self.beta_collection = w.Text(value="beta", description="Coleção:")
        self.beta_mode = w.ToggleButtons(options=["Estilo do catálogo", "Minhas cores"], style={"button_width": "160px"})
        self.beta_style = w.Dropdown(description="Estilo:", layout=w.Layout(width="560px"))
        self.beta_style_legend = w.HTML()
        self.beta_min = w.FloatText(value=0, description="Mínimo:")
        self.beta_max = w.FloatText(value=1, description="Máximo:")
        self.beta_detect = w.Button(description="Detectar valores", icon="calculator")
        self.beta_colors = w.Text(placeholder="ex.: ffffff, ffcc00, 990000", description="Cores:",
                                  layout=w.Layout(width="560px"))
        self.beta_labels = w.Text(placeholder="opcional: um rótulo por cor (ex.: Baixo, Médio, Alto)",
                                  description="Rótulos:", layout=w.Layout(width="560px"))
        self.beta_title = w.Text(placeholder="opcional: título da legenda", description="Legenda:")
        self.beta_preview = w.Button(description="Ver prévia", icon="eye")
        self.beta_save = w.Button(description="Salvar em Meus mapas", icon="save", button_style="success",
                                  layout=w.Layout(width="220px"))
        self.beta_msg = w.HTML()
        self.beta_preview_box = w.HTML()
        self.beta_style_box = w.VBox([self.beta_style, self.beta_style_legend])
        self.beta_colors_box = w.VBox([w.HBox([self.beta_min, self.beta_max, self.beta_detect]), self.beta_colors,
                                       self.beta_labels, self.beta_title], layout=w.Layout(display="none"))
        self.beta_mode.observe(self._beta_mode_changed, names="value")
        self.beta_style.observe(self._show_style, names="value")
        self.beta_read.on_click(self._read_asset)
        self.beta_detect.on_click(self._detect_values)
        self.beta_preview.on_click(self._beta_preview)
        self.beta_save.on_click(self._save_beta)
        note = ("Os mapas salvos ficam no seu Drive e voltam nas próximas sessões." if self.library.persistent else
                "<span style='color:#a66f00'>Sem Drive conectado: o mapa salvo vale só nesta sessão.</span>")
        return w.VBox([
            w.HTML("<h4 style='margin:8px 0 2px'>Novo mapa</h4><p style='margin:0'>Cole o asset de uma imagem "
                   "(uma banda por ano) ou de uma coleção (uma imagem por ano). Os anos são detectados sozinhos. "
                   f"{note}</p>"),
            w.HBox([self.beta_asset, self.beta_read]), self.beta_info,
            self.beta_name, w.HBox([self.beta_theme, self.beta_collection]),
            w.HTML("<b>Cores e legenda</b>"), self.beta_mode, self.beta_style_box, self.beta_colors_box,
            w.HBox([self.beta_preview, self.beta_save]), self.beta_msg, self.beta_preview_box,
        ], layout=w.Layout(display="none", border="1px solid #ddd", padding="8px", margin="6px 0"))

    def _beta_mode_changed(self, *_):
        catalog_style = self.beta_mode.value == "Estilo do catálogo"
        self.beta_style_box.layout.display = "" if catalog_style else "none"
        self.beta_colors_box.layout.display = "none" if catalog_style else ""

    def _toggle_beta_panel(self, change):
        self.beta_panel.layout.display = "" if change["new"] else "none"
        if change["new"] and not self.beta_style.options:
            choices = analyst.style_choices()
            self.beta_style.options = choices
            keys = [key for _, key in choices]
            self.beta_style.value = "fire" if "fire" in keys else (keys[0] if keys else None)
            self._show_style()

    def _show_style(self, *_):
        if not self.beta_style.value:
            return
        _, legend = analyst.style_params(self.beta_style.value)
        self.beta_style_legend.value = _swatches(legend, 16)

    def _beta_layer(self):
        if self.beta_mode.value == "Estilo do catálogo":
            if not self.beta_style.value:
                raise ValueError("Escolha um estilo do catálogo ou use 'Minhas cores'.")
            return analyst.build_layer(self.beta_asset.value, self.beta_name.value, self.beta_theme.value,
                                       self.beta_collection.value, style_key=self.beta_style.value)
        return analyst.build_layer(self.beta_asset.value, self.beta_name.value, self.beta_theme.value,
                                   self.beta_collection.value, minimum=self.beta_min.value, maximum=self.beta_max.value,
                                   colors=self.beta_colors.value, labels=self.beta_labels.value,
                                   legend_title=self.beta_title.value)

    def _in_background(self, target, message_widget, busy):
        message_widget.value = f"<i>{busy}</i>"

        def work():
            try:
                target()
            except Exception as exc:
                message_widget.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    def _read_asset(self, _=None):
        def work():
            info = analyst.inspect_asset(self.beta_asset.value)
            years = f"{info['years'][0]}–{info['years'][1]}" if info["years"] else "sem anos no nome das bandas"
            if info["axis"] == "depth":
                years = "sequência por profundidade"
            sample = ", ".join(info["bands"][:3]) + (" …" if len(info["bands"]) > 3 else "")
            self.beta_info.value = (f"✓ {'Imagem' if info['kind'] == 'IMAGE' else 'Coleção'} com "
                                    f"{len(info['bands'])} período(s) · {years}<br>"
                                    f"<span style='font-size:12px;color:#555'>{escape(sample)}</span>")
            if not self.beta_name.value:
                self.beta_name.value = self.beta_asset.value.rstrip("/").split("/")[-1].replace("_", " ")
        self._in_background(work, self.beta_info, "Lendo o asset…")

    def _detect_values(self, _=None):
        def work():
            low, high = analyst.detect_range(self.beta_asset.value)
            self.beta_min.value, self.beta_max.value = low, high
            self.beta_msg.value = f"Valores encontrados no último período: {low} a {high}."
        self._in_background(work, self.beta_msg, "Calculando valores…")

    def _beta_preview(self, _=None):
        def work():
            layer = self._beta_layer()
            territory = self.selected_territories[0] if self.selected_territories else None
            url, label = preview.layer_thumbnail(_ee(), layer, territory)
            where = escape(territory["name"]) if territory else "área do asset"
            self.beta_preview_box.value = (f"<div style='font-size:12px;color:#555'>{escape(label)} · {where}</div>"
                                           f"<img src='{escape(url)}' style='max-width:560px;border:1px solid #ddd'>"
                                           f"<div style='margin-top:4px'>{_swatches(layer['legend'], 12)}</div>")
            self.beta_msg.value = ""
        self._in_background(work, self.beta_msg, "Gerando prévia…")

    def _save_beta(self, _=None):
        try:
            layer = self._beta_layer()
            validate([layer], [])
            self.library.save_map(layer)
            self._add_custom_layer(layer)
            self.beta_msg.value = (f"✓ “{escape(layer['name'])}” salvo em Meus mapas e escolhido. "
                                   "Ele já pode ser gerado como qualquer outro mapa.")
        except Exception as exc:
            self.beta_msg.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"

    def _add_custom_layer(self, layer):
        layer = dict(layer, initiative=MY_MAPS)
        self.catalog = [item for item in self.catalog if item["id"] != layer["id"]] + [layer]
        self.by_id[layer["id"]] = layer
        self.product_buttons.pop(layer["id"], None)
        self.year_ranges.pop(layer["id"], None)
        if MY_MAPS not in self.initiative.options:
            self.initiative.options = list(self.initiative.options) + [MY_MAPS]
        if layer["id"] not in self.selected_layers:
            self.selected_layers.append(layer["id"])
        self.initiative.value = MY_MAPS
        self._refresh_filters()
        self._render_products()
        self._render_my_maps()
        self._toggle_layer(layer["id"], True)

    def _render_my_maps(self, *_):
        if self.initiative.value != MY_MAPS:
            self.my_maps_panel.children = []
            return
        rows = [w.HTML("<b>Gerenciar meus mapas</b>")]
        for layer in [item for item in self.catalog if item["initiative"] == MY_MAPS]:
            remove = w.Button(description="Remover", icon="trash", layout=w.Layout(width="110px"))
            propose = w.Button(description="Proposta para o catálogo", icon="share", layout=w.Layout(width="230px"))
            remove.on_click(lambda _b, i=layer["id"]: self._remove_custom_layer(i))
            propose.on_click(lambda _b, i=layer["id"]: self._show_proposal(i))
            rows.append(w.HBox([w.HTML(f"<span style='line-height:28px'>{escape(layer['name'])}</span>",
                                       layout=w.Layout(width="45%")), propose, remove]))
        self.my_maps_panel.children = rows + [self.proposal] if len(rows) > 1 else []

    def _show_proposal(self, layer_id):
        self.proposal.value = analyst.catalog_snippet(self.by_id[layer_id])
        self.proposal.layout.display = ""

    def _remove_custom_layer(self, layer_id):
        self.library.remove_map(layer_id)
        self.catalog = [item for item in self.catalog if item["id"] != layer_id]
        self.by_id.pop(layer_id, None)
        self.product_buttons.pop(layer_id, None)
        if layer_id in self.selected_layers:
            self.selected_layers.remove(layer_id)
        self.proposal.layout.display = "none"
        if not any(item["initiative"] == MY_MAPS for item in self.catalog):
            self.initiative.value = self.initiative.options[0]
            self.initiative.options = [o for o in self.initiative.options if o != MY_MAPS]
        self._refresh_filters()
        self._render_products()
        self._render_my_maps()
        self._update_summary()

    def _visible_catalog(self):
        return [layer for layer in self.catalog if layer["initiative"] == self.initiative.value
                and (self.show_beta.value or layer.get("status") != "beta")]

    def _refresh_filters(self, *_):
        items = self._visible_catalog()
        lang = "pt"
        themes = list(dict.fromkeys(layer.get("theme") or "outros" for layer in items))
        self.theme.options = [("Todos", "")] + [(theme_label(t, lang), t) for t in themes]
        if self.theme.value not in ("",) + tuple(themes):
            self.theme.value = ""
        collections = sorted({str(layer.get("collection") or "") for layer in items} - {""},
                             key=lambda c: (not c.replace(".", "").isdigit(), [int(x) if x.isdigit() else 0 for x in c.split(".")], c))
        self.collection.options = [("Todas", "")] + [(f"Coleção {c}" if c[:1].isdigit() else c, c) for c in collections]
        if self.collection.value not in ("",) + tuple(collections):
            self.collection.value = ""

    def _render_products(self, *_):
        term = territories.plain(self.map_search.value)
        buttons = []
        for layer in self._visible_catalog():
            if self.theme.value and (layer.get("theme") or "outros") != self.theme.value:
                continue
            if self.collection.value and str(layer.get("collection")) != self.collection.value:
                continue
            haystack = territories.plain(f"{layer['name']} {layer.get('collection', '')} {layer.get('theme', '')}")
            if term and not all(part in haystack for part in term.split()):
                continue
            button = self.product_buttons.get(layer["id"])
            if button is None:
                suffix = " · beta" if layer.get("status") == "beta" else ""
                col = layer.get("collection")
                label = f"{layer['name']}" + (f"  (col. {col})" if col and str(col)[:1].isdigit() else "") + suffix
                button = w.ToggleButton(description=label, tooltip=layer.get("reason") or layer["name"],
                                        layout=w.Layout(width="98%", height="32px", min_height="32px", flex="0 0 auto"))
                button.observe(lambda change, i=layer["id"]: self._toggle_layer(i, change["new"]), names="value")
                self.product_buttons[layer["id"]] = button
            button.value = layer["id"] in self.selected_layers
            buttons.append(button)
        if not buttons:
            buttons = [w.HTML("<i>Nenhum mapa com esses filtros.</i>")]
        self.product_list.children = buttons

    def _toggle_layer(self, layer_id, on):
        if on and layer_id not in self.selected_layers:
            self.selected_layers.append(layer_id)
        if not on and layer_id in self.selected_layers:
            self.selected_layers.remove(layer_id)
        if not on:
            # Mostra a legenda do último mapa ainda escolhido (ou a dica inicial).
            if self.selected_layers:
                return self._toggle_layer(self.selected_layers[-1], True)
            self.product_detail.value = _card("<i>Clique em um mapa para ver a legenda.</i>").value
        layer = self.by_id[layer_id]
        if on:
            col = layer.get("collection")
            info = (f"<b>{escape(layer['name'])}</b><br><span style='font-size:12px;color:#555'>"
                    f"{escape(layer['initiative'])} · {escape(theme_label(layer.get('theme') or 'outros', 'pt'))}"
                    f"{' · Coleção ' + escape(str(col)) if col else ''}</span>")
            if layer.get("status") == "beta":
                info += f"<br><span style='color:#a66f00;font-size:12px'>Beta: {escape(layer.get('reason', ''))}</span>"
            if layer.get("legend_label"):
                info += f"<div style='margin-top:6px;font-size:12px'><b>{escape(layer['legend_label'])}</b></div>"
            self.product_detail.value = _card(info + _swatches(layer.get("legend"))).value
        self._update_summary()

    # ------------------------------------------------------------------ 2. território
    def _build_territories(self):
        self.base = w.Dropdown(options=TERRITORY_BASES, value="brasil", description="Base:")
        self.mode = w.ToggleButtons(options=["Buscar", "Desenhar no mapa", "Enviar arquivo", "Asset do GEE"],
                                    style={"button_width": "150px"})
        # Buscar
        self.kind = w.Dropdown(description="Tipo:")
        self.place_search = w.Text(placeholder="Digite parte do nome (ex.: são félix, yanomami, chapada)",
                                   continuous_update=True, layout=w.Layout(width="420px"))
        self.place_results = w.VBox(layout=w.Layout(max_height="240px", overflow_y="auto"))
        self.place_combine = w.Checkbox(value=False, description="Juntar os itens marcados em um único recorte", indent=False)
        self.place_add = w.Button(description="Adicionar", icon="plus", button_style="success")
        self.place_forget = w.Button(description="Apagar dos meus territórios", icon="trash",
                                     layout=w.Layout(width="240px", display="none"))
        self.place_forget.on_click(self._forget_saved)
        self.place_msg = w.HTML()
        self._found = []
        self._marked = []
        self._search_thread = None
        search_box = w.VBox([self.kind, self.place_search, self.place_results, self.place_combine,
                             w.HBox([self.place_add, self.place_forget]), self.place_msg])
        # Desenhar
        self.draw_name = w.Text(placeholder="Nome da área (ex.: Entorno do Parque X)", description="Nome:")
        self.draw_add = w.Button(description="Adicionar área desenhada", icon="plus", button_style="success")
        self.draw_msg = w.HTML("Use os botões de retângulo ou polígono no canto do mapa. "
                               "Vários desenhos viram uma única área.")
        draw_box = w.VBox([self.draw_msg, self.draw_name, self.draw_add])
        # Arquivo
        self.upload = w.FileUpload(accept=".geojson,.json,.kml,.zip,.gpkg", multiple=False,
                                   description="Escolher arquivo")
        self.upload_name = w.Text(placeholder="Nome da área (opcional)", description="Nome:")
        self.upload_add = w.Button(description="Adicionar arquivo", icon="plus", button_style="success")
        self.upload_msg = w.HTML("GeoJSON, KML, GeoPackage ou shapefile compactado em .zip. "
                                 "Todas as feições do arquivo viram uma única área.")
        upload_box = w.VBox([self.upload_msg, self.upload, self.upload_name, self.upload_add])
        # Asset
        self.asset_id = w.Text(placeholder="projects/meu-projeto/assets/minha_area", description="Asset:",
                               layout=w.Layout(width="520px"))
        self.asset_field = w.Text(placeholder="opcional, ex.: NM_MUN", description="Campo:")
        self.asset_value = w.Text(placeholder="opcional, valores separados por vírgula", description="Valor:")
        self.asset_name = w.Text(placeholder="Nome da área (opcional)", description="Nome:")
        self.asset_add = w.Button(description="Adicionar asset", icon="plus", button_style="success")
        self.asset_msg = w.HTML("A conta conectada precisa ter acesso de leitura ao asset (FeatureCollection).")
        asset_box = w.VBox([self.asset_msg, self.asset_id, self.asset_field, self.asset_value,
                            self.asset_name, self.asset_add])
        self.mode_boxes = {"Buscar": search_box, "Desenhar no mapa": draw_box,
                           "Enviar arquivo": upload_box, "Asset do GEE": asset_box}
        self.chosen = w.VBox()
        self.map, self.draw = self._make_map()
        self.base.observe(self._refresh_kinds, names="value")
        self.kind.observe(self._start_search, names="value")
        self.place_search.observe(self._start_search, names="value")
        self.place_add.on_click(self._add_found)
        self.draw_add.on_click(self._add_drawing)
        self.upload_add.on_click(self._add_upload)
        self.asset_add.on_click(self._add_asset)
        self.mode.observe(self._show_mode, names="value")
        self._refresh_kinds()
        self._show_mode()
        left = w.VBox([self.base, self.mode, *self.mode_boxes.values(),
                       w.HTML("<b>Territórios escolhidos</b>"), self.chosen], layout=w.Layout(width="48%"))
        right = w.VBox([self.map if self.map is not None else
                        w.HTML("<i>Mapa indisponível (ipyleaflet não instalado). A busca, o arquivo e o asset "
                               "continuam funcionando.</i>")], layout=w.Layout(width="50%"))
        self._render_chosen()
        return w.VBox([w.HTML("<h3>Onde?</h3><p>Escolha um ou mais territórios. "
                              "Cada mapa será gerado para cada território.</p>"), w.HBox([left, right])])

    def _make_map(self):
        try:
            import ipyleaflet
        except ImportError:
            return None, None
        m = ipyleaflet.Map(center=(-14, -52), zoom=4, scroll_wheel_zoom=True,
                           layout=w.Layout(height="460px", width="98%"))
        draw = ipyleaflet.DrawControl(polyline={}, circlemarker={}, marker={}, circle={},
                                      rectangle={"shapeOptions": {"color": "#1565c0"}},
                                      polygon={"shapeOptions": {"color": "#1565c0"}})
        m.add(draw)
        return m, draw

    def _show_mode(self, *_):
        for name, box in self.mode_boxes.items():
            box.layout.display = "" if name == self.mode.value else "none"

    def _refresh_kinds(self, *_):
        self.place_forget.layout.display = "" if self.base.value == "meus" else "none"
        self.place_combine.layout.display = "none" if self.base.value == "meus" else ""
        if self.base.value == "meus":
            self.kind.options = [("Salvos por você", 0)]
            self.kind.value = 0
            if self._territories_started:
                self._start_search()
            return
        sources = territories.sources_for(self.base.value)
        self.kind.options = [(s["label"], i) for i, s in enumerate(sources)]
        self.kind.value = 1 if len(sources) > 1 and self.base.value == "brasil" else 0
        if self._territories_started:
            self._start_search()

    def _source(self):
        return territories.sources_for(self.base.value)[self.kind.value]

    def _start_search(self, *_):
        if self.kind.value is None or not self._territories_started:
            return
        if self.base.value == "meus":
            term = territories.plain(self.place_search.value)
            self.place_search.disabled = False
            self._found = [{"saved": True, "name": t["name"], "territory": t} for t in self.library.territories
                           if not term or term in territories.plain(t["name"])]
            self._marked = []
            self.place_msg.value = "" if self._found else ("Nenhum território salvo ainda. Áreas desenhadas, "
                                                           "arquivos e assets adicionados ficam guardados aqui.")
            self._render_found()
            return
        source = self._source()
        if source.get("whole"):
            self._found = [{"whole": True, "name": source["label"], "source": source}]
            self._marked = []
            self._render_found()
            self.place_search.disabled = True
            return
        self.place_search.disabled = False
        text, kind = self.place_search.value, self.kind.value
        self.place_msg.value = "<i>Buscando…</i>"

        def work():
            try:
                found = territories.search(_ee(), source, text)
                if (text, kind) != (self.place_search.value, self.kind.value):
                    return  # o usuário já digitou outra coisa
                self._found, self._marked = found, []
                self.place_msg.value = "" if found else "Nenhum resultado."
            except Exception as exc:
                self._found = []
                self.place_msg.value = f"<span style='color:#c62828'>Não foi possível buscar: {escape(str(exc))}</span>"
            self._render_found()
        thread = Thread(target=work, daemon=True)
        self._search_thread = thread
        thread.start()

    def _render_found(self):
        buttons = []
        for i, record in enumerate(self._found):
            button = w.ToggleButton(value=False, description=record["name"], layout=w.Layout(width="98%", min_height="30px", flex="0 0 auto"))
            button.observe(lambda change, i=i: self._mark(i, change["new"]), names="value")
            buttons.append(button)
        self.place_results.children = buttons

    def _mark(self, index, on):
        record = self._found[index]
        if on and record not in self._marked:
            self._marked.append(record)
        if not on and record in self._marked:
            self._marked.remove(record)

    def _add_territory(self, territory):
        if any(t["id"] == territory["id"] for t in self.selected_territories):
            return False
        self.selected_territories.append(territory)
        self._render_chosen()
        self._draw_on_map(territory)
        self._update_summary()
        return True

    def _add_found(self, _=None):
        try:
            marked = list(self._marked) or ([self._found[0]] if len(self._found) == 1 else [])
            if not marked:
                raise ValueError("Marque pelo menos um item da lista.")
            if marked[0].get("saved"):
                items = [record["territory"] for record in marked]
            elif marked[0].get("whole"):
                items = [territories.whole(marked[0]["source"])]
            else:
                items = territories.from_records(marked, combine=self.place_combine.value)
            added = sum(self._add_territory(item) for item in items)
            self.place_msg.value = f"✓ {added} território(s) adicionado(s)."
            for button in self.place_results.children:
                button.value = False
        except Exception as exc:
            self.place_msg.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"

    def _keep(self, territory):
        """Adiciona à seleção e guarda em 'Meus territórios' para as próximas sessões."""
        self._add_territory(territory)
        try:
            self.library.save_territory(territory)
        except OSError as exc:
            self.place_msg.value = f"<span style='color:#a66f00'>Não foi possível guardar o território: {escape(str(exc))}</span>"

    def _forget_saved(self, _=None):
        marked = [record for record in self._marked if record.get("saved")]
        if not marked:
            self.place_msg.value = "Marque os territórios salvos que quer apagar."
            return
        for record in marked:
            self.library.remove_territory(record["territory"]["id"])
        self.place_msg.value = f"✓ {len(marked)} território(s) apagado(s) dos seus salvos."
        self._start_search()

    def _add_drawing(self, _=None):
        try:
            features = list(getattr(self.draw, "data", None) or [])
            geometry = territories.merge_drawings(features)
            self._keep(territories.from_geometry(geometry, self.draw_name.value or "Área desenhada"))
            if self.draw is not None:
                self.draw.clear()
            self.draw_name.value = ""
            self.draw_msg.value = "✓ Área adicionada. Desenhe outra se quiser."
        except Exception as exc:
            self.draw_msg.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"

    def _add_upload(self, _=None):
        try:
            files = self.upload.value
            files = list(files.values()) if isinstance(files, dict) else list(files)
            if not files:
                raise ValueError("Escolha um arquivo primeiro.")
            item = files[0]
            filename = item.get("name") or item.get("metadata", {}).get("name")
            geometry = territories.read_upload(filename, item["content"])
            name = self.upload_name.value or Path(filename).stem
            self._keep(territories.from_geometry(geometry, name))
            self.upload_msg.value = f"✓ {escape(name)} adicionado."
        except Exception as exc:
            self.upload_msg.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"

    def _add_asset(self, _=None):
        try:
            item = territories.from_asset(self.asset_id.value, self.asset_name.value,
                                          self.asset_field.value, self.asset_value.value)
            size = territories.feature_collection(_ee(), item).size().getInfo()
            if not size:
                raise ValueError("O asset (ou o filtro) não retornou nenhuma feição.")
            self._keep(item)
            self.asset_msg.value = f"✓ {escape(item['name'])} adicionado ({size} feição(ões))."
        except Exception as exc:
            self.asset_msg.value = f"<span style='color:#c62828'>Não foi possível usar o asset: {escape(str(exc))}</span>"

    def _remove_territory(self, territory_id):
        self.selected_territories = [t for t in self.selected_territories if t["id"] != territory_id]
        if self.map is not None:
            for layer in list(self.map.layers):
                if getattr(layer, "name", "") == f"territorio:{territory_id}":
                    self.map.remove(layer)
        self._render_chosen()
        self._update_summary()

    def _render_chosen(self):
        rows = []
        for territory in self.selected_territories:
            remove = w.Button(icon="times", tooltip="Remover", layout=w.Layout(width="36px"))
            remove.on_click(lambda _b, i=territory["id"]: self._remove_territory(i))
            rows.append(w.HBox([remove, w.HTML(f"<span style='line-height:28px'>{escape(territory['name'])}</span>")]))
        self.chosen.children = rows or [w.HTML("<i style='color:#999'>Nenhum ainda.</i>")]

    def _draw_on_map(self, territory):
        if self.map is None:
            return

        def work():
            try:
                import ipyleaflet
                ee = _ee()
                url = preview.outline_url(ee, territory)
                self.map.add(ipyleaflet.TileLayer(url=url, name=f"territorio:{territory['id']}",
                                                  attribution="Google Earth Engine"))
                # fit_bounds depende do loop assíncrono do kernel; centro e zoom funcionam de qualquer thread.
                self.map.center, self.map.zoom = preview.center_zoom(preview.bounds(ee, territory))
            except Exception as exc:
                self.place_msg.value = f"<span style='color:#c62828'>Não foi possível mostrar no mapa: {escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 3. período e saídas
    def _build_outputs(self):
        self.all_years = w.Checkbox(value=True, description="Todos os anos disponíveis", indent=False)
        self.period = w.IntRangeSlider(value=(1985, 2025), min=1985, max=2025, step=1, description="Anos:",
                                       continuous_update=False, layout=w.Layout(width="520px", display="none"))
        self.period_msg = w.HTML()
        self.kind_checks = {key: w.Checkbox(value=default, description=label, indent=False)
                            for key, label, default in KINDS}
        for key, label in ADVANCED_KINDS:
            self.kind_checks[key] = w.Checkbox(value=False, description=label, indent=False)
        self.advanced = w.ToggleButton(description="Opções avançadas", icon="cog", value=False)
        self.save_drive = w.Checkbox(value=drive_mounted(), description="Salvar no meu Google Drive", indent=False)
        self.drive_folder = w.Text(value=DEFAULT_DRIVE_FOLDER, description="Pasta:", layout=w.Layout(width="520px"))
        self.drive_msg = w.HTML()
        advanced_box = w.VBox([*(self.kind_checks[key] for key, _ in ADVANCED_KINDS), self.drive_folder],
                              layout=w.Layout(display="none"))
        self.advanced.observe(lambda c: setattr(advanced_box.layout, "display", "" if c["new"] else "none"), names="value")
        self.preview_button = w.Button(description="Ver prévia", icon="eye")
        self.preview_box = w.HTML()
        self.preview_button.on_click(self._preview)
        self.all_years.observe(self._period_mode, names="value")
        for widget in [self.all_years, self.period, self.save_drive, *self.kind_checks.values()]:
            widget.observe(lambda _c: self._update_summary(), names="value")
        self.save_drive.observe(self._drive_note, names="value")
        self._drive_note()
        return w.VBox([
            w.HTML("<h3>Quando e como?</h3>"),
            w.HTML("<b>Período</b>"), self.all_years, self.period, self.period_msg,
            w.HTML("<b>Arquivos</b>"), *(self.kind_checks[key] for key, _, _ in KINDS),
            w.HTML("<b>Destino</b>"), self.save_drive, self.drive_msg, self.advanced, advanced_box,
            w.HTML("<b>Prévia</b> <span style='color:#777;font-size:12px'>(primeiro mapa e primeiro território, "
                   "último ano escolhido)</span>"), self.preview_button, self.preview_box])

    def _period_mode(self, *_):
        self.period.layout.display = "none" if self.all_years.value else ""

    def _drive_note(self, *_):
        if self.save_drive.value and not drive_mounted():
            self.drive_msg.value = ("<span style='color:#c62828'>O Drive não está conectado. Rode a célula 1 "
                                    "de novo ou desmarque esta opção (os arquivos ficam só nesta sessão).</span>")
        elif self.save_drive.value:
            self.drive_msg.value = "Os arquivos ficam na sua pasta do Drive e o que já existir não é refeito."
        else:
            self.drive_msg.value = ("<span style='color:#a66f00'>Os arquivos ficam só nesta sessão do Colab "
                                    "e somem quando ela fechar. Baixe-os no passo 4.</span>")

    def _load_years(self):
        pending = [i for i in self.selected_layers if i not in self.year_ranges]
        if not pending:
            self._apply_years()
            return
        self.period_msg.value = "<i>Consultando os anos disponíveis…</i>"

        def work():
            for layer_id in pending:
                try:
                    self.year_ranges[layer_id] = preview.years(self.by_id[layer_id])
                except Exception as exc:
                    self.year_ranges[layer_id] = None
                    self.period_msg.value = (f"<span style='color:#c62828'>{escape(self.by_id[layer_id]['name'])}: "
                                             f"{escape(str(exc))}</span>")
                    return
            self._apply_years()
        Thread(target=work, daemon=True).start()

    def _apply_years(self):
        ranges = [self.year_ranges.get(i) for i in self.selected_layers if self.year_ranges.get(i)]
        if not ranges:
            self.period_msg.value = ""
            return
        low, high = min(r[0] for r in ranges), max(r[1] for r in ranges)
        start, end = self.period.value
        if self.all_years.value:
            start, end = low, high
        start, end = max(low, min(start, high)), min(high, max(end, low))
        # Abre os limites antes de mover o valor, depois fecha nos anos disponíveis.
        self.period.max = max(self.period.max, high)
        self.period.min = min(self.period.min, low)
        self.period.value = (start, end)
        self.period.min, self.period.max = low, high
        self.period_msg.value = f"Disponível: {low}–{high}."

    def _layer_for_run(self, layer_id):
        layer = dict(self.by_id[layer_id])
        span = self.year_ranges.get(layer_id)
        if not self.all_years.value and span:
            start, end = self.period.value
            layer["start_year"], layer["end_year"] = max(start, span[0]), min(end, span[1])
            if layer["start_year"] > layer["end_year"]:
                return None
        return layer

    def _preview(self, _=None):
        if not self.selected_layers or not self.selected_territories:
            self.preview_box.value = "Escolha um mapa e um território primeiro."
            return
        self.preview_box.value = "<i>Gerando prévia…</i>"
        layer = self._layer_for_run(self.selected_layers[0]) or self.by_id[self.selected_layers[0]]
        territory = self.selected_territories[0]
        year = None if self.all_years.value else self.period.value[1]

        def work():
            try:
                url, label = preview.thumbnail_url(_ee(), layer, territory, year)
                self.preview_box.value = (f"<div style='font-size:12px;color:#555'>{escape(layer['name'])} · "
                                          f"{escape(territory['name'])} · {escape(label)}</div>"
                                          f"<img src='{escape(url)}' style='max-width:640px;border:1px solid #ddd'>"
                                          f"<div style='font-size:11px;color:#777'>Prévia rápida, sem moldura, "
                                          f"legenda ou título; o arquivo final terá o layout completo.</div>")
            except Exception as exc:
                self.preview_box.value = f"<span style='color:#c62828'>Prévia indisponível: {escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 4. gerar
    def _build_generate(self):
        self.plan = w.HTML()
        self.check = w.Button(description="Verificar o que já existe", icon="search")
        self.check_out = w.HTML()
        self.run = w.Button(description="Gerar", icon="play", button_style="success",
                            layout=w.Layout(width="220px", height="40px"))
        self.results = w.VBox()
        self.check.on_click(self._check)
        self.run.on_click(lambda _: self.start(wait=False))
        return w.VBox([w.HTML("<h3>Conferir e gerar</h3>"), self.plan, w.HBox([self.run, self.check]),
                       self.check_out, self.runner.widget(), w.HTML("<b>Resultados</b>"), self.results])

    def combinations(self):
        combos = []
        for layer_id in self.selected_layers:
            layer = self._layer_for_run(layer_id)
            if layer is None:
                continue
            for territory in self.selected_territories:
                combos.append((layer["initiative"], layer, dict(territory)))
        return combos

    def kinds(self):
        return {key for key, check in self.kind_checks.items() if check.value}

    def _render_plan(self):
        combos = self.combinations()
        if not combos:
            self.plan.value = ("<span style='color:#c62828'>Falta escolher pelo menos um mapa (passo 1) "
                               "e um território (passo 2).</span>")
            return
        rows = "".join(f"<tr><td>{escape(layer['name'])}</td><td>{escape(territory['name'])}</td></tr>"
                       for _, layer, territory in combos[:30])
        more = f"<p>… e mais {len(combos) - 30}.</p>" if len(combos) > 30 else ""
        self.plan.value = (f"<p>{len(combos)} combinação(ões). Cada uma leva de 1 a alguns minutos, "
                           f"conforme o tamanho do território e o número de anos.</p>"
                           f"<table style='font-size:13px'><tr><th align=left>Mapa</th><th align=left>Território</th></tr>"
                           f"{rows}</table>{more}")

    def _check(self, _=None):
        combos, kinds = self.combinations(), self.kinds()
        if not combos or not kinds:
            self.check_out.value = "Escolha mapa, território e tipo de arquivo."
            return
        self.check_out.value = "<i>Verificando…</i>"

        def work():
            try:
                lines = self.runner.availability(combos, kinds, self.save_drive.value, self.drive_folder.value)
                self.check_out.value = "<br>".join(
                    f"{escape(state)} — {escape(layer['name'])} / {escape(territory['name'])}"
                    for (_, layer, territory), state in lines)
            except Exception as exc:
                self.check_out.value = f"<span style='color:#c62828'>{escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    def start(self, wait=True):
        """Gera todas as combinações escolhidas (também usado pela célula opcional do notebook)."""
        self.results.children = []
        return self.runner.start(self.combinations(), self.kinds(), self.save_drive.value,
                                 self.drive_folder.value, wait=wait)

    def _show_results(self, files, summary):
        files = [Path(f) for f in files]
        gifs = [f for f in files if f.suffix.lower() == ".gif"]
        items = []
        for f in gifs[:6]:
            try:
                items.append(w.VBox([w.Image(value=f.read_bytes(), format="gif", layout=w.Layout(max_width="420px")),
                                     w.HTML(f"<span style='font-size:11px'>{escape(f.name)}</span>")]))
            except OSError:
                pass
        listing = "".join(f"<li style='font-size:12px'>{escape(str(f))}</li>" for f in files[:200])
        download = self._download_button(files)
        self.results.children = [w.HBox(items[:3]), w.HBox(items[3:6]), *( [download] if download else []),
                                 w.HTML(f"<details><summary>{len(files)} arquivo(s)</summary><ul>{listing}</ul></details>")]

    def _download_button(self, files):
        try:
            from google.colab import files as colab_files
        except ImportError:
            return None
        button = w.Button(description="Baixar tudo (.zip)", icon="download")

        def download(_):
            import zipfile
            target = Path("/content/mapbiomas_gifs.zip")
            with zipfile.ZipFile(target, "w") as archive:
                for f in files:
                    if f.exists():
                        archive.write(f, f.name)
            colab_files.download(str(target))
        button.on_click(download)
        return button


def show_app(project, layers=(), **kwargs):
    """Abre a Fábrica. `layers` são mapas próprios (opcionais), mostrados em 'Meus mapas'."""
    return FactoryApp(project, layers, **kwargs)
