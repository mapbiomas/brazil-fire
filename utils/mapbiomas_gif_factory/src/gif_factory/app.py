"""Fábrica de GIFs MapBiomas: interface em 5 passos para o Google Colab.

1. Mapa        — escolher produtos do catálogo (busca, tema, coleção)
2. Território  — buscar nas bases oficiais, desenhar no mapa, enviar arquivo ou colar asset
3. Período e saídas — anos, tipos de arquivo, destino e prévia
4. Gerar       — conferir (com estimativa de tempo e volume), gerar em paralelo e acompanhar na vitrine
5. Biblioteca  — navegar no que já foi gerado (Drive ou sessão), ver, copiar caminho e baixar

Ninguém precisa editar código: tudo é escolhido na interface.
"""
from html import escape
from pathlib import Path
from threading import Thread

import ipywidgets as w
from IPython.display import display

from .i18n import LANGUAGES, N_, _, map_language, set_language, ui_text
from . import analyst, environment, estimate, preview, territories, theme
from .catalog import catalog_layers
from .engine import validate
from .gallery import Gallery, scan_library
from .legends import INITIATIVES, theme_label
from .runner import BatchRunner, drive_mounted
from .user_library import UserLibrary

STEPS = [N_("1 · Mapa"), N_("2 · Território"), N_("3 · Período e saídas"), N_("4 · Gerar"), N_("5 · Biblioteca")]
LABEL_WIDTH = "90px"
LABELED = (w.Text, w.Dropdown, w.FloatText, w.IntRangeSlider, w.Textarea)
KINDS = [("gif", N_("GIF animado"), True), ("colagem", N_("Colagem (anos lado a lado)"), False),
         ("finais", N_("Imagens de cada ano (mapa completo)"), False)]
ADVANCED_KINDS = [("brutos", N_("Imagens de cada ano sem moldura")),
                  ("preparados", N_("Imagens de cada ano com ano e escala"))]
DEFAULT_DRIVE_FOLDER = "/content/drive/MyDrive/MapBiomas_GIF_Factory"
TERRITORY_BASES = [("Brasil", "brasil"), ("Perú", "peru"), ("Paraguay", "paraguay"), ("RD Congo", "drc"),
                   (N_("Meus territórios"), "meus")]
MY_MAPS = N_("Meus mapas")
ROLES = [(N_("Recorte"), "recorte"), (N_("Linha"), "linha")]
LINE_COLORS = [(N_("Preto"), "000000"), (N_("Branco"), "ffffff"), (N_("Amarelo"), "f2c94c"),
               (N_("Terracota"), "c0643f"), (N_("Cinza"), "7a7a7a"), (N_("Azul"), "2f6fb0"), (N_("Vermelho"), "d32f2f")]
LINE_WIDTHS = [(N_("Fina"), 2), (N_("Média"), 3), (N_("Grossa"), 5)]
TERRITORY_MODES = [N_("Buscar"), N_("Desenhar no mapa"), N_("Enviar arquivo"), N_("Asset do GEE")]
BETA_MODES = [N_("Estilo do catálogo"), N_("Minhas cores")]


def _options(pairs):
    """[(rótulo, valor)] com o rótulo no idioma da interface (o valor não muda)."""
    return [(_(label), value) for label, value in pairs]


def _ee():
    import ee
    return ee


def _swatches(legend, limit=30):
    items = []
    for item in (legend or [])[:limit]:
        color = escape(str(item.get("color", "")).lstrip("#"))
        items.append(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px">'
                     f'<span style="width:14px;height:14px;border:1px solid var(--muted);background:#{color}"></span>'
                     f'{escape(str(ui_text(item.get("label", ""))))}</div>')
    more = (f'<div style="font-size:11px;color:var(--muted)">{_("+ {n} classes", n=len(legend) - limit)}</div>'
            if len(legend or []) > limit else "")
    return "".join(items) + more


def _is_number(text):
    try:
        float(str(text).replace(",", "."))
        return True
    except ValueError:
        return False


def _is_continuous(legend, viz):
    """Escala contínua (barra de cor) ou classes (quadradinhos)?

    Usa o `cmap_type` da visualização quando existe (sequential/diverging = contínua;
    categorical/binary = classes). Sem ele: contínua se a paleta tem 2+ cores, não há
    classes nomeadas e a legenda só tem números.
    """
    viz = viz or {}
    palette = viz.get("palette") or []
    if len(palette) < 2 or viz.get("classes"):
        return False
    kind = str(viz.get("cmap_type") or "").lower()
    if kind:
        return kind in {"sequential", "diverging", "continuous"}
    if viz.get("discrete_labels"):
        return False
    labels = [item.get("label", "") for item in legend or []]
    return bool(labels) and all(_is_number(label) for label in labels)


def _class_legend(legend, viz):
    """Classes numeradas sem nomes (ex.: 0, 1, 2) mostram uma cor por valor, não só os extremos."""
    viz = viz or {}
    palette = viz.get("palette") or []
    low, high = viz.get("min"), viz.get("max")
    labels = [item.get("label", "") for item in legend or []]
    if (len(labels) == 2 and all(_is_number(x) for x in labels) and _is_number(low) and _is_number(high)
            and float(low).is_integer() and float(high).is_integer()
            and int(float(high)) - int(float(low)) + 1 == len(palette)):
        return [{"label": str(int(float(low)) + i), "color": color} for i, color in enumerate(palette)]
    return legend


def _colorbar(viz, title=""):
    palette = ["#" + str(c).lstrip("#") for c in viz.get("palette") or []]
    low, high = viz.get("min", ""), viz.get("max", "")
    middle = ""
    if _is_number(low) and _is_number(high):
        mid = (float(low) + float(high)) / 2
        middle = f"{mid:g}"
    return (f'<div style="font-size:12px">'
            f'<div style="height:14px;border:1px solid var(--line);border-radius:4px;'
            f'background:linear-gradient(to right, {", ".join(escape(c) for c in palette)})"></div>'
            f'<div style="display:flex;justify-content:space-between;color:var(--muted);margin-top:2px">'
            f'<span>{escape(str(low))}</span><span>{escape(middle)}</span><span>{escape(str(high))}</span></div>'
            f'<div style="font-size:11px;color:var(--muted)">{_("escala contínua")}</div></div>')


def _layer_viz(layer):
    """Parâmetros de visualização da camada (próprios ou do catálogo), ou {}."""
    if isinstance(layer.get("visualization"), dict):
        return layer["visualization"]
    key = layer.get("visualization_key") or layer.get("style_key")
    if not key:
        return {}
    try:
        return analyst.style_params(key)[0] or {}
    except Exception:
        return {}


def _legend_html(legend, viz=None, limit=30):
    if _is_continuous(legend, viz):
        return _colorbar(viz)
    return _swatches(_class_legend(legend, viz), limit)


def _align_labels(widget):
    """Mesma largura de rótulo em todos os campos, para nada ficar cortado ou desalinhado."""
    if isinstance(widget, LABELED) and getattr(widget, "description", ""):
        widget.style.description_width = LABEL_WIDTH
    for child in getattr(widget, "children", ()) or ():
        _align_labels(child)


def _card(html):
    return w.HTML(f'<div class="gf-panel">{html}</div>')


class FactoryApp:
    def __init__(self, project, layers=(), territories=(), output_root="/content/mapbiomas_gif_factory_output",
                 temp_root="/content/temp", show=True, library=None, language=None, map_language=None):
        if language is not None or map_language is not None:
            set_language(ui=language, maps=map_language if map_language is not None else language)
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
        self.output_root = Path(output_root)
        self.environment = environment.detect()
        self.runner = BatchRunner(output_root, temp_root, project)
        self.runner.workers = self.environment["workers"]
        self.gallery = Gallery(zip_root=Path(temp_root) / "downloads",
                               empty=_("Os cartões aparecem aqui assim que o lote começar."))
        self.runner.on_start = self._gallery_start
        self.runner.on_combo = self._gallery_update
        self.runner.on_finished = self._show_results
        self.summary = w.HTML()
        self.stepper = theme.classed(w.ToggleButtons(options=[(_(step), step) for step in STEPS], value=STEPS[0],
                                                     style={"button_width": "auto"}),
                                     "gf-steps")
        self.back = w.Button(description=_("Voltar"), icon="arrow-left")
        self.next = w.Button(description=_("Avançar"), icon="arrow-right", button_style="primary",
                             layout=w.Layout(width="auto"))
        self.pages = [self._build_maps(), self._build_territories(), self._build_outputs(), self._build_generate(),
                      self._build_library()]
        self.stepper.observe(self._go, names="value")
        self.back.on_click(lambda _b: self._move(-1))
        self.next.on_click(lambda _b: self._move(1))
        title = theme.classed(w.HTML(f"<h2>🌱 {_('Fábrica de GIFs MapBiomas')}</h2>"
                                     f"<p class='gf-sub'>{_('Escolha o mapa, o território e o período. Nada aqui exige programação.')}</p>"),
                              "gf-title")
        self.root = theme.classed(w.VBox([theme.style_widget(), title, self.stepper, self.summary, *self.pages,
                                          w.HBox([self.back, self.next], layout=w.Layout(margin="10px 0 0"))]),
                                  "gf-root")
        _align_labels(self.root)
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

    def _go(self, *_args):
        index = STEPS.index(self.stepper.value)
        for i, page in enumerate(self.pages):
            page.layout.display = "" if i == index else "none"
        self.back.disabled = index == 0
        last = index == len(STEPS) - 1
        self.next.layout.display = "none" if last else ""
        if not last:
            self.next.description = _("Avançar: {passo}", passo=_(STEPS[index + 1]).split("·", 1)[-1].strip())
        if index == 1 and not self._territories_started:
            self._territories_started = True
            self._start_search()
        if index in (2, 3):
            self._load_years()
        if index == 3:
            self._render_plan()
        if index == 4:
            self._scan_library()

    def _update_summary(self):
        maps = [ui_text(self.by_id[i]["name"]) for i in self.selected_layers]
        places = [t["name"] for t in self.clips()]
        color_name = {v: _(k).lower() for k, v in LINE_COLORS}
        lines = [_("{nome} em {cor}", nome=t["name"], cor=color_name.get(t.get("color"), _("cor própria")))
                 for t in self.lines()]
        kinds = [_(label) for key, label, _x in KINDS if self.kind_checks[key].value] + \
                [_(label) for key, label in ADVANCED_KINDS if self.kind_checks[key].value]

        def fmt(items):
            if not items:
                return f"<i style='color:var(--muted)'>{_('nenhum')}</i>"
            text = ", ".join(escape(i) for i in items[:3])
            return text + (_(" e mais {n}", n=len(items) - 3) if len(items) > 3 else "")
        period = (_("todos os anos") if self.all_years.value
                  else f"{self.period.value[0]}–{self.period.value[1]}")
        total = len(maps) * len(places)
        guess = ""
        if total and self.kinds():
            info = self.estimate()
            emoji = estimate.mood(info["seconds"])[0]
            guess = (f" &nbsp;·&nbsp; {emoji} {estimate.about(info['seconds'])}, "
                     f"~{estimate.human_size(info['deliver_mb'])}")
        self.summary.value = (
            "<div style='border-left:4px solid var(--green);padding:6px 12px;margin:8px 0;background:var(--surface);"
            "border-radius:0 10px 10px 0;box-shadow:0 1px 3px var(--shadow);font-size:13px'>"
            f"<b>{_('Mapas:')}</b> {fmt(maps)} &nbsp;·&nbsp; <b>{_('Territórios:')}</b> {fmt(places)} &nbsp;·&nbsp; "
            + (f"<b>{_('Linhas:')}</b> {fmt(lines)} &nbsp;·&nbsp; " if lines else "") +
            f"<b>{_('Período:')}</b> {period} &nbsp;·&nbsp; <b>{_('Arquivos:')}</b> {fmt(kinds)}"
            f" &nbsp;·&nbsp; <b>{_('{n} combinação(ões)', n=total)}</b>{guess}</div>")

    # ------------------------------------------------------------------ estimativa
    def _frames_for(self, layer_id):
        span = self.year_ranges.get(layer_id)
        if not span:
            return None
        start, end = span
        if not self.all_years.value:
            start, end = max(start, self.period.value[0]), min(end, self.period.value[1])
        return max(0, end - start + 1)

    def estimate(self):
        """Tempo e volume previstos para a seleção atual (ver `estimate.batch`)."""
        frames = [self._frames_for(layer_id) for layer_id in self.selected_layers
                  for _x in self.clips()]
        return estimate.batch(frames, self.kinds(), self.runner.workers, self.runner.calibration.per_frame(),
                              to_drive=self.save_drive.value)

    def _estimate_html(self):
        info = self.estimate()
        emoji, phrase, level = estimate.mood(info["seconds"])
        note, size_level = estimate.size_note(info["deliver_mb"], self.save_drive.value)
        colors = {0: ("var(--green)", "var(--green-soft)"), 1: ("var(--warn)", "var(--warn-soft)"), 2: ("var(--danger)", "var(--danger-soft)")}
        border, background = colors[max(level, size_level)]
        where = _("no seu Drive") if self.save_drive.value else _("nesta sessão do Colab")
        basis = (_("calculada com o ritmo real desta sessão") if info["calibrated"] else
                 _("estimativa média; fica mais precisa depois da primeira combinação"))
        together = _(", {n} ao mesmo tempo", n=self.runner.workers) if self.runner.workers > 1 else ""
        size_line = _("📦 ~{final} de arquivos finais {onde} · ~{trabalho} de arquivos de trabalho temporários",
                      final=estimate.human_size(info["deliver_mb"]), onde=where,
                      trabalho=estimate.human_size(info["work_mb"]))
        return (f"<div style='border-left:4px solid {border};background:{background};padding:6px 10px;margin:6px 0'>"
                f"<div style='font-size:15px'>{emoji} <b>{estimate.about(info['seconds'])}</b> "
                f"{_('para {n} quadro(s)', n=info['frames'])}{together}</div>"
                f"<div>{escape(phrase)}</div>"
                f"<div style='font-size:12px;color:var(--muted);margin-top:4px'>{size_line}</div>"
                + (f"<div style='font-size:12px;color:{border}'>{escape(note)}</div>" if note else "")
                + f"<div style='font-size:11px;color:var(--muted)'>"
                f"({basis}; {_('o que já existir é pulado e termina antes')})</div></div>")

    # ------------------------------------------------------------------ 1. mapa
    def _build_maps(self):
        initiatives = list(dict.fromkeys(layer["initiative"] for layer in self.catalog))
        initiatives.sort(key=lambda name: (name != "Brasil", name == MY_MAPS, name))
        self.initiative = w.Dropdown(options=[(_(name) if name == MY_MAPS else ui_text(name), name) for name in initiatives],
                                     value=initiatives[0], description=_("Iniciativa:"))
        self.map_search = w.Text(placeholder=_("Buscar mapa (ex.: frequência, cobertura nível 2)"),
                                 continuous_update=True, layout=w.Layout(width="95%", max_width="420px"))
        self.theme = w.Dropdown(description=_("Tema:"))
        self.collection = w.Dropdown(description=_("Coleção:"))
        self.show_beta = w.Checkbox(value=False, description=_("Mostrar mapas beta (em revisão)"), indent=False)
        self.product_list = w.VBox(layout=w.Layout(max_height="360px", overflow="hidden auto", width="100%"))
        self.product_detail = _card(f"<i>{_('Clique em um mapa para ver a legenda.')}</i>")
        self.product_buttons = {}
        for widget in (self.initiative, self.theme, self.collection, self.show_beta):
            widget.observe(self._render_products, names="value")
        self.map_search.observe(self._render_products, names="value")
        self.initiative.observe(self._refresh_filters, names="value")
        self.show_beta.observe(self._refresh_filters, names="value")
        self._refresh_filters()
        self.new_map = w.ToggleButton(description=_("Novo mapa a partir de um asset"), icon="plus",
                                      tooltip=_("Para mapas beta ou fora do catálogo"), layout=w.Layout(width="280px"))
        self.beta_panel = self._build_beta_panel()
        self.my_maps_panel = w.VBox()
        self.proposal = w.Textarea(layout=w.Layout(width="98%", height="220px", display="none"))
        self.new_map.observe(self._toggle_beta_panel, names="value")
        self.initiative.observe(self._render_my_maps, names="value")
        left = w.VBox([self.initiative, self.map_search, w.HBox([self.theme, self.collection], layout=w.Layout(flex_flow="row wrap")), self.show_beta,
                       self.product_list, self.my_maps_panel], layout=w.Layout(width="62%", overflow="hidden"))
        right = w.VBox([theme.label(_("Legenda")), self.product_detail], layout=w.Layout(width="36%"))
        self._render_my_maps()
        return w.VBox([theme.section(_("Qual mapa?"), _("Escolha um ou mais mapas. "
                                                         "Eles serão gerados para cada território escolhido no passo 2.")),
                       w.HBox([left, right]), self.new_map, self.beta_panel])

    # ------------------------------------------------------------------ 1b. modo analista
    def _build_beta_panel(self):
        self.beta_asset = w.Text(placeholder=_("projects/.../assets/minha_imagem"), description=_("Asset:"),
                                 layout=w.Layout(width="560px"))
        self.beta_read = w.Button(description=_("Ler asset"), icon="search")
        self.beta_info = w.HTML()
        self.beta_name = w.Text(placeholder=_("Nome que aparece no título do mapa"), description=_("Nome:"),
                                layout=w.Layout(width="560px"))
        self.beta_theme = w.Dropdown(options=analyst.THEMES, value="fire", description=_("Tema:"))
        self.beta_collection = w.Text(value="beta", description=_("Coleção:"))
        self.beta_mode = theme.classed(w.ToggleButtons(options=[(_(m), m) for m in BETA_MODES],
                                                       style={"button_width": "160px"}), "gf-tabs")
        self.beta_style = w.Dropdown(description=_("Estilo:"), layout=w.Layout(width="560px"))
        self.beta_style_legend = w.HTML()
        self.beta_min = w.FloatText(value=0, description=_("Mínimo:"))
        self.beta_max = w.FloatText(value=1, description=_("Máximo:"))
        self.beta_detect = w.Button(description=_("Detectar valores"), icon="calculator")
        self.beta_colors = w.Text(placeholder=_("ex.: ffffff, ffcc00, 990000"), description=_("Cores:"),
                                  layout=w.Layout(width="560px"))
        self.beta_labels = w.Text(placeholder=_("opcional: um rótulo por cor (ex.: Baixo, Médio, Alto)"),
                                  description=_("Rótulos:"), layout=w.Layout(width="560px"))
        self.beta_title = w.Text(placeholder=_("opcional: título da legenda"), description=_("Legenda:"))
        self.beta_preview = w.Button(description=_("Ver prévia"), icon="eye")
        self.beta_save = w.Button(description=_("Salvar em Meus mapas"), icon="save", button_style="success",
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
        note = (_("Os mapas salvos ficam no seu Drive e voltam nas próximas sessões.") if self.library.persistent else
                f"<span style='color:var(--warn)'>{_('Sem Drive conectado: o mapa salvo vale só nesta sessão.')}</span>")
        intro = _("Cole o asset de uma imagem (uma banda por ano) ou de uma coleção (uma imagem por ano). "
                  "Os anos são detectados sozinhos.")
        return theme.classed(w.VBox([
            w.HTML(f"<h4 style='margin:8px 0 2px'>{_('Novo mapa')}</h4><p style='margin:0'>{intro} {note}</p>"),
            w.HBox([self.beta_asset, self.beta_read]), self.beta_info,
            self.beta_name, w.HBox([self.beta_theme, self.beta_collection]),
            theme.label(_("Cores e legenda")), self.beta_mode, self.beta_style_box, self.beta_colors_box,
            w.HBox([self.beta_preview, self.beta_save]), self.beta_msg, self.beta_preview_box,
        ], layout=w.Layout(display="none", margin="8px 0")), "gf-panel")

    def _beta_mode_changed(self, *_args):
        catalog_style = self.beta_mode.value == "Estilo do catálogo"
        self.beta_style_box.layout.display = "" if catalog_style else "none"
        self.beta_colors_box.layout.display = "none" if catalog_style else ""

    def _toggle_beta_panel(self, change):
        self.beta_panel.layout.display = "" if change["new"] else "none"
        if change["new"] and not self.beta_style.options:
            choices = analyst.style_choices()
            self.beta_style.options = choices
            keys = [key for _x, key in choices]
            self.beta_style.value = "fire" if "fire" in keys else (keys[0] if keys else None)
            self._show_style()

    def _show_style(self, *_args):
        if not self.beta_style.value:
            return
        viz, legend = analyst.style_params(self.beta_style.value)
        self.beta_style_legend.value = _legend_html(legend, viz, 16)

    def _beta_layer(self):
        if self.beta_mode.value == "Estilo do catálogo":
            if not self.beta_style.value:
                raise ValueError(_("Escolha um estilo do catálogo ou use 'Minhas cores'."))
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
                message_widget.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    def _read_asset(self, _event=None):
        def work():
            info = analyst.inspect_asset(self.beta_asset.value)
            years = (f"{info['years'][0]}–{info['years'][1]}" if info["years"]
                     else _("sem anos no nome das bandas"))
            if info["axis"] == "depth":
                years = _("sequência por profundidade")
            sample = ", ".join(info["bands"][:3]) + (" …" if len(info["bands"]) > 3 else "")
            kind = _("Imagem") if info["kind"] == "IMAGE" else _("Coleção")
            self.beta_info.value = (f"✓ {_('{tipo} com {n} período(s)', tipo=kind, n=len(info['bands']))} · {years}<br>"
                                    f"<span style='font-size:12px;color:var(--muted)'>{escape(sample)}</span>")
            if not self.beta_name.value:
                self.beta_name.value = self.beta_asset.value.rstrip("/").split("/")[-1].replace("_", " ")
        self._in_background(work, self.beta_info, _("Lendo o asset…"))

    def _detect_values(self, _event=None):
        def work():
            low, high = analyst.detect_range(self.beta_asset.value)
            self.beta_min.value, self.beta_max.value = low, high
            self.beta_msg.value = _("Valores encontrados no último período: {low} a {high}.", low=low, high=high)
        self._in_background(work, self.beta_msg, _("Calculando valores…"))

    def _beta_preview(self, _event=None):
        def work():
            layer = self._beta_layer()
            clips = self.combinations()
            territory = clips[0][2] if clips else (self.clips() or [None])[0]
            url, label = preview.layer_thumbnail(_ee(), layer, territory)
            where = escape(territory["name"]) if territory else _("área do asset")
            self.beta_preview_box.value = (f"<div style='font-size:12px;color:var(--muted)'>{escape(label)} · {where}</div>"
                                           f"<img src='{escape(url)}' style='max-width:560px;border:1px solid var(--line)'>"
                                           f"<div style='margin-top:4px'>{_legend_html(layer['legend'], _layer_viz(layer), 12)}</div>")
            self.beta_msg.value = ""
        self._in_background(work, self.beta_msg, _("Gerando prévia…"))

    def _save_beta(self, _event=None):
        try:
            layer = self._beta_layer()
            validate([layer], [])
            self.library.save_map(layer)
            self._add_custom_layer(layer)
            self.beta_msg.value = "✓ " + _("“{nome}” salvo em Meus mapas e escolhido. Ele já pode ser gerado "
                                           "como qualquer outro mapa.", nome=escape(layer["name"]))
        except Exception as exc:
            self.beta_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"

    def _add_custom_layer(self, layer):
        layer = dict(layer, initiative=MY_MAPS)
        self.catalog = [item for item in self.catalog if item["id"] != layer["id"]] + [layer]
        self.by_id[layer["id"]] = layer
        self.product_buttons.pop(layer["id"], None)
        self.year_ranges.pop(layer["id"], None)
        if MY_MAPS not in [value for _label, value in self.initiative.options]:
            self.initiative.options = list(self.initiative.options) + [(_(MY_MAPS), MY_MAPS)]
        if layer["id"] not in self.selected_layers:
            self.selected_layers.append(layer["id"])
        self.initiative.value = MY_MAPS
        self._refresh_filters()
        self._render_products()
        self._render_my_maps()
        self._toggle_layer(layer["id"], True)

    def _render_my_maps(self, *_args):
        if self.initiative.value != MY_MAPS:
            self.my_maps_panel.children = []
            return
        rows = [w.HTML(f"<b>{_('Gerenciar meus mapas')}</b>")]
        for layer in [item for item in self.catalog if item["initiative"] == MY_MAPS]:
            remove = w.Button(description=_("Remover"), icon="trash", layout=w.Layout(width="110px"))
            propose = w.Button(description=_("Proposta para o catálogo"), icon="share", layout=w.Layout(width="230px"))
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
            self.initiative.value = self.initiative.options[0][1]
            self.initiative.options = [o for o in self.initiative.options if o[1] != MY_MAPS]
        self._refresh_filters()
        self._render_products()
        self._render_my_maps()
        self._update_summary()

    def _visible_catalog(self):
        return [layer for layer in self.catalog if layer["initiative"] == self.initiative.value
                and (self.show_beta.value or layer.get("status") != "beta")]

    def _refresh_filters(self, *_args):
        items = self._visible_catalog()
        themes = list(dict.fromkeys(layer.get("theme") or "outros" for layer in items))
        self.theme.options = [(_("Todos"), "")] + [(ui_text(theme_label(t, "pt")), t) for t in themes]
        if self.theme.value not in ("",) + tuple(themes):
            self.theme.value = ""
        collections = sorted({str(layer.get("collection") or "") for layer in items} - {""},
                             key=lambda c: (not c.replace(".", "").isdigit(), [int(x) if x.isdigit() else 0 for x in c.split(".")], c))
        self.collection.options = [(_("Todas"), "")] + [(_("Coleção {c}", c=c) if c[:1].isdigit() else c, c)
                                                         for c in collections]
        if self.collection.value not in ("",) + tuple(collections):
            self.collection.value = ""

    def _render_products(self, *_args):
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
                label = ui_text(layer["name"]) + (f"  (col. {col})" if col and str(col)[:1].isdigit() else "") + suffix
                button = w.ToggleButton(description=label, tooltip=layer.get("reason") or ui_text(layer["name"]),
                                        layout=w.Layout(width="98%", height="32px", min_height="32px", flex="0 0 auto"))
                button.add_class("gf-option")
                button.observe(lambda change, i=layer["id"]: self._toggle_layer(i, change["new"]), names="value")
                self.product_buttons[layer["id"]] = button
            button.value = layer["id"] in self.selected_layers
            buttons.append(button)
        if not buttons:
            buttons = [w.HTML(f"<i>{_('Nenhum mapa com esses filtros.')}</i>")]
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
            self.product_detail.value = _card(f"<i>{_('Clique em um mapa para ver a legenda.')}</i>").value
        layer = self.by_id[layer_id]
        if on:
            col = layer.get("collection")
            initiative = _(layer["initiative"]) if layer["initiative"] == MY_MAPS else ui_text(layer["initiative"])
            theme_name = ui_text(theme_label(layer.get("theme") or "outros", "pt"))
            info = (f"<b>{escape(ui_text(layer['name']))}</b><br><span style='font-size:12px;color:var(--muted)'>"
                    f"{escape(initiative)} · {escape(theme_name)}"
                    f"{' · ' + escape(_('Coleção {c}', c=col)) if col else ''}</span>")
            if layer.get("status") == "beta":
                info += f"<br><span style='color:var(--warn);font-size:12px'>Beta: {escape(layer.get('reason', ''))}</span>"
            if layer.get("legend_label"):
                info += f"<div style='margin-top:6px;font-size:12px'><b>{escape(ui_text(layer['legend_label']))}</b></div>"
            rgb_note = _("Composição colorida (RGB): as cores vêm das bandas, sem legenda.")
            legend = (f"<div style='font-size:12px;color:var(--muted)'>{rgb_note}</div>" if layer.get("rgb") else
                      _legend_html(layer.get("legend"), _layer_viz(layer)))
            self.product_detail.value = _card(info + legend).value
        self._update_summary()

    # ------------------------------------------------------------------ 2. território
    def _build_territories(self):
        self.base = w.Dropdown(options=_options(TERRITORY_BASES), value="brasil", description=_("Base:"))
        self.mode = w.ToggleButtons(options=[(_(m), m) for m in TERRITORY_MODES],
                                    style={"button_width": "150px"})
        # Buscar
        self.kind = w.Dropdown(description=_("Tipo:"))
        self.place_search = w.Text(placeholder=_("Digite parte do nome (ex.: são félix, yanomami, chapada)"),
                                   continuous_update=True, layout=w.Layout(width="95%", max_width="420px"))
        self.place_results = w.VBox(layout=w.Layout(max_height="240px", overflow="hidden auto"))
        self.place_combine = w.Checkbox(value=False, description=_("Juntar os itens marcados em um único recorte"), indent=False)
        self.place_add = w.Button(description=_("Adicionar"), icon="plus", button_style="success")
        self.place_forget = w.Button(description=_("Apagar dos meus territórios"), icon="trash",
                                     layout=w.Layout(width="240px", display="none"))
        self.place_forget.on_click(self._forget_saved)
        self.place_msg = w.HTML()
        self._found = []
        self._marked = []
        self._search_thread = None
        self.place_lines = w.Button(description=_("Camada inteira como linhas"), icon="pencil",
                                    tooltip=_("Ex.: todos os biomas ou todos os estados desenhados sobre o mapa"),
                                    layout=w.Layout(width="250px"))
        self.place_lines.on_click(self._add_whole_as_lines)
        search_box = w.VBox([self.kind, self.place_search, self.place_results, self.place_combine,
                             w.HBox([self.place_add, self.place_lines, self.place_forget],
                                    layout=w.Layout(flex_flow="row wrap")), self.place_msg])
        # Desenhar
        self.draw_name = w.Text(placeholder=_("Nome da área (ex.: Entorno do Parque X)"), description=_("Nome:"))
        self.draw_add = w.Button(description=_("Adicionar área desenhada"), icon="plus", button_style="success")
        self.draw_msg = w.HTML(_("Desenhe com o retângulo ou o polígono no canto do mapa "
                                 "(vários desenhos viram uma área só)."))
        draw_box = w.VBox([self.draw_msg, self.draw_name, self.draw_add])
        # Arquivo
        self.upload = w.FileUpload(accept=".geojson,.json,.kml,.zip,.gpkg", multiple=False,
                                   description=_("Escolher arquivo"))
        self.upload_name = w.Text(placeholder=_("Nome da área (opcional)"), description=_("Nome:"))
        self.upload_add = w.Button(description=_("Adicionar arquivo"), icon="plus", button_style="success")
        self.upload_msg = w.HTML(_("GeoJSON, KML, GeoPackage ou shapefile em .zip "
                                   "(todas as feições viram uma área só)."))
        upload_box = w.VBox([self.upload_msg, self.upload, self.upload_name, self.upload_add])
        # Asset
        self.asset_id = w.Text(placeholder=_("projects/meu-projeto/assets/minha_area"), description=_("Asset:"),
                               layout=w.Layout(width="520px"))
        self.asset_field = w.Text(placeholder=_("opcional, ex.: NM_MUN"), description=_("Campo:"))
        self.asset_value = w.Text(placeholder=_("opcional, valores separados por vírgula"), description=_("Valor:"))
        self.asset_name = w.Text(placeholder=_("Nome da área (opcional)"), description=_("Nome:"))
        self.asset_add = w.Button(description=_("Adicionar asset"), icon="plus", button_style="success")
        self.asset_msg = w.HTML(_("A conta conectada precisa ter acesso de leitura ao asset (FeatureCollection)."))
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
                       theme.label(_("Territórios escolhidos")), self.chosen], layout=w.Layout(width="48%"))
        right = w.VBox([self.map if self.map is not None else
                        w.HTML("<i>" + _("Mapa indisponível (ipyleaflet não instalado). A busca, o arquivo e o "
                                         "asset continuam funcionando.") + "</i>")], layout=w.Layout(width="50%"))
        self._render_chosen()
        return w.VBox([theme.section(_("Onde?"), _("Escolha um ou mais territórios. "
                                                   "Cada mapa será gerado para cada território.")),
                       w.HBox([left, right])])

    def _make_map(self):
        try:
            import ipyleaflet
        except ImportError:
            return None, None
        # Os servidores do OpenStreetMap (403 no Colab) e da CARTO (pede chave) bloqueiam o uso no notebook;
        # o mapa topográfico da Esri é servido sem chave.
        light = ipyleaflet.TileLayer(
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            name="Mapa", base=True, max_zoom=19, attribution="Esri, HERE, Garmin, © OpenStreetMap contributors")
        m = ipyleaflet.Map(center=(-14, -52), zoom=4, scroll_wheel_zoom=True, basemap=light,
                           layout=w.Layout(height="460px", width="98%"))
        draw = ipyleaflet.DrawControl(polyline={}, circlemarker={}, marker={}, circle={},
                                      rectangle={"shapeOptions": {"color": "#1565c0"}},
                                      polygon={"shapeOptions": {"color": "#1565c0"}})
        m.add(draw)
        return m, draw

    def _show_mode(self, *_args):
        for name, box in self.mode_boxes.items():
            box.layout.display = "" if name == self.mode.value else "none"

    def _refresh_kinds(self, *_args):
        self.place_forget.layout.display = "" if self.base.value == "meus" else "none"
        self.place_combine.layout.display = "none" if self.base.value == "meus" else ""
        if self.base.value == "meus":
            self.kind.options = [(_("Salvos por você"), 0)]
            self.kind.value = 0
            if self._territories_started:
                self._start_search()
            return
        sources = territories.sources_for(self.base.value)
        self.kind.options = [(ui_text(s["label"]), i) for i, s in enumerate(sources)]
        self.kind.value = 1 if len(sources) > 1 and self.base.value == "brasil" else 0
        if self._territories_started:
            self._start_search()

    def _source(self):
        return territories.sources_for(self.base.value)[self.kind.value]

    def _start_search(self, *_args):
        if self.kind.value is None or not self._territories_started:
            return
        if self.base.value == "meus":
            term = territories.plain(self.place_search.value)
            self.place_search.disabled = False
            self._found = [{"saved": True, "name": t["name"], "territory": t} for t in self.library.territories
                           if not term or term in territories.plain(t["name"])]
            self._marked = []
            self.place_msg.value = "" if self._found else _("Nenhum território salvo ainda. Áreas desenhadas, "
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
        self.place_msg.value = f"<i>{_('Buscando…')}</i>"

        def work():
            try:
                found = territories.search(_ee(), source, text)
                if (text, kind) != (self.place_search.value, self.kind.value):
                    return  # o usuário já digitou outra coisa
                self._found, self._marked = found, []
                self.place_msg.value = "" if found else _("Nenhum resultado.")
            except Exception as exc:
                self._found = []
                self.place_msg.value = (f"<span style='color:var(--danger)'>{_('Não foi possível buscar:')} "
                                        f"{escape(str(exc))}</span>")
            self._render_found()
        thread = Thread(target=work, daemon=True)
        self._search_thread = thread
        thread.start()

    def _render_found(self):
        buttons = []
        for i, record in enumerate(self._found):
            button = w.ToggleButton(value=False, description=record["name"], layout=w.Layout(width="98%", min_height="30px", flex="0 0 auto"))
            button.add_class("gf-option")
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

    def _add_found(self, _event=None):
        try:
            marked = list(self._marked) or ([self._found[0]] if len(self._found) == 1 else [])
            if not marked:
                raise ValueError(_("Marque pelo menos um item da lista."))
            if marked[0].get("saved"):
                items = [record["territory"] for record in marked]
            elif marked[0].get("whole"):
                items = [territories.whole(marked[0]["source"])]
            else:
                items = territories.from_records(marked, combine=self.place_combine.value)
            added = sum(self._add_territory(item) for item in items)
            self.place_msg.value = "✓ " + _("{n} território(s) adicionado(s).", n=added)
            for button in self.place_results.children:
                button.value = False
        except Exception as exc:
            self.place_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"

    def _keep(self, territory):
        """Adiciona à seleção e guarda em 'Meus territórios' para as próximas sessões."""
        self._add_territory(territory)
        try:
            self.library.save_territory(territory)
        except OSError as exc:
            self.place_msg.value = (f"<span style='color:var(--warn)'>{_('Não foi possível guardar o território:')} "
                                    f"{escape(str(exc))}</span>")

    def _forget_saved(self, _event=None):
        marked = [record for record in self._marked if record.get("saved")]
        if not marked:
            self.place_msg.value = _("Marque os territórios salvos que quer apagar.")
            return
        for record in marked:
            self.library.remove_territory(record["territory"]["id"])
        self.place_msg.value = "✓ " + _("{n} território(s) apagado(s) dos seus salvos.", n=len(marked))
        self._start_search()

    def _add_drawing(self, _event=None):
        try:
            features = list(getattr(self.draw, "data", None) or [])
            geometry = territories.merge_drawings(features)
            self._keep(territories.from_geometry(geometry, self.draw_name.value or _("Área desenhada")))
            if self.draw is not None:
                self.draw.clear()
            self.draw_name.value = ""
            self.draw_msg.value = "✓ " + _("Área adicionada. Desenhe outra se quiser.")
        except Exception as exc:
            self.draw_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"

    def _add_upload(self, _event=None):
        try:
            files = self.upload.value
            files = list(files.values()) if isinstance(files, dict) else list(files)
            if not files:
                raise ValueError(_("Escolha um arquivo primeiro."))
            item = files[0]
            filename = item.get("name") or item.get("metadata", {}).get("name")
            geometry = territories.read_upload(filename, item["content"])
            name = self.upload_name.value or Path(filename).stem
            self._keep(territories.from_geometry(geometry, name))
            self.upload_msg.value = "✓ " + _("{nome} adicionado.", nome=escape(name))
        except Exception as exc:
            self.upload_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"

    def _add_asset(self, _event=None):
        try:
            item = territories.from_asset(self.asset_id.value, self.asset_name.value,
                                          self.asset_field.value, self.asset_value.value)
            size = territories.feature_collection(_ee(), item).size().getInfo()
            if not size:
                raise ValueError(_("O asset (ou o filtro) não retornou nenhuma feição."))
            self._keep(item)
            self.asset_msg.value = "✓ " + _("{nome} adicionado ({n} feição(ões)).", nome=escape(item["name"]), n=size)
        except Exception as exc:
            self.asset_msg.value = (f"<span style='color:var(--danger)'>{_('Não foi possível usar o asset:')} "
                                    f"{escape(str(exc))}</span>")

    def _remove_territory(self, territory_id):
        self.selected_territories = [t for t in self.selected_territories if t["id"] != territory_id]
        if self.map is not None:
            for layer in list(self.map.layers):
                if getattr(layer, "name", "") == f"territorio:{territory_id}":
                    self.map.remove(layer)
        self._render_chosen()
        self._update_summary()

    def clips(self):
        """Territórios que recortam o mapa (cada um gera uma combinação)."""
        return [t for t in self.selected_territories if t.get("role", "recorte") != "linha"]

    def lines(self):
        """Territórios desenhados como linhas sobre todos os recortes."""
        return [t for t in self.selected_territories if t.get("role") == "linha"]

    def _render_chosen(self):
        rows = []
        for territory in self.selected_territories:
            tid = territory["id"]
            remove = w.Button(icon="times", tooltip=_("Remover"), layout=w.Layout(width="36px"))
            remove.on_click(lambda _b, i=tid: self._remove_territory(i))
            role = theme.classed(w.ToggleButtons(options=_options(ROLES), value=territory.get("role", "recorte"),
                                                 tooltips=[_("Filtra o mapa: gera uma combinação"),
                                                           _("Desenha o contorno sobre o mapa")],
                                                 style={"button_width": "72px"}), "gf-tabs")
            color = w.Dropdown(options=_options(LINE_COLORS), value=territory.get("color", "000000"),
                               layout=w.Layout(width="110px"))
            width = w.Dropdown(options=_options(LINE_WIDTHS), value=territory.get("width", 2),
                               layout=w.Layout(width="90px"))
            line_box = w.HBox([color, width], layout=w.Layout(display="" if territory.get("role") == "linha" else "none"))
            role.observe(lambda c, i=tid, box=line_box: self._set_line(i, box, role=c["new"]), names="value")
            color.observe(lambda c, i=tid: self._set_line(i, None, color=c["new"]), names="value")
            width.observe(lambda c, i=tid: self._set_line(i, None, width=c["new"]), names="value")
            name = w.HTML(f"<span style='line-height:28px'>{escape(territory['name'])}</span>",
                          layout=w.Layout(flex="1 1 120px", min_width="0"))
            rows.append(w.HBox([remove, name, role, line_box],
                               layout=w.Layout(flex_flow="row wrap", align_items="center")))
        hint = ("<div style='font-size:12px;color:var(--muted)'>"
                + _("<b>Recorte</b> filtra o mapa (uma combinação por recorte). <b>Linha</b> desenha o contorno "
                    "sobre todos os recortes, na cor e espessura escolhidas.") + "</div>")
        empty = f"<i style='color:var(--muted)'>{_('Nenhum ainda.')}</i>"
        self.chosen.children = ([w.HTML(hint)] + rows) if rows else [w.HTML(empty)]

    def _set_line(self, territory_id, box, **changes):
        for territory in self.selected_territories:
            if territory["id"] == territory_id:
                territory.update(changes)
                territory.setdefault("color", "000000")
                territory.setdefault("width", 2)
                if box is not None:
                    box.layout.display = "" if territory.get("role") == "linha" else "none"
                self._redraw_on_map(territory)
        self._update_summary()

    def _add_whole_as_lines(self, _event=None):
        try:
            if self.base.value == "meus":
                raise ValueError(_("Escolha uma base oficial (ex.: Brasil) e o tipo (ex.: Bioma, Estado)."))
            source = self._source()
            item = dict(territories.whole(source), name=_("{camada} (todos)", camada=ui_text(source["label"])), role="linha",
                        color="000000", width=2)
            if self._add_territory(item):
                self.place_msg.value = "✓ " + _("{nome} adicionado como linhas. Troque a cor e a espessura na "
                                                "lista ao lado.", nome=escape(item["name"]))
            else:
                self.place_msg.value = _("Essa camada já está na lista.")
        except Exception as exc:
            self.place_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"

    def _redraw_on_map(self, territory):
        if self.map is None:
            return
        for layer in list(self.map.layers):
            if getattr(layer, "name", "") == f"territorio:{territory['id']}":
                self.map.remove(layer)
        self._draw_on_map(territory, fit=False)

    def _draw_on_map(self, territory, fit=True):
        if self.map is None:
            return

        def work():
            try:
                import ipyleaflet
                ee = _ee()
                url = (preview.outline_url(ee, territory, territory.get("color", "000000"))
                       if territory.get("role") == "linha" else preview.outline_url(ee, territory))
                self.map.add(ipyleaflet.TileLayer(url=url, name=f"territorio:{territory['id']}",
                                                  attribution="Google Earth Engine"))
                # fit_bounds depende do loop assíncrono do kernel; centro e zoom funcionam de qualquer thread.
                if fit:
                    self.map.center, self.map.zoom = preview.center_zoom(preview.bounds(ee, territory))
            except Exception as exc:
                self.place_msg.value = (f"<span style='color:var(--danger)'>{_('Não foi possível mostrar no mapa:')} "
                                        f"{escape(str(exc))}</span>")
        Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 3. período e saídas
    def _build_outputs(self):
        self.all_years = w.Checkbox(value=True, description=_("Todos os anos disponíveis"), indent=False)
        self.period = w.IntRangeSlider(value=(1985, 2025), min=1985, max=2025, step=1, description=_("Anos:"),
                                       continuous_update=False, layout=w.Layout(width="520px", display="none"))
        self.period_msg = w.HTML()
        self.kind_checks = {key: w.Checkbox(value=default, description=_(label), indent=False)
                            for key, label, default in KINDS}
        for key, label in ADVANCED_KINDS:
            self.kind_checks[key] = w.Checkbox(value=False, description=_(label), indent=False)
        self.advanced = w.ToggleButton(description=_("Opções avançadas"), icon="cog", value=False)
        self.save_drive = w.Checkbox(value=drive_mounted(), description=_("Salvar no meu Google Drive"), indent=False)
        self.drive_folder = w.Text(value=DEFAULT_DRIVE_FOLDER, description=_("Pasta:"), layout=w.Layout(width="520px"))
        self.drive_msg = w.HTML()
        env = self.environment
        options = sorted({1, 2, 3, 4, env["workers"]})
        self.workers = w.Dropdown(options=[(_("{n} combinação(ões)", n=n) + (_(" (sugerido)") if n == env["workers"] else ""), n)
                                           for n in options], value=env["workers"], description=_("Simultâneas:"),
                                  layout=w.Layout(width="360px"))
        self.workers.observe(self._set_workers, names="value")
        env_note = w.HTML(f"<span style='font-size:12px;color:var(--muted)'>"
                          + _("Ambiente: {ambiente}. Mais combinações ao mesmo tempo terminam antes, mas usam mais "
                              "memória.", ambiente=escape(env["label"])) + "</span>")
        self.map_lang = w.Dropdown(options=LANGUAGE_OPTIONS, value=map_language(), description=_("Idioma:"),
                                   layout=w.Layout(width="320px"))
        self.map_lang.observe(lambda c: set_language(maps=c["new"]), names="value")
        advanced_box = w.VBox([*(self.kind_checks[key] for key, _x in ADVANCED_KINDS), self.drive_folder,
                               self.workers, env_note], layout=w.Layout(display="none"))
        self.advanced.observe(lambda c: setattr(advanced_box.layout, "display", "" if c["new"] else "none"), names="value")
        self.preview_button = w.Button(description=_("Ver prévia"), icon="eye")
        self.preview_box = w.HTML()
        self.preview_button.on_click(self._preview)
        self.all_years.observe(self._period_mode, names="value")
        for widget in [self.all_years, self.period, self.save_drive, *self.kind_checks.values()]:
            widget.observe(lambda _c: self._update_summary(), names="value")
        self.save_drive.observe(self._drive_note, names="value")
        self._drive_note()
        return w.VBox([
            theme.section(_("Quando e como?")),
            theme.label(_("Período")), self.all_years, self.period, self.period_msg,
            theme.label(_("Arquivos")), *(self.kind_checks[key] for key, _x, _y in KINDS),
            theme.label(_("Idioma dos mapas")), self.map_lang,
            theme.label(_("Destino")), self.save_drive, self.drive_msg, self.advanced, advanced_box,
            w.HTML(f"<b>{_('Prévia')}</b> <span style='color:var(--muted);font-size:12px'>"
                   f"({_('primeiro mapa e primeiro território, último ano escolhido')})</span>"),
            self.preview_button, self.preview_box])

    def _set_workers(self, change):
        self.runner.workers = change["new"]
        self._update_summary()

    def _period_mode(self, *_args):
        self.period.layout.display = "none" if self.all_years.value else ""

    def _drive_note(self, *_args):
        if self.save_drive.value and not drive_mounted():
            self.drive_msg.value = ("<span style='color:var(--danger)'>"
                                    + _("O Drive não está conectado. Rode a célula 1 de novo ou desmarque esta opção "
                                        "(os arquivos ficam só nesta sessão).") + "</span>")
        elif self.save_drive.value:
            self.drive_msg.value = _("Os arquivos ficam na sua pasta do Drive e o que já existir não é refeito.")
        else:
            self.drive_msg.value = ("<span style='color:var(--warn)'>"
                                    + _("Os arquivos ficam só nesta sessão do Colab e somem quando ela fechar. "
                                        "Baixe-os no passo 4.") + "</span>")

    def _load_years(self):
        pending = [i for i in self.selected_layers if i not in self.year_ranges]
        if not pending:
            self._apply_years()
            return
        self.period_msg.value = f"<i>{_('Consultando os anos disponíveis…')}</i>"

        def work():
            for layer_id in pending:
                try:
                    self.year_ranges[layer_id] = preview.years(self.by_id[layer_id])
                except Exception as exc:
                    self.year_ranges[layer_id] = None
                    self.period_msg.value = (f"<span style='color:var(--danger)'>{escape(self.by_id[layer_id]['name'])}: "
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
        self.period_msg.value = _("Disponível: {inicio}–{fim}.", inicio=low, fim=high)
        self._update_summary()
        if STEPS.index(self.stepper.value) == 3:
            self._render_plan()

    def _layer_for_run(self, layer_id):
        layer = dict(self.by_id[layer_id])
        span = self.year_ranges.get(layer_id)
        if not self.all_years.value and span:
            start, end = self.period.value
            layer["start_year"], layer["end_year"] = max(start, span[0]), min(end, span[1])
            if layer["start_year"] > layer["end_year"]:
                return None
        return layer

    def _preview(self, _event=None):
        if not self.selected_layers or not self.selected_territories:
            self.preview_box.value = _("Escolha um mapa e um território primeiro.")
            return
        self.preview_box.value = f"<i>{_('Gerando prévia…')}</i>"
        layer = self._layer_for_run(self.selected_layers[0]) or self.by_id[self.selected_layers[0]]
        combos = self.combinations()
        if not combos:
            self.preview_box.value = _("Marque pelo menos um território como Recorte.")
            return
        territory = combos[0][2]
        year = None if self.all_years.value else self.period.value[1]

        def work():
            try:
                url, label = preview.thumbnail_url(_ee(), layer, territory, year)
                note = _("Prévia rápida, sem moldura, legenda ou título; o arquivo final terá o layout completo.")
                self.preview_box.value = (f"<div style='font-size:12px;color:var(--muted)'>{escape(ui_text(layer['name']))} · "
                                          f"{escape(territory['name'])} · {escape(label)}</div>"
                                          f"<img src='{escape(url)}' style='max-width:640px;border:1px solid var(--line)'>"
                                          f"<div style='font-size:11px;color:var(--muted)'>{note}</div>")
            except Exception as exc:
                self.preview_box.value = (f"<span style='color:var(--danger)'>{_('Prévia indisponível:')} "
                                          f"{escape(str(exc))}</span>")
        Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------ 4. gerar
    def _build_generate(self):
        self.plan = w.HTML()
        self.estimate_box = w.HTML()
        self.check = w.Button(description=_("Verificar o que já existe"), icon="search", layout=w.Layout(width="230px"))
        self.check_out = w.HTML()
        self.run = w.Button(description=_("Gerar"), icon="play", button_style="success",
                            layout=w.Layout(width="220px", height="40px"))
        self.results = self.gallery.widget
        self.check.on_click(self._check)
        self.run.on_click(lambda _b: self.start(wait=False))
        return w.VBox([theme.section(_("Conferir e gerar")), self.plan, self.estimate_box,
                       w.HBox([self.run, self.check], layout=w.Layout(align_items="center", grid_gap="12px")),
                       self.check_out, self.runner.widget(),
                       w.HTML(f"<b>{_('Vitrine do lote')}</b> <span style='color:var(--muted);font-size:12px'>("
                              + _("os cartões se atualizam enquanto gera; clique em Abrir para ver, copiar o caminho "
                                  "ou baixar") + ")</span>"),
                       self.gallery.widget])

    def combinations(self):
        combos = []
        for layer_id in self.selected_layers:
            layer = self._layer_for_run(layer_id)
            if layer is None:
                continue
            lines = [{k: v for k, v in line.items() if k in ("id", "name", "asset", "filter", "geometry",
                                                              "color", "width")} for line in self.lines()]
            for territory in self.clips():
                item = {k: v for k, v in territory.items() if k not in ("role", "color", "width")}
                if lines:
                    item["overlays"] = lines
                combos.append((layer["initiative"], dict(layer, map_lang=map_language()), item))
        return combos

    def kinds(self):
        return {key for key, check in self.kind_checks.items() if check.value}

    def _render_plan(self):
        combos = self.combinations()
        if not combos:
            missing = (_("todos os territórios estão como Linha; marque pelo menos um como Recorte (passo 2)")
                       if self.selected_layers and self.selected_territories else
                       _("falta escolher pelo menos um mapa (passo 1) e um território (passo 2)"))
            self.plan.value = (f"<span style='color:var(--danger)'>"
                               f"{_('Não dá para gerar ainda: {motivo}.', motivo=missing)}</span>")
            self.estimate_box.value = ""
            return
        rows = "".join(f"<tr><td>{escape(ui_text(layer['name']))}</td><td>{escape(territory['name'])}</td></tr>"
                       for _x, layer, territory in combos[:30])
        more = f"<p>{_('… e mais {n}.', n=len(combos) - 30)}</p>" if len(combos) > 30 else ""
        self.plan.value = (f"<details><summary><b>{_('{n} combinação(ões)', n=len(combos))}</b> "
                           f"({_('mapa × território')})</summary>"
                           f"<table style='font-size:13px'><tr><th align=left>{_('Mapa')}</th>"
                           f"<th align=left>{_('Território')}</th></tr>{rows}</table>{more}</details>")
        self.estimate_box.value = self._estimate_html() if self.kinds() else ""

    def _check(self, _event=None):
        combos, kinds = self.combinations(), self.kinds()
        if not combos or not kinds:
            self.check_out.value = _("Escolha mapa, território e tipo de arquivo.")
            return
        self.check_out.value = f"<i>{_('Verificando…')}</i>"

        def work():
            try:
                lines = self.runner.availability(combos, kinds, self.save_drive.value, self.drive_folder.value)
                self.check_out.value = "<br>".join(
                    f"{escape(_(state))} — {escape(ui_text(layer['name']))} / {escape(territory['name'])}"
                    for (_x, layer, territory), state in lines)
            except Exception as exc:
                self.check_out.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"
        Thread(target=work, daemon=True).start()

    def start(self, wait=True):
        """Gera todas as combinações escolhidas (também usado pela célula opcional do notebook)."""
        return self.runner.start(self.combinations(), self.kinds(), self.save_drive.value,
                                 self.drive_folder.value, wait=wait)

    def _gallery_start(self, combos):
        self.gallery.reset()
        for index, (_x, layer, territory) in enumerate(combos):
            self.gallery.add(index, ui_text(layer["name"]), territory["name"])

    def _gallery_update(self, index, state, info):
        card = self.gallery.card(index)
        if card is None or (state == "gerando" and card.state in ("pronto", "existe", "falha")):
            return
        card.update(state, info.get("detail", ""), info.get("done", 0), info.get("total", 0))
        if info.get("files"):
            card.set_files(info["files"], info.get("folder"))

    def _show_results(self, files, summary):
        # A vitrine já foi preenchida cartão a cartão; aqui só se ajusta a estimativa ao ritmo real.
        if STEPS.index(self.stepper.value) == 3:
            self._render_plan()

    # ------------------------------------------------------------------ 5. biblioteca
    def _build_library(self):
        self.library_source = w.ToggleButtons(options=_options([(N_("Meu Drive"), "drive"),
                                                                (N_("Esta sessão do Colab"), "local")]),
                                              value="drive" if drive_mounted() else "local",
                                              style={"button_width": "170px"})
        self.library_refresh = w.Button(description=_("Atualizar"), icon="refresh", layout=w.Layout(width="120px"))
        self.library_search = w.Text(placeholder=_("Filtrar por mapa ou território"), continuous_update=True,
                                     layout=w.Layout(width="320px"))
        self.library_msg = w.HTML()
        self.library_gallery = Gallery(zip_root=self.gallery.zip_root, empty=_("Nenhum resultado encontrado nesta pasta."),
                                       filters=False)
        self._library_entries = []
        self.library_source.observe(lambda _c: self._scan_library(), names="value")
        self.library_refresh.on_click(lambda _b: self._scan_library())
        self.library_search.observe(lambda _c: self._fill_library(), names="value")
        return w.VBox([theme.section(_("Biblioteca"), _("Tudo o que já foi gerado, na pasta do Drive (passo 3) ou "
                                                         "nesta sessão. Abra um cartão para ver a animação, copiar o "
                                                         "caminho ou baixar.")),
                       w.HBox([self.library_source, self.library_refresh], layout=w.Layout(grid_gap="8px")),
                       self.library_search, self.library_msg, self.library_gallery.widget])

    def _library_root(self):
        return Path(self.drive_folder.value) if self.library_source.value == "drive" else self.output_root

    def _scan_library(self):
        root = self._library_root()
        if self.library_source.value == "drive" and not drive_mounted():
            self.library_msg.value = ("<span style='color:var(--warn)'>"
                                      + _("O Drive não está conectado nesta sessão. Rode a célula 1 com o Drive ou "
                                          "veja “Esta sessão do Colab”.") + "</span>")
            self._library_entries = []
            self._fill_library()
            return
        self.library_msg.value = f"<i>{_('Procurando em {pasta}…', pasta=escape(str(root)))}</i>"

        def work():
            try:
                self._library_entries = scan_library(root)
                found = _("{n} resultado(s) em", n=len(self._library_entries))
                self.library_msg.value = (f"<span style='font-size:12px;color:var(--muted)'>{found} "
                                          f"<code style='user-select:all'>{escape(str(root))}</code></span>")
            except Exception as exc:
                self._library_entries = []
                self.library_msg.value = f"<span style='color:var(--danger)'>{escape(str(exc))}</span>"
            self._fill_library()
        Thread(target=work, daemon=True).start()

    def _fill_library(self):
        term = territories.plain(self.library_search.value)
        self.library_gallery.reset()
        for entry in self._library_entries:
            if term and not all(part in territories.plain(f"{entry['title']} {entry['subtitle']}")
                                for part in term.split()):
                continue
            card = self.library_gallery.add(entry["key"], ui_text(entry["title"]), entry["subtitle"])
            card.update("pronto")
            card.set_files(entry["files"], entry["folder"])


LANGUAGE_OPTIONS = [(name, code) for name, code in LANGUAGES]


def show_app(project, layers=(), **kwargs):
    """Abre a Fábrica. `layers` são mapas próprios (opcionais), mostrados em 'Meus mapas'.

    `language` escolhe o idioma da interface e `map_language` o dos mapas (padrão: o mesmo da interface).
    Opções: pt, en, es, fr, id, zh (ou o nome: "English", "Español"...).
    """
    return FactoryApp(project, layers, **kwargs)
