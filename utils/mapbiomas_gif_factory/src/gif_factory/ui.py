"""Seleção hierárquica e progresso do lote no Colab."""
from datetime import datetime
from pathlib import Path
from threading import Thread, Event, Lock, Timer
from html import escape
from copy import deepcopy
import json
import re
import ipywidgets as w
from IPython.display import display
from .engine import generate, validate, discover
from .storage import list_artifacts
from .presets import LAYERS, TERRITORIES
from .layout import combination_path, ensure_unique_paths
from .progress import ProgressState
from .library import assess, fingerprint, ingest, variant_for, sync_selected
from .territory_search import (TERRITORY_SEARCH_SOURCES, search_configured,
                               search_municipalities, territory as searched_territory)
from .reference_catalog import INITIATIVES, build_reference_layers


def _duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def show_ui(layers, territories, output_root="/content/mapbiomas_gif_factory_output", project=None, temp_root="/content/temp"):
    validate(layers, territories)
    validate(LAYERS, TERRITORIES)
    log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="250px", overflow="auto"))
    status = w.HTML(value="<b>Pronto / Ready / Listo.</b> Selecione as opções e clique em Gerar GIFs / Select options and click Generate / Seleccione opciones y pulse Generar.")
    overall = w.IntProgress(value=0, min=0, max=1, description="Lote:", layout=w.Layout(width="95%"))
    progress = w.IntProgress(value=0, min=0, max=1, description="Download:", layout=w.Layout(width="95%", display="none"))
    batch_lock = Lock()
    sprout = w.HTML(
        value=(
            '<style>@keyframes gif-factory-sprout {'
            '0%,100% { transform: rotate(-8deg); } 50% { transform: rotate(8deg); }}'
            '.gif-factory-sprout { display: inline-block; transform-origin: bottom center; '
            'animation: gif-factory-sprout 1.5s ease-in-out infinite; }'
            '@media (prefers-reduced-motion: reduce) { .gif-factory-sprout { animation: none; }}'
            '</style><span role="status" aria-label="Processamento em andamento">'
            '<span class="gif-factory-sprout" aria-hidden="true">🌱</span> '
            '<span style="color:#4b7255; font-size:12px">Cultivando mapas…</span></span>'
        ),
        layout=w.Layout(display="none", margin="0 0 0 10px"),
    )
    selection = w.SelectMultiple(description="Arquivos:", layout=w.Layout(width="95%", height="180px"))
    drive = w.Text(value="/content/drive/MyDrive/mapbiomas_gif_factory", description="Drive:", layout=w.Layout(width="95%"))
    auto = w.Checkbox(value=False, description="Usar biblioteca no Drive e enviar automaticamente")
    kinds = {k: w.Checkbox(value=(k == "gif"), description=label) for k, label in [("gif", "GIFs"), ("colagem", "Colagens"), ("brutos", "Frames: mapa básico"), ("preparados", "Frames: mapa com ano e escala"), ("finais", "Frames: mapa completo")]}
    initiative_options = ["Brasil", "Peru"] + [name for name in INITIATIVES.values() if name not in ("Brasil", "Peru")] + ["Personalizado"]
    initiative = w.ToggleButtons(options=initiative_options, value="Brasil", description="Iniciativa:", layout=w.Layout(width="100%"))
    theme_nav = w.ToggleButtons(options=[], description="Tema:")

    inventory_path = Path(__file__).with_name("reference_inventory.json")
    records = json.loads(inventory_path.read_text(encoding="utf-8"))["records"] if inventory_path.is_file() else []
    reference_layers = build_reference_layers(records)

    # Catálogo de presets e camadas do usuário ficam separados por iniciativa.
    catalog = [(country, [x for x in LAYERS if x.get("initiative", "Brasil") == country]
                + [x for x in reference_layers if x["initiative"] == country])
               for country in initiative.options if country != "Personalizado"] + [("Personalizado", layers)]
    product_checks = []
    theme_panels = {}
    for origin, entries in catalog:
        groups = {}
        for layer in entries:
            theme = layer.get("theme", "Fogo" if origin == "Brasil" else "Outros")
            collection = layer.get("collection", "Coleção 5" if origin == "Brasil" else "Sem coleção")
            groups.setdefault((theme, collection), []).append(layer)
        sections = []
        for (theme, collection), group in groups.items():
            checks = []
            for layer in group:
                enabled = layer.get("enabled", True)
                label = layer["name"] + (" [pendente]" if not enabled else " [experimental]" if layer.get("status") == "experimental" else "")
                check = w.Checkbox(value=False, description=label, disabled=not enabled, indent=False, layout=w.Layout(width="95%"))
                check._reference = str(layer.get("id", "")).startswith("ref_")
                product_checks.append((origin, layer, check))
                checks.append(check)
                if not enabled:
                    checks.append(w.HTML(f"<small>{escape(layer.get('reason', 'Configuração pendente.'))}</small>"))
            panel = w.VBox([w.HTML(f"<b>{escape(str(collection))}</b>"), *checks])
            panel.theme = str(theme)
            sections.append(panel)
        if not sections:
            sections.append(w.HTML("Nenhuma camada configurada."))
        theme_panels[origin] = sections
    theme_body = w.VBox([])
    catalog_panel = w.VBox([w.HTML("<h3>Produtos</h3>"), theme_nav, theme_body])

    # Territórios personalizados continuam disponíveis em qualquer iniciativa.
    territory_checks = []
    territory_boxes = []
    brazil_territories = [t for t in TERRITORIES if not t["id"].startswith("peru")]
    peru_territories = [t for t in TERRITORIES if t["id"].startswith("peru")]
    for name, entries in [("Brasil", brazil_territories), ("Peru", peru_territories), ("Personalizados", territories)]:
        checks = []
        for territory in entries:
            check = w.Checkbox(value=False, description=territory["name"], indent=False)
            territory_checks.append((name, territory, check))
            checks.append(check)
        territory_boxes.append(w.VBox([w.HTML(f"<b>{name}</b>"), *checks]))

    territory_type = w.ToggleButtons(options=list(TERRITORY_SEARCH_SOURCES), value="Município", description="Tipo:")
    municipality_text = w.Text(description="Município:", placeholder="Digite o nome", continuous_update=True, layout=w.Layout(width="60%"))
    municipality_results = w.VBox([])
    municipality_combine = w.Checkbox(value=False, description="Combinar selecionados em um recorte")
    municipality_add = w.Button(description="Adicionar selecionados", button_style="primary")
    municipality_log = w.Output()
    municipality_found = []
    _search_timer = [None]

    def run_municipality_search():
        with municipality_log:
            municipality_log.clear_output()
            term = municipality_text.value.strip()
            if len(term) < 2:
                municipality_found.clear()
                _render_municipality_results()
                return
            try:
                import ee
                municipality_found[:] = (search_municipalities(ee, term)
                                         if territory_type.value == "Município"
                                         else search_configured(ee, territory_type.value, term))
                if not municipality_found:
                    print("Nenhum resultado / No results / Sin resultados.")
            except Exception as exc:
                municipality_found.clear()
                print(f"Não foi possível buscar: {exc}")
            _render_municipality_results()

    def _render_municipality_results():
        buttons = []
        for i, record in enumerate(municipality_found):
            button = w.Button(description=record["name"], layout=w.Layout(width="95%"), icon="plus")
            button.style.button_color = "#e0e0e0"
            button.on_click(lambda _b, i=i: _toggle_result(i))
            buttons.append(button)
        municipality_results.children = buttons

    def _toggle_result(index):
        record = municipality_found[index]
        if record in _selected_municipalities:
            _selected_municipalities.remove(record)
            municipality_results.children[index].style.button_color = "#e0e0e0"
            municipality_results.children[index].icon = "plus"
        else:
            _selected_municipalities.append(record)
            municipality_results.children[index].style.button_color = "#cfe8d0"
            municipality_results.children[index].icon = "check"

    _selected_municipalities = []

    def schedule_search(change=None):
        if _search_timer[0] is not None:
            _search_timer[0].cancel()
        _search_timer[0] = Timer(0.4, run_municipality_search)
        _search_timer[0].daemon = True
        _search_timer[0].start()

    def add_municipalities(_):
        with municipality_log:
            try:
                selected = list(_selected_municipalities)
                if not selected:
                    raise ValueError("Selecione pelo menos um município.")
                groups = [selected] if municipality_combine.value else [[r] for r in selected]
                for group in groups:
                    item = searched_territory(group)
                    existing = next((check for _, t, check in territory_checks if t["id"] == item["id"]), None)
                    if existing is not None:
                        existing.value = True
                        continue
                    check = w.Checkbox(value=True, description=item["name"], indent=False, layout=w.Layout(width="95%"))
                    territory_checks.append(("Personalizados", item, check))
                    territory_boxes[2].children = (*territory_boxes[2].children, check)
                _selected_municipalities.clear()
                _render_municipality_results()
                print("Adicionado em Personalizados; desmarque para remover da seleção.")
            except Exception as exc:
                print(f"Não foi possível adicionar: {exc}")

    municipality_text.observe(schedule_search, names="value")
    municipality_add.on_click(add_municipalities)
    def territory_type_changed(change):
        municipality_found.clear()
        _selected_municipalities.clear()
        municipality_results.children = []
        municipality_text.value = ""
        municipality_text.description = "Nome:" if change["new"] == "Município" else "Nome:"
        municipality_text.placeholder = "Digite o nome" if change["new"] == "Município" else "Digite o nome"
    territory_type.observe(territory_type_changed, names="value")

    def render_themes(*_):
        origin = initiative.value
        sections = theme_panels[origin]
        themes = []
        for panel in sections:
            theme = getattr(panel, "theme", None)
            if theme and theme not in themes:
                themes.append(theme)
        previous = theme_nav.value
        theme_nav.options = themes
        if themes and previous not in themes:
            theme_nav.value = themes[0]
        theme_nav.layout.display = "" if len(themes) > 1 else "none"
        for panel in sections:
            theme = getattr(panel, "theme", None)
            visible = theme is None or len(themes) <= 1 or theme == theme_nav.value
            panel.layout.display = "" if visible else "none"
        theme_body.children = [*sections]

    def change_initiative(change=None):
        if change:
            for _, _, check in product_checks + territory_checks:
                check.value = False
        active = initiative.value
        # "Personalizado" é um espaço livre: camadas próprias e qualquer território.
        territory_boxes[0].layout.display = "" if active in ("Brasil", "Personalizado") else "none"
        territory_boxes[1].layout.display = "" if active in ("Peru", "Personalizado") else "none"
        render_themes()
    theme_nav.observe(render_themes, names="value")
    initiative.observe(change_initiative, names="value")

    def update_files(*_):
        files = list_artifacts(output_root)
        selection.options = [(str(p.relative_to(output_root)), str(p)) for p in files]

    def publish(files, destination):
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Monte o Google Drive antes de enviar os arquivos.")
        return sync_selected(files, output_root, destination)

    availability = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="200px", overflow="auto"))
    check_status = w.Button(description="Verificar disponibilidade", layout=w.Layout(width="210px"))
    def selected_combinations():
        active = initiative.value
        allowed_territories = ("Brasil", "Peru", "Personalizados") if active == "Personalizado" else (active, "Personalizados")
        return [(active, deepcopy(layer), deepcopy(territory))
                for origin, layer, check in product_checks if check.value and origin == active
                for group, territory, tcheck in territory_checks if tcheck.value and group in allowed_territories]
    def library_root():
        if auto.value:
            if not Path("/content/drive/MyDrive").is_dir():
                raise RuntimeError("Monte o Drive antes de consultar ou gerar para a biblioteca do Drive.")
            return Path(drive.value)
        return Path(output_root)
    def check_selected(_):
        combos = selected_combinations()
        selected_kinds = {k for k, checkbox in kinds.items() if checkbox.value}
        availability.clear_output()
        if not combos or not selected_kinds:
            availability.append_stdout("Selecione camada, território e tipo de saída.\n")
            return
        try:
            root = library_root()
        except Exception as exc:
            availability.append_stdout(f"Erro: {exc}\n")
            return
        availability.append_stdout(f"Biblioteca consultada: {root}\n")
        check_status.disabled = True
        def work():
            try:
                for origin, layer, territory in combos:
                    try:
                        all_bands, bands = discover(layer)
                        variant = variant_for(all_bands, bands)
                        result = assess(root / combination_path(origin, layer, territory), variant,
                                        fingerprint(layer, territory, bands), selected_kinds)
                        labels = {"disponivel": "✓ Disponível", "parcial": "◐ Parcial", "nao_gerado": "○ Não gerado", "verificar": "⚠ Verificar"}
                        availability.append_stdout(f"{labels[result]}: {origin} / {layer['name']} / {territory['name']} ({variant})\n")
                    except Exception as exc:
                        availability.append_stdout(f"⚠ Verificar: {layer['name']} / {territory['name']}: {exc}\n")
            finally:
                check_status.disabled = False
        Thread(target=work, daemon=True).start()
    check_status.on_click(check_selected)
    refresh = w.Button(description="Atualizar arquivos")
    send = w.Button(description="Enviar selecionados")
    refresh.on_click(update_files)
    def on_send(_):
        try:
            if not selection.value:
                raise ValueError("Selecione ao menos um arquivo.")
            for path, state, detail in publish(selection.value, drive.value):
                log.append_stdout(f"Drive: {state}: {path} {detail}\n")
        except Exception as exc:
            log.append_stdout(f"ERRO no envio: {exc}\n")
    send.on_click(on_send)

    def on_run(_):
        combos = selected_combinations()
        selected_kinds = {k for k, checkbox in kinds.items() if checkbox.value}
        if not combos or not selected_kinds:
            status.value = "<b>Selecione produto, território e saída.</b>"
            return
        try:
            for _, layer, territory in combos:
                validate([layer], [territory])
            ensure_unique_paths(combos)
        except ValueError as exc:
            status.value = f"<b>Seleção inválida:</b> {escape(str(exc))}"
            return
        sprout.layout.display = ""
        destination, automatic = drive.value, auto.value
        run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        delivery_root = Path(output_root)
        status.value = "<b>▶ INICIANDO</b> — preparando lote..."
        progress.value = 0
        overall.value = 0
        overall.max = len(combos)
        log.clear_output()
        def worker():
            total = len(combos)
            state = ProgressState(total)
            stop_clock = Event()
            ok = failed = skipped = 0
            def render():
                snap = state.snapshot()
                eta = "calculando…" if snap["eta"] is None else "~" + _duration(snap["eta"])
                details = f"<br>{escape(snap['detail'])}" if snap['detail'] else ""
                download = snap["stage"] == "Baixando frames"
                remaining = f"<br>Previsão do download atual: {eta}" if download else ""
                status.value = (f"<b>◉ LOTE EM ANDAMENTO</b> — {snap['completed']} de {snap['total']} combinações processadas"
                                f"<br>{escape(snap['header'])}<br><b>{escape(snap['stage'])}</b>{details}"
                                f"<br>Tempo total: {_duration(snap['elapsed'])} · combinação: {_duration(snap['combo_elapsed'])}{remaining}")
                overall.value = snap["completed"]
                progress.layout.display = "" if download else "none"
                if download:
                    progress.max = max(1, snap["frame_total"])
                    progress.value = min(snap["frame_done"], progress.max)
            def tick():
                while not stop_clock.wait(1):
                    try:
                        render()
                    except Exception as exc:
                        log.append_stdout(f"Aviso de atualização do painel: {exc}\n")
            ticker = Thread(target=tick, daemon=True)
            try:
                ticker.start()
                for number, (origin, layer, territory) in enumerate(combos, 1):
                    header = f"{origin} / {layer.get('theme', 'Fogo' if origin == 'Brasil' else 'Personalizado')} / {layer.get('collection', 'Coleção 5' if origin == 'Brasil' else 'Sem coleção')} / {layer['name']} / {territory['name']}"
                    state.begin(number, header)
                    render()
                    log.append_stdout(f"INICIANDO {number}/{total}: {header}\n")
                    discovered = []
                    def event(kind, *args):
                        if kind == "discovered":
                            discovered[:] = args[0]
                            log.append_stdout(f"Descobertas {len(discovered)} bandas: {discovered[0]} → {discovered[-1]}\n")
                        if kind == "stage":
                            log.append_stdout(f"{args[0]}...\n")
                        state.update(kind, *args)
                        if kind in ("frame_retry", "frame_failed"):
                            log.append_stdout(state.snapshot()["detail"] + "\n")
                        render()
                    try:
                        relative = combination_path(origin, layer, territory)
                        local_root = delivery_root / relative
                        if automatic and not Path("/content/drive/MyDrive").is_dir():
                            raise RuntimeError("Drive não montado; monte o Drive ou desative o envio automático.")
                        primary_root = (Path(destination) / relative) if automatic else local_root
                        state.update("stage", "Consultando bandas e biblioteca")
                        render()
                        all_bands, selected_bands = discover(layer)
                        variant = variant_for(all_bands, selected_bands)
                        signature = fingerprint(layer, territory, selected_bands)
                        existing = assess(primary_root, variant, signature, selected_kinds)
                        if existing == "disponivel":
                            skipped += 1
                            log.append_stdout(f"DISPONÍVEL {number}/{total}: já existe no destino ({variant}); não reprocessado.\n")
                            continue
                        if existing == "verificar":
                            raise RuntimeError("Existem arquivos incompatíveis ou sem metadados confiáveis; verifique a biblioteca. Nada foi sobrescrito.")
                        if existing == "parcial":
                            log.append_stdout("Resultado parcial: gerando novamente para preencher apenas arquivos ausentes.\n")
                        missing_kinds = {kind for kind in selected_kinds if assess(primary_root, variant, signature, {kind}) != "disponivel"}
                        run_root = Path(temp_root) / relative / variant / run_id
                        stage = run_root / "entrega"
                        configured = dict(layer, bands=selected_bands)
                        files = generate(configured, territory, stage, missing_kinds, None,
                                         project=project, on_progress=event, work_dir=run_root / "processamento", variant=variant)
                        state.update("stage", "Enviando ao Drive" if automatic else "Publicando arquivos locais")
                        render()
                        if automatic:
                            ingest(stage, primary_root, variant, signature, selected_bands, layer, territory)
                            try:
                                ingest(stage, local_root, variant, signature, selected_bands, layer, territory)
                            except Exception as exc:
                                log.append_stdout(f"Aviso: cópia local não atualizada: {exc}\n")
                        else:
                            ingest(stage, local_root, variant, signature, selected_bands, layer, territory)
                        ok += 1
                        log.append_stdout(f"FINALIZADO {number}/{total}: {len(files)} arquivos publicados em {primary_root}\n")
                    except Exception as exc:
                        failed += 1
                        log.append_stdout(f"FALHA {number}/{total}: {exc}\n")
                    finally:
                        state.finish_combo()
                        render()
                update_files()
                stop_clock.set()
                if ticker.is_alive():
                    ticker.join()
                title = "✓ LOTE CONCLUÍDO" if not failed else "⚠ CONCLUÍDO COM FALHAS" if ok or skipped else "✕ LOTE FALHOU"
                drive_status = ("verifique as falhas no log" if failed else "resultados disponíveis no Drive") if automatic else "não utilizado para esta execução"
                status.value = f"<b>{title}</b> · {ok} gerado(s), {skipped} já disponível(is), {failed} falha(s)<br>Drive: {drive_status}<br>Tempo total: {_duration(state.snapshot()['elapsed'])}"
            except Exception as exc:
                stop_clock.set()
                if ticker.is_alive():
                    ticker.join()
                status.value = f"<b>Erro ao finalizar o lote:</b> {escape(str(exc))}"
                log.append_stdout(f"Erro ao finalizar lote: {exc}\n")
            finally:
                stop_clock.set()
                sprout.layout.display = "none"
                batch_lock.release()
        thread = Thread(target=worker, daemon=True)
        thread.start()
        return thread
    def start_batch(wait=True):
        if not batch_lock.acquire(blocking=False):
            status.value = "<b>Já existe um lote em andamento.</b>"
            return
        thread = None
        try:
            thread = on_run(None)
            if thread is None:
                batch_lock.release()
                return
            if wait:
                thread.join()
        except Exception as exc:
            if thread is None:
                batch_lock.release()
            sprout.layout.display = "none"
            status.value = f"<b>Erro ao iniciar o lote:</b> {escape(str(exc))}"
            log.append_stdout(f"Erro ao iniciar lote: {exc}\n")
    run_button = w.Button(description="Gerar / Generate / Generar", button_style="success", icon="play", layout=w.Layout(width="250px"))
    run_button.on_click(lambda _: start_batch(wait=False))
    update_files()
    change_initiative()
    municipality_panel = w.VBox([w.HTML("<b>Buscar território</b><p>Digite o nome para ver resultados. Municípios: FAO GAUL 2025 (não é a malha oficial IBGE). UC e TI aguardam conexão das fontes.</p>"),
                                territory_type, municipality_text, municipality_results,
                                municipality_combine, municipality_add, municipality_log])
    territories_panel = w.VBox([w.HTML("<h3>Territórios</h3>"), municipality_panel, *territory_boxes])
    output_panel = w.VBox([w.HTML("<h3>Saídas e destino</h3>"), *kinds.values(), auto, drive, check_status, availability])
    delivery_panel = w.VBox([w.HTML("<h3>Arquivos gerados e envio posterior</h3>"), refresh, selection, send])
    style_title = w.HTML(value="<i>Selecione uma layer para ajustar paleta/legenda.</i>")
    style_colors = w.Text(description="Cores:", placeholder="hex separadas por vírgula", continuous_update=True, layout=w.Layout(width="95%"))
    style_labels = w.Text(description="Rótulos:", placeholder="um rótulo por cor", continuous_update=True, layout=w.Layout(width="95%"))
    style_min = w.FloatText(value=0, description="Mín:")
    style_max = w.FloatText(value=1, description="Máx:")
    style_panel = w.VBox([w.HTML("<h4>Ajustar paleta e legenda</h4><p>Selecione uma layer do inventário para revisar ou ajustar detalhes.</p>"),
                          style_title, style_colors, style_labels, w.HBox([style_min, style_max])],
                         layout=w.Layout(display="none"))

    def _active_reference_layer():
        for _, layer, check in product_checks:
            if check.value and getattr(check, "_reference", False):
                return layer
        return None

    def _refresh_style_panel(*_):
        layer = _active_reference_layer()
        if layer is None:
            style_panel.layout.display = "none"
            return
        viz = layer.get("visualization") or {}
        legend = layer.get("legend") or []
        style_title.value = f"<b>{escape(layer['name'])}</b> — {layer.get('theme', '')} · {layer.get('collection', '')}"
        style_colors.value = ",".join(viz.get("palette", []))
        style_labels.value = ",".join(str(x.get("label", "")) for x in legend)
        style_min.value = viz.get("min", 0)
        style_max.value = viz.get("max", 1)
        style_panel.layout.display = ""

    def _apply_style_fields(*_):
        layer = _active_reference_layer()
        if layer is None:
            return
        colors = [s.strip().lstrip("#") for s in style_colors.value.split(",") if s.strip()]
        labels = [s.strip() for s in style_labels.value.split(",")]
        if len(colors) >= 2 and len(labels) == len(colors) and all(re.fullmatch(r"(?:[0-9a-fA-F]{6})", c) for c in colors):
            viz = layer.setdefault("visualization", {})
            viz["palette"] = colors
            viz["min"] = style_min.value
            viz["max"] = style_max.value
            layer["legend"] = [{"label": label, "color": color} for label, color in zip(labels, colors)]

    for _, _, check in product_checks:
        check.observe(_refresh_style_panel, names="value")
    style_colors.observe(_apply_style_fields, names="value")
    style_labels.observe(_apply_style_fields, names="value")
    style_min.observe(_apply_style_fields, names="value")
    style_max.observe(_apply_style_fields, names="value")
    catalog_panel.children = (*catalog_panel.children, style_panel)
    navigation = w.ToggleButtons(options=["Produtos", "Territórios", "Saídas e destino", "Resultados"], description="Guia:")
    panels = [catalog_panel, territories_panel, output_panel, delivery_panel]
    def change_panel(*_):
        for i, panel in enumerate(panels):
            panel.layout.display = "" if navigation.value == navigation.options[i] else "none"
    navigation.observe(change_panel, names="value")
    change_panel()
    ui = w.VBox([initiative, navigation, *panels, run_button, sprout, status, overall, progress, log])
    ui.start_batch = start_batch
    ui.initiative = initiative
    ui.navigation = navigation
    ui.theme_nav = theme_nav
    ui.product_checks = product_checks
    ui.territory_checks = territory_checks
    ui.status_html = status
    ui.sprout = sprout
    ui.overall = overall
    ui.download_progress = progress
    ui.log_output = log
    ui.run_button = run_button
    ui.style_panel = style_panel
    ui.style_colors = style_colors
    ui.style_labels = style_labels
    ui.style_min = style_min
    ui.style_max = style_max
    ui.municipality_text = municipality_text
    ui.municipality_results = municipality_results
    ui.municipality_add = municipality_add
    ui.municipality_combine = municipality_combine
    ui.territory_type = territory_type
    ui.municipality_search = run_municipality_search
    display(ui)
    return ui
