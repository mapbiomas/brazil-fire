"""Seleção hierárquica e progresso do lote no Colab."""
from datetime import datetime
from pathlib import Path
from threading import Thread, Event, Lock
from html import escape
from copy import deepcopy
import json
import ipywidgets as w
from IPython.display import display
from .engine import generate, validate, discover, collection_variables
from .storage import list_artifacts
from .presets import LAYERS, TERRITORIES
from .layout import combination_path, ensure_unique_paths
from .progress import ProgressState
from .library import assess, fingerprint, ingest, variant_for, sync_selected
from .territory_search import (TERRITORY_SEARCH_SOURCES, search_configured,
                               search_municipalities, territory as searched_territory)
from .reference_catalog import INITIATIVES, experimental_layer


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
    initiative = w.Dropdown(options=["Brasil", "Peru"] + [name for name in INITIATIVES.values() if name not in ("Brasil", "Peru")] + ["Personalizado"], value="Brasil", description="Iniciativa:")
    theme_nav = w.ToggleButtons(options=[], description="Tema:")

    # Catálogo de presets e camadas do usuário ficam separados por iniciativa.
    catalog = [(country, [x for x in LAYERS if x.get("initiative", "Brasil") == country])
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

    territory_type = w.Dropdown(description="Tipo:", options=list(TERRITORY_SEARCH_SOURCES))
    municipality_text = w.Text(description="Município:", placeholder="Ex.: Petrolina", layout=w.Layout(width="60%"))
    municipality_button = w.Button(description="Buscar", icon="search")
    municipality_results = w.SelectMultiple(description="Encontrados:", rows=5, layout=w.Layout(width="95%"))
    municipality_combine = w.Checkbox(value=False, description="Combinar selecionados em um recorte")
    municipality_add = w.Button(description="Adicionar", button_style="primary")
    municipality_log = w.Output()
    municipality_found = []

    def find_municipalities(_):
        with municipality_log:
            municipality_log.clear_output()
            try:
                import ee
                municipality_found[:] = (search_municipalities(ee, municipality_text.value)
                                         if territory_type.value == "Município"
                                         else search_configured(ee, territory_type.value, municipality_text.value))
                municipality_results.options = [(r["name"], i) for i, r in enumerate(municipality_found)]
                print(f"{len(municipality_found)} resultado(s). A busca diferencia maiúsculas; limite de 30.")
            except Exception as exc:
                municipality_found.clear()
                municipality_results.options = []
                print(f"Não foi possível buscar municípios: {exc}")

    def add_municipalities(_):
        with municipality_log:
            try:
                selected = [municipality_found[i] for i in municipality_results.value]
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
                print("Adicionado em Personalizados; desmarque para remover da seleção.")
            except Exception as exc:
                print(f"Não foi possível adicionar: {exc}")

    municipality_button.on_click(find_municipalities)
    municipality_add.on_click(add_municipalities)
    def territory_type_changed(change):
        municipality_found.clear()
        municipality_results.options = []
        municipality_text.value = ""
        municipality_text.description = "Nome:"
        municipality_text.placeholder = "Ex.: Petrolina" if change["new"] == "Município" else "Digite o nome"
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
    municipality_panel = w.VBox([w.HTML("<b>Buscar território</b><p>Municípios: FAO GAUL 2025 (não é a malha oficial IBGE). UC e TI aguardam conexão das fontes.</p>"),
                                territory_type, w.HBox([municipality_text, municipality_button]), municipality_results,
                                municipality_combine, municipality_add, municipality_log])
    territories_panel = w.VBox([w.HTML("<h3>Territórios</h3>"), municipality_panel, *territory_boxes])
    output_panel = w.VBox([w.HTML("<h3>Saídas e destino</h3>"), *kinds.values(), auto, drive, check_status, availability])
    delivery_panel = w.VBox([w.HTML("<h3>Arquivos gerados e envio posterior</h3>"), refresh, selection, send])
    inventory_path = Path(__file__).with_name("reference_inventory.json")
    records = json.loads(inventory_path.read_text(encoding="utf-8"))["records"] if inventory_path.is_file() else []
    ref_theme = w.Dropdown(description="Theme:")
    ref_collection = w.Dropdown(description="Collection:")
    ref_product = w.Select(description="Layer:", rows=6, layout=w.Layout(width="95%"))
    ref_asset = w.Textarea(description="Asset:", disabled=True, layout=w.Layout(width="95%", height="65px"))
    ref_min = w.FloatText(value=0, description="Min:")
    ref_max = w.FloatText(value=1, description="Max:")
    ref_colors = w.Text(description="Colors:", placeholder="ffffff,800000", layout=w.Layout(width="95%"))
    ref_labels = w.Text(description="Labels:", placeholder="classe 0,classe 1", layout=w.Layout(width="95%"))
    ref_confirm = w.Checkbox(value=False, description="Confirmei asset, bandas, legenda e tipo de imagem / I verified asset, bands, legend and image type / Confirmé asset, bandas, leyenda y tipo")
    ref_check = w.Button(description="Verificar opções", layout=w.Layout(width="160px"))
    ref_variable = w.Dropdown(description="Variável:", options=[], layout=w.Layout(display="none"))
    ref_add = w.Button(description="Adicionar / Add / Añadir", layout=w.Layout(width="220px"))
    ref_log = w.Output()
    ref_checks = w.VBox([])
    code_by_country = {name: code for code, name in INITIATIVES.items()}

    def ref_product_changed(*_):
        index = ref_product.value
        ref_asset.value = str(records[index].get("asset") or "Asset indisponível / unavailable / no disponible") if index is not None else ""
        ref_confirm.value = False
        ref_variable.options = []
        ref_variable.layout.display = "none"

    def check_reference_options(_):
        with ref_log:
            ref_log.clear_output()
            try:
                if ref_product.value is None or not records[ref_product.value].get("asset"):
                    raise ValueError("Asset indisponível para verificação.")
                options = collection_variables(records[ref_product.value]["asset"])
                ref_variable.options = [("Selecione / Select / Seleccione", None), *options] if options else []
                ref_variable.value = None
                ref_variable.layout.display = "" if options else "none"
                print("Escolha uma variável." if options else "Nenhuma variável adicional precisa ser escolhida.")
            except Exception as exc:
                ref_variable.options = []
                ref_variable.layout.display = "none"
                print(f"Não foi possível verificar a layer: {exc}")

    def ref_collection_changed(*_):
        code = code_by_country.get(initiative.value)
        ref_product.options = [(str(r["product"]), i) for i, r in enumerate(records)
                               if r["initiative"] == code and r["theme"] == ref_theme.value
                               and r["collection"] == ref_collection.value]
        if ref_product.options and ref_product.value is None:
            ref_product.value = ref_product.options[0][1]
        ref_product_changed()

    def ref_theme_changed(*_):
        code = code_by_country.get(initiative.value)
        ref_collection.options = sorted({r["collection"] for r in records if r["initiative"] == code and r["theme"] == ref_theme.value})
        if ref_collection.options and ref_collection.value is None:
            ref_collection.value = ref_collection.options[0]
        ref_collection_changed()

    def ref_initiative_changed(*_):
        code = code_by_country.get(initiative.value)
        ref_theme.options = sorted({r["theme"] for r in records if r["initiative"] == code})
        if ref_theme.options and ref_theme.value is None:
            ref_theme.value = ref_theme.options[0]
        ref_theme_changed()
        for origin, _, check in product_checks:
            if getattr(check, "_reference", False):
                check.layout.display = "" if origin == initiative.value else "none"

    def add_reference_layer(_):
        with ref_log:
            ref_log.clear_output()
            try:
                if ref_product.value is None or not ref_confirm.value:
                    raise ValueError("Selecione a layer e confirme os dados / Select and confirm / Seleccione y confirme.")
                if ref_variable.options and ref_variable.value is None:
                    raise ValueError("Selecione a variável / Select a variable / Seleccione una variable.")
                layer = experimental_layer(records[ref_product.value], ref_colors.value, ref_labels.value, ref_min.value, ref_max.value)
                if ref_variable.value is not None:
                    layer["collection_member"] = ref_variable.value
                validate([layer], [])
                previous = next((check for origin, item, check in product_checks if origin == initiative.value and item["id"] == layer["id"]), None)
                if previous is not None:
                    raise ValueError("Layer já adicionada; reabra a interface para alterar a configuração / Already added; reopen UI to change / Ya añadida; reabra la interfaz.")
                check = w.Checkbox(value=True, description=layer["name"] + " [experimental]", indent=False, layout=w.Layout(width="95%"))
                check._reference = True
                product_checks.append((initiative.value, layer, check))
                ref_checks.children = (*ref_checks.children, check)
                print("Layer adicionada como experimental / Added as experimental / Añadida como experimental.")
            except Exception as exc:
                print(str(exc))

    ref_theme.observe(ref_theme_changed, names="value")
    ref_collection.observe(ref_collection_changed, names="value")
    ref_product.observe(ref_product_changed, names="value")
    initiative.observe(ref_initiative_changed, names="value")
    ref_add.on_click(add_reference_layer)
    ref_check.on_click(check_reference_options)
    ref_initiative_changed()
    ref_panel = w.VBox([
        w.HTML("<h4>Outras layers / More layers / Más capas</h4><p>Layers sem visualização configurada podem ser adicionadas com legenda própria após conferência. Imagens e coleções temporais simples são reconhecidas automaticamente; classes não consecutivas ou produtos com processor podem precisar de adaptação. / Configure and verify the legend. / Configure y verifique la leyenda.</p>"),
        ref_theme, ref_collection, ref_product, ref_asset, ref_check, ref_variable, w.HBox([ref_min, ref_max]), ref_colors, ref_labels,
        ref_confirm, ref_add, ref_log, ref_checks,
    ])
    catalog_panel.children = (*catalog_panel.children, ref_panel)
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
    ui.reference_theme = ref_theme
    ui.reference_collection = ref_collection
    ui.reference_product = ref_product
    ui.reference_colors = ref_colors
    ui.reference_labels = ref_labels
    ui.reference_confirm = ref_confirm
    ui.reference_add = ref_add
    ui.reference_check = ref_check
    ui.reference_variable = ref_variable
    ui.municipality_text = municipality_text
    ui.municipality_button = municipality_button
    ui.municipality_results = municipality_results
    ui.municipality_add = municipality_add
    ui.municipality_combine = municipality_combine
    ui.territory_type = territory_type
    display(ui)
    return ui
