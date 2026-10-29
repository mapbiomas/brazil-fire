"""Seleção hierárquica e progresso do lote no Colab."""
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from html import escape
import ipywidgets as w
from IPython.display import display
from .engine import generate, validate
from .storage import list_artifacts, copy_artifacts
from .presets import LAYERS, TERRITORIES
from .layout import combination_path, ensure_unique_paths, safe_name
from .progress import ProgressState


def _duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def show_ui(layers, territories, output_root="/content/mapbiomas_gif_factory_output", project=None):
    validate(layers, territories)
    validate(LAYERS, TERRITORIES)
    log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="250px", overflow="auto"))
    status = w.HTML(value="<b>Pronto para iniciar.</b>")
    overall = w.IntProgress(value=0, min=0, max=1, description="Lote:", layout=w.Layout(width="95%"))
    progress = w.IntProgress(value=0, min=0, max=1, description="Download:", layout=w.Layout(width="95%", display="none"))
    run = w.Button(description="Gerar lote", button_style="primary")
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
    auto = w.Checkbox(value=False, description="Enviar automaticamente após cada combinação")
    delivery = w.Text(value="Resultados", description="Entrega:", layout=w.Layout(width="95%"))
    year_start = w.IntText(value=0, description="Ano inicial:")
    year_end = w.IntText(value=0, description="Ano final:")
    kinds = {k: w.Checkbox(value=(k == "gif"), description=label) for k, label in [("gif", "GIFs"), ("colagem", "Colagens"), ("brutos", "Frames: mapa básico"), ("preparados", "Frames: mapa com ano e escala"), ("finais", "Frames: mapa completo")]}

    # Preset catalog and user-provided layers are intentionally separate.
    catalog = [("Brasil", LAYERS), ("Personalizado", layers)]
    product_checks = []
    tab_contents = []
    for origin, entries in catalog:
        groups = {}
        for layer in entries:
            theme = layer.get("theme", "Fogo" if origin == "Brasil" else "Outros")
            collection = layer.get("collection", "Coleção 5" if origin == "Brasil" else "Sem coleção")
            groups.setdefault((theme, collection), []).append(layer)
        children, titles = [], []
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
            children.append(w.VBox(checks))
            titles.append(f"{theme} / {collection}")
        accordion = w.Accordion(children=children, selected_index=None)
        for i, title in enumerate(titles):
            accordion.set_title(i, title)
        tab_contents.append(accordion if children else w.HTML("Nenhuma camada configurada."))
    if product_checks:
        product_checks[0][2].value = True
    tabs = w.Tab(children=tab_contents)
    for i, (name, _) in enumerate(catalog):
        tabs.set_title(i, name)

    # Notebook territories remain customizable and separate from preset territories.
    territory_checks = []
    territory_boxes = []
    for name, entries in [("Biomas do catálogo", TERRITORIES), ("Personalizados", territories)]:
        checks = []
        for territory in entries:
            check = w.Checkbox(value=(name == "Biomas do catálogo" and not checks), description=territory["name"], indent=False)
            territory_checks.append((name, territory, check))
            checks.append(check)
        territory_boxes.append(w.VBox([w.HTML(f"<b>{name}</b>"), *checks]))

    def update_files(*_):
        files = list_artifacts(output_root)
        selection.options = [(str(p.relative_to(output_root)), str(p)) for p in files]

    def publish(files, destination):
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Monte o Google Drive antes de enviar os arquivos.")
        return copy_artifacts(files, output_root, destination)

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
        chosen_layers = [(origin, item) for origin, item, check in product_checks if check.value]
        chosen_territories = [item for _, item, check in territory_checks if check.value]
        selected_kinds = {k for k, checkbox in kinds.items() if checkbox.value}
        if not chosen_layers or not chosen_territories or not selected_kinds:
            status.value = "<b>Selecione produto, território e saída.</b>"
            return
        first_year, last_year = year_start.value, year_end.value
        if first_year < 0 or last_year < 0 or (first_year and last_year and first_year > last_year):
            status.value = "<b>Período inválido.</b> Use 0 para não limitar o ano."
            return
        combos = [(origin, layer, territory) for origin, layer in chosen_layers for territory in chosen_territories]
        try:
            ensure_unique_paths(combos)
        except ValueError as exc:
            status.value = f"<b>Seleção inválida:</b> {escape(str(exc))}"
            return
        run.disabled = True
        sprout.layout.display = ""
        destination, automatic = drive.value, auto.value
        run_id = f"{safe_name(delivery.value, 'Resultados')}_{datetime.now():%Y-%m-%d_%H%M%S_%f}"
        delivery_root = Path(output_root) / run_id
        status.value = "<b>▶ INICIANDO</b> — preparando lote..."
        progress.value = 0
        overall.value = 0
        overall.max = len(combos)
        log.clear_output()
        def worker():
            total = len(chosen_layers) * len(chosen_territories)
            state = ProgressState(total)
            stop_clock = Event()
            ok = failed = drive_failures = 0
            completed = []
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
            ticker.start()
            try:
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
                    def after(path):
                        nonlocal drive_failures
                        if automatic:
                            state.update("stage", f"Enviando ao Drive: {path.name}")
                            render()
                            try:
                                for _, copy_status, detail in publish([path], destination):
                                    if copy_status != "copiado":
                                        drive_failures += 1
                                    log.append_stdout(f"Drive: {copy_status}: {path} {detail}\n")
                            except Exception as exc:
                                drive_failures += 1
                                log.append_stdout(f"Drive falhou; arquivo local preservado: {exc}\n")
                            state.update("stage", "Organizando arquivos selecionados")
                            render()
                    try:
                        combo_root = delivery_root / combination_path(origin, layer, territory)
                        configured_layer = dict(layer, start_year=first_year or None, end_year=last_year or None)
                        files = generate(configured_layer, territory, combo_root, selected_kinds, after,
                                         project=project, on_progress=event)
                        ok += 1
                        completed.append((origin, layer, territory, list(discovered)))
                        log.append_stdout(f"FINALIZADO {number}/{total}: {len(files)} arquivos em /content\n")
                    except Exception as exc:
                        failed += 1
                        log.append_stdout(f"FALHA {number}/{total}: {exc}\n")
                    finally:
                        state.finish_combo()
                        render()
                if completed:
                    readme = delivery_root / "LEIA-ME.txt"
                    lines = ["RESULTADOS CARTOGRÁFICOS MAPBIOMAS", "", f"Entrega: {run_id}",
                             "Dados: MapBiomas; verifique as fontes e condições de uso de cada coleção.",
                             "A lista abaixo descreve a execução; um envio seletivo pode conter apenas parte dos arquivos.",
                             "", "Produtos e territórios concluídos:"]
                    for origin, layer, territory, bands in completed:
                        lines.append(f"- {origin} / {layer.get('theme', 'Sem tema')} / {layer.get('collection', 'Sem coleção')} / {layer['name']} / {territory['name']}")
                        lines.append(f"  Períodos/bandas: {', '.join(bands)}")
                    lines.extend(["", "Frames, quando disponíveis:",
                                  "- Mapa_basico: imagem visualizada do Earth Engine, redimensionada.",
                                  "- Mapa_com_ano_e_escala: mapa preparado com ano e elementos cartográficos aplicáveis.",
                                  "- Mapa_completo: frame com título e legenda usado no GIF.",
                                  "", "Apenas as saídas escolhidas na geração estão incluídas."])
                    with readme.open("x", encoding="utf-8") as file:
                        file.write("\n".join(lines) + "\n")
                    if automatic:
                        state.update("stage", "Enviando LEIA-ME ao Drive")
                        render()
                        try:
                            for _, copy_status, detail in publish([readme], destination):
                                if copy_status != "copiado":
                                    drive_failures += 1
                                log.append_stdout(f"Drive: {copy_status}: {readme} {detail}\n")
                        except Exception as exc:
                            drive_failures += 1
                            log.append_stdout(f"Drive falhou; arquivo local preservado: {exc}\n")
                    log.append_stdout(f"Pasta de entrega: {delivery_root}\n")
                update_files()
                stop_clock.set()
                ticker.join()
                title = "✓ LOTE CONCLUÍDO" if not failed else "⚠ CONCLUÍDO COM FALHAS" if ok else "✕ LOTE FALHOU"
                drive_status = (f"{drive_failures} envio(s) não concluído(s)" if drive_failures else "cópias solicitadas concluídas") if automatic else "envio automático não solicitado"
                status.value = f"<b>{title}</b> · {ok} sucesso(s), {failed} falha(s)<br>Drive: {drive_status}<br>Tempo total: {_duration(state.snapshot()['elapsed'])}"
            except Exception as exc:
                stop_clock.set()
                ticker.join()
                status.value = f"<b>Erro ao finalizar o lote:</b> {escape(str(exc))}"
                log.append_stdout(f"Erro ao finalizar lote: {exc}\n")
            finally:
                stop_clock.set()
                run.disabled = False
                sprout.layout.display = "none"
        Thread(target=worker, daemon=True).start()
    run.on_click(on_run)
    sections = w.Accordion(children=[w.VBox(territory_boxes), w.VBox([delivery, w.HTML("<b>Período</b> — 0 = todos os anos"), year_start, year_end, *kinds.values(), auto, drive]), w.VBox([refresh, selection, send])], selected_index=None)
    for i, title in enumerate(["Territórios", "Saídas e destino", "Arquivos gerados e envio posterior"]):
        sections.set_title(i, title)
    update_files()
    controls = w.HBox([run, sprout], layout=w.Layout(align_items="center"))
    ui = w.VBox([w.HTML("<h3>Catálogo de camadas</h3>"), tabs, sections, controls, status, overall, progress, log])
    display(ui)
    return ui
