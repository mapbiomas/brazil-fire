"""Seleção hierárquica e progresso do lote no Colab."""
from datetime import datetime
from pathlib import Path
from threading import Thread
from time import monotonic
from html import escape
import ipywidgets as w
from IPython.display import display
from .engine import generate, validate
from .storage import list_artifacts, copy_artifacts
from .presets import LAYERS, TERRITORIES


def _duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def show_ui(layers, territories, output_root="/content/mapbiomas_gif_factory_output", project=None):
    validate(layers, territories)
    validate(LAYERS, TERRITORIES)
    log = w.Output(layout=w.Layout(border="1px solid #ddd", max_height="250px", overflow="auto"))
    status = w.HTML(value="<b>Pronto para iniciar.</b>")
    progress = w.IntProgress(value=0, min=0, max=1, description="Frames:", layout=w.Layout(width="95%"))
    run = w.Button(description="Gerar lote", button_style="primary")
    selection = w.SelectMultiple(description="Arquivos:", layout=w.Layout(width="95%", height="180px"))
    drive = w.Text(value="/content/drive/MyDrive/mapbiomas_gif_factory", description="Drive:", layout=w.Layout(width="95%"))
    auto = w.Checkbox(value=False, description="Enviar automaticamente após cada combinação")
    kinds = {k: w.Checkbox(value=(k == "gif"), description=label) for k, label in [("gif", "GIFs"), ("colagem", "Colagens"), ("brutos", "Frames puros"), ("preparados", "Frames clean"), ("finais", "Frames mapas completos")]}

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
                check = w.Checkbox(value=False, description=layer["name"], indent=False, layout=w.Layout(width="95%"))
                product_checks.append((origin, layer, check))
                checks.append(check)
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
        run.disabled = True
        destination, automatic = drive.value, auto.value
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        status.value = "<b>▶ INICIANDO</b> — preparando lote..."
        progress.value = 0
        log.clear_output()
        def worker():
            started = monotonic()
            total = len(chosen_layers) * len(chosen_territories)
            ok = failed = 0
            try:
                combos = [(origin, layer, territory) for origin, layer in chosen_layers for territory in chosen_territories]
                for number, (origin, layer, territory) in enumerate(combos, 1):
                    header = f"{origin} / {layer.get('theme', 'Fogo' if origin == 'Brasil' else 'Personalizado')} / {layer.get('collection', 'Coleção 5' if origin == 'Brasil' else 'Sem coleção')} / {layer['name']} / {territory['name']}"
                    progress.max = max(1, len(layer['bands']))
                    progress.value = 0
                    status.value = f"<b>◉ PREPARANDO {number} de {total}</b><br>{escape(header)}<br>Tempo: {_duration(monotonic()-started)} · previsão: calculando…"
                    log.append_stdout(f"INICIANDO {number}/{total}: {header}\n")
                    frame_start = monotonic()
                    def event(kind, *args):
                        if kind in ("frame_start", "frame_done"):
                            done, count, band = args
                            progress.max = max(1, count)
                            progress.value = done
                            elapsed_frames = monotonic() - frame_start
                            estimate = f"~{_duration(elapsed_frames / done * (count-done))} para download" if done else "calculando…"
                            current = min(done + (kind == "frame_start"), count)
                            status.value = f"<b>◉ {number} de {total}</b><br>{escape(header)}<br>Frame {current} de {count} — {escape(str(band))}<br>Decorrido: {_duration(monotonic()-started)} · restante estimado: {estimate}"
                        elif kind == "stage":
                            status.value = f"<b>◉ {number} de {total}</b><br>{escape(header)}<br>{escape(str(args[0]))} · {_duration(monotonic()-started)}"
                            log.append_stdout(f"{args[0]}...\n")
                    def after(path):
                        if automatic:
                            try:
                                for _, state, detail in publish([path], destination):
                                    log.append_stdout(f"Drive: {state}: {path} {detail}\n")
                            except Exception as exc:
                                log.append_stdout(f"Drive falhou; arquivo local preservado: {exc}\n")
                    try:
                        combo_root = Path(output_root) / run_id / f"{number:03d}_{'brasil' if origin == 'Brasil' else 'personalizado'}"
                        files = generate(layer, territory, combo_root, selected_kinds, after,
                                         project=project, on_progress=event)
                        ok += 1
                        log.append_stdout(f"FINALIZADO {number}/{total}: {len(files)} arquivos em /content\n")
                    except Exception as exc:
                        failed += 1
                        log.append_stdout(f"FALHA {number}/{total}: {exc}\n")
                status.value = f"<b>✓ LOTE FINALIZADO</b> · {ok} sucesso(s), {failed} falha(s) · {_duration(monotonic()-started)}"
                update_files()
            finally:
                run.disabled = False
        Thread(target=worker, daemon=True).start()
    run.on_click(on_run)
    sections = w.Accordion(children=[w.VBox(territory_boxes), w.VBox(list(kinds.values()) + [auto, drive]), w.VBox([refresh, selection, send])], selected_index=None)
    for i, title in enumerate(["Territórios", "Saídas e destino", "Arquivos gerados e envio posterior"]):
        sections.set_title(i, title)
    update_files()
    ui = w.VBox([w.HTML("<h3>Catálogo de camadas</h3>"), tabs, sections, run, status, progress, log])
    display(ui)
    return ui
