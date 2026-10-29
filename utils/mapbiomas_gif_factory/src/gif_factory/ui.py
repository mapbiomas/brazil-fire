"""Interface simples para seleção e envio posterior."""
from datetime import datetime
from pathlib import Path
import ipywidgets as w
from IPython.display import display, clear_output
from .engine import generate, validate
from .storage import list_artifacts, copy_artifacts
from .presets import LAYERS, TERRITORIES


def show_ui(layers, territories, output_root="/content/mapbiomas_gif_factory_output"):
    validate(layers, territories)
    source = w.ToggleButtons(options=[("Arrays do notebook", "custom"), ("Exemplo Fogo 5 / biomas (experimental)", "preset")], description="Origem:")
    layer_select = w.SelectMultiple(description="Camadas:", layout=w.Layout(width="95%", height="140px"))
    territory_select = w.SelectMultiple(description="Territórios:", layout=w.Layout(width="95%", height="140px"))
    kinds = {k: w.Checkbox(value=(k == "gif"), description=label) for k, label in [("gif", "GIF"), ("colagem", "Colagem"), ("brutos", "Frames brutos"), ("preparados", "Frames preparados"), ("finais", "Frames finais")]}
    auto = w.Checkbox(value=False, description="Copiar automaticamente para o Drive")
    drive = w.Text(value="/content/drive/MyDrive/mapbiomas_gif_factory", description="Drive:", layout=w.Layout(width="95%"))
    selection = w.SelectMultiple(description="Arquivos:", layout=w.Layout(width="95%", height="200px"))
    log = w.Output()
    run = w.Button(description="Gerar lote", button_style="primary")
    refresh = w.Button(description="Atualizar lista")
    send = w.Button(description="Enviar selecionados")

    def selected_catalog():
        return (LAYERS, TERRITORIES) if source.value == "preset" else (layers, territories)

    def update_catalog(*_):
        ls, ts = selected_catalog()
        layer_select.options = [(x["name"], x["id"]) for x in ls]
        territory_select.options = [(x["name"], x["id"]) for x in ts]
        layer_select.value = tuple(x["id"] for x in ls[:1])
        territory_select.value = tuple(x["id"] for x in ts[:1])
    source.observe(update_catalog, names="value")
    update_catalog()

    def update_files(*_):
        files = list_artifacts(output_root)
        selection.options = [(str(p.relative_to(output_root)), str(p)) for p in files]
    refresh.on_click(update_files)

    def publish(files):
        if not Path("/content/drive/MyDrive").is_dir():
            raise RuntimeError("Monte o Google Drive antes de enviar os arquivos.")
        for path, status, detail in copy_artifacts(files, output_root, drive.value):
            print(f"{status}: {path} {detail}")

    def on_send(_):
        with log:
            try:
                if not selection.value:
                    raise ValueError("Selecione ao menos um arquivo.")
                publish(selection.value)
            except Exception as exc:
                print(f"Erro no envio: {exc}")
    send.on_click(on_send)

    def on_run(_):
        with log:
            clear_output()
            ls, ts = selected_catalog()
            chosen_layers = [x for x in ls if x["id"] in layer_select.value]
            chosen_territories = [x for x in ts if x["id"] in territory_select.value]
            selected_kinds = {k for k, checkbox in kinds.items() if checkbox.value}
            if not chosen_layers or not chosen_territories or not selected_kinds:
                print("Selecione pelo menos uma camada, um território e uma saída.")
                return
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            print(f"Execução {run_id}: {len(chosen_layers) * len(chosen_territories)} combinações; saídas: {', '.join(sorted(selected_kinds))}")
            def after(path):
                if auto.value:
                    try:
                        publish([path])
                    except Exception as exc:
                        print(f"Falha no envio; arquivo local preservado: {exc}")
            for layer in chosen_layers:
                for territory in chosen_territories:
                    print(f"Gerando {layer['name']} / {territory['name']}...")
                    try:
                        generate(layer, territory, Path(output_root) / run_id, selected_kinds, after)
                        print("Concluído.")
                    except Exception as exc:
                        print(f"Falha: {exc}")
            update_files()
    run.on_click(on_run)
    tabs = w.Tab(children=[w.VBox([source, layer_select]), w.VBox([territory_select]), w.VBox(list(kinds.values()) + [auto, drive]), w.VBox([refresh, selection, send]), w.VBox([run, log])])
    for index, title in enumerate(["Camadas e presets", "Territórios", "Saídas", "Enviar ao Drive", "Execução"]):
        tabs.set_title(index, title)
    update_files()
    display(tabs)
    return tabs
