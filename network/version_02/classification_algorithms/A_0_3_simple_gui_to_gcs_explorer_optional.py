# last_update: '2026/01/26', github:'mapbiomas/chile-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms - Step A_0_3_simple_gui_to_gcs_explorer_optional.py
### Step A_0_3 - Optional step to visualize and navigate files/folders in Google Cloud Storage (GCS)
# Changes (2026/01/26):
# - Implemented a folder+file explorer (supports subfolders) using fs.ls(..., detail=True)
# - Added navigation state (current_path), breadcrumb display, and "Back" + "Refresh" controls
# - Fixed ipywidgets.Select auto-selection side effects (prevent auto-enter into first folder)
#   by suppressing events during render and clearing selection (index=None)

import gcsfs
import ipywidgets as widgets
from ipywidgets import HBox, VBox
from IPython.display import display, clear_output

# ----------------------------
# Config / FS
# ----------------------------
fs = gcsfs.GCSFileSystem(project=bucket_name)

base_folder = "mapbiomas-fire/sudamerica/"  # seu "root" de países

def _ensure_dir(path: str) -> str:
    return path if path.endswith("/") else path + "/"

def _basename(name: str) -> str:
    # name vem tipo: "mapbiomas-fire/sudamerica/ARG/" ou ".../arquivo.tif"
    name = name[:-1] if name.endswith("/") else name
    return name.split("/")[-1]

def list_countries(base_folder: str):
    fs.invalidate_cache()
    items = fs.ls(_ensure_dir(base_folder), detail=True)
    # Em alguns buckets, países são diretórios; filtramos por type
    countries = sorted([_basename(i["name"]) for i in items if i.get("type") == "directory"])
    return countries

def list_dir(path: str):
    """
    Retorna (dirs, files) do path atual, como listas de nomes (apenas o último componente).
    """
    fs.invalidate_cache()
    items = fs.ls(_ensure_dir(path), detail=True)

    dirs = []
    files = []
    for i in items:
        t = i.get("type")
        name = i.get("name", "")
        if t == "directory":
            dirs.append(_basename(name))
        elif t == "file":
            files.append(_basename(name))

    return sorted(dirs), sorted(files)

# ----------------------------
# UI widgets
# ----------------------------
dropdown_countries = widgets.Dropdown(
    options=list_countries(base_folder),
    description="Country:",
    disabled=False,
    layout=widgets.Layout(width="360px")
)

btn_up = widgets.Button(description="↑ Voltar", layout=widgets.Layout(width="110px"))
btn_refresh = widgets.Button(description="⟳ Atualizar", layout=widgets.Layout(width="130px"))

path_html = widgets.HTML(value="")

dirs_select = widgets.Select(
    options=[],
    description="Pastas:",
    rows=14,
    layout=widgets.Layout(width="380px")
)

files_select = widgets.Select(
    options=[],
    description="Arquivos:",
    rows=14,
    layout=widgets.Layout(width="380px")
)

details_out = widgets.Output(layout={"border": "1px solid #999", "padding": "8px", "width": "780px"})

# Estado de navegação
root_path = None
current_path = None
_suppress_events = True


def render():
    global current_path, root_path, _suppress_events

    if current_path is None:
        return

    path_html.value = f"<b>Path:</b> <code>{current_path}</code>"

    try:
        dirs, files = list_dir(current_path)
    except Exception as e:
        dirs, files = [], []
        with details_out:
            clear_output()
            print("Erro ao listar diretório:", e)

    # BLOQUEIA callbacks enquanto atualiza a UI
    _suppress_events = True
    try:
        dirs_select.options = dirs
        files_select.options = files

        # Impede auto-seleção do primeiro item (e portanto impede entrar sozinho)
        dirs_select.index = None
        files_select.index = None
    finally:
        _suppress_events = False

    btn_up.disabled = (current_path == root_path)


def set_country(country: str):
    """
    Define o root da navegação como a pasta do país e entra nela.
    """
    global root_path, current_path
    root_path = _ensure_dir(f"{base_folder}{country}")
    current_path = root_path
    with details_out:
        clear_output()
        print(f"Selecionado: {country}")
    render()

def go_up(_=None):
    """
    Sobe um nível (até root_path).
    """
    global current_path, root_path
    if current_path is None or root_path is None:
        return
    if current_path == root_path:
        return

    # Remove barra final e sobe
    p = current_path[:-1] if current_path.endswith("/") else current_path
    parent = "/".join(p.split("/")[:-1]) + "/"

    # Não permitir subir acima do root
    if len(parent) < len(root_path) or not parent.startswith(root_path):
        current_path = root_path
    else:
        current_path = parent

    with details_out:
        clear_output()
        print("Voltando para:", current_path)
    render()

def enter_dir(change):
    global current_path, _suppress_events
    if _suppress_events:
        return

    folder = change["new"]
    if not folder:
        return

    current_path = _ensure_dir(f"{current_path}{folder}")
    with details_out:
        clear_output()
        print("Entrando em:", current_path)
    render()


def show_file_details(change):
    global _suppress_events
    if _suppress_events:
        return

    filename = change["new"]
    if not filename:
        return

    full_path = f"{current_path}{filename}"
    with details_out:
        clear_output()
        print("Arquivo selecionado:")
        print(full_path)
        try:
            info = fs.info(full_path)
            print("\nMetadados:")
            for k in ["size", "updated", "timeCreated", "crc32c", "md5Hash", "contentType"]:
                if k in info:
                    print(f"- {k}: {info[k]}")
        except Exception:
            print("\n(Não foi possível obter metadados via fs.info)")

def refresh(_=None):
    with details_out:
        clear_output()
        print("Atualizando listagem...")
    render()

# ----------------------------
# Bindings
# ----------------------------
dropdown_countries.observe(lambda ch: set_country(ch["new"]), names="value")
btn_up.on_click(go_up)
btn_refresh.on_click(refresh)
dirs_select.observe(enter_dir, names="value")
files_select.observe(show_file_details, names="value")

# ----------------------------
# Layout / Display
# ----------------------------
top_bar = HBox([dropdown_countries, btn_up, btn_refresh])
lists = HBox([dirs_select, files_select])
ui = VBox([top_bar, path_html, lists, details_out])

display(ui)

# Inicializa no primeiro país, se existir
if dropdown_countries.options:
    set_country(dropdown_countries.value)
