"""Idiomas da interface e dos mapas.

Os textos ficam em dois CSVs dentro de `gif_factory/textos/`:

- `interface.csv`: tudo o que aparece na tela. Coluna `pt` = texto original (é a chave);
  colunas `en`, `es`, `fr`, `id`, `zh` = traduções.
- `mapas.csv`: o que vai desenhado nos GIFs e quadros (nome do produto, título e classes da
  legenda, temas, textos fixos do pipeline). Coluna `original` = texto como está no catálogo
  (é a chave); colunas `pt` ... `zh` = traduções. `tipo` e `exemplo` só dão contexto.

Célula vazia = usa o original. Assim dá para traduzir aos poucos.

Para atualizar os CSVs depois de mudar a interface ou o catálogo (mantém o que já foi traduzido):

    python -m gif_factory.i18n
"""
import ast
import csv
import sys
from functools import lru_cache
from pathlib import Path

LANGUAGES = [("Português", "pt"), ("English", "en"), ("Español", "es"), ("Français", "fr"),
             ("Bahasa Indonesia", "id"), ("中文 (Mandarim)", "zh")]
CODES = [code for _, code in LANGUAGES]
TEXT_DIR = Path(__file__).resolve().parent / "textos"
INTERFACE_CSV = TEXT_DIR / "interface.csv"
MAPS_CSV = TEXT_DIR / "mapas.csv"
INTERFACE_COLUMNS = ["pt", "en", "es", "fr", "id", "zh", "onde"]
MAPS_COLUMNS = ["original", "tipo", "exemplo", "pt", "en", "es", "fr", "id", "zh"]
_state = {"ui": "pt", "maps": "pt"}

# Textos fixos desenhados pelo pipeline (fora do catálogo).
PIPELINE_TEXTS = ["Composição RGB:", "bandas:", "Sem vegetação", "Cada mancha de vegetação possui uma cor única"]


def code_for(language):
    """'English', 'en', 'EN', 'Inglês'... -> 'en' (padrão: 'pt')."""
    text = str(language or "pt").strip()
    for name, code in LANGUAGES:
        if text.lower() in (code, name.lower()) or text.lower() == name.split(" ")[0].lower():
            return code
    aliases = {"portugues": "pt", "inglês": "en", "ingles": "en", "espanhol": "es", "francês": "fr",
               "frances": "fr", "indonésio": "id", "indonesio": "id", "mandarim": "zh", "chinês": "zh",
               "chines": "zh", "中文": "zh"}
    return aliases.get(text.lower(), "pt")


def set_language(ui=None, maps=None):
    if ui is not None:
        _state["ui"] = code_for(ui)
    if maps is not None:
        _state["maps"] = code_for(maps)


def ui_language():
    return _state["ui"]


def map_language():
    return _state["maps"]


def _read(path, key):
    table = {}
    if not path.exists():
        return table
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            original = row.get(key) or ""
            if original:
                table[original] = {code: (row.get(code) or "").strip() for code in CODES}
    return table


@lru_cache(maxsize=1)
def interface_table():
    return _read(INTERFACE_CSV, "pt")


@lru_cache(maxsize=1)
def maps_table():
    return _read(MAPS_CSV, "original")


def reload():
    interface_table.cache_clear()
    maps_table.cache_clear()


def _(text, **values):
    """Texto da interface no idioma escolhido; `values` preenche {marcadores}."""
    lang = _state["ui"]
    translated = text
    if lang != "pt":
        translated = (interface_table().get(text) or {}).get(lang) or text
    if values:
        try:
            return translated.format(**values)
        except (KeyError, IndexError, ValueError):
            return text.format(**values)
    return translated


def N_(text):
    """Marca um texto para o CSV sem traduzir agora (listas fixas traduzidas na hora de mostrar)."""
    return text


def ui_text(text):
    """Conteúdo do catálogo (nome de mapa, tema, classe) no idioma da interface."""
    return map_text(text, _state["ui"])


def map_text(text, lang=None):
    """Texto de mapa (produto, legenda, classe...) no idioma dos mapas; sem tradução, o original."""
    if text is None or not isinstance(text, str) or not text.strip():
        return text
    lang = lang or _state["maps"]
    return (maps_table().get(text) or {}).get(lang) or text


def translate_visualization(viz, lang=None):
    """Cópia de uma visualização com título, nome e classes traduzidos."""
    viz = dict(viz or {})
    for key in ("label", "name", "legend_label", "unit"):
        if isinstance(viz.get(key), str):
            viz[key] = map_text(viz[key], lang)
    if isinstance(viz.get("discrete_labels"), list):
        viz["discrete_labels"] = [map_text(x, lang) if isinstance(x, str) else x for x in viz["discrete_labels"]]
    if isinstance(viz.get("classes"), list):
        viz["classes"] = [dict(c, label=map_text(c.get("label"), lang)) if isinstance(c, dict) else c
                          for c in viz["classes"]]
    return viz


# ---------------------------------------------------------------------- fontes
CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.ttf",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
]
CJK_FONT_URL = "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"


def cjk_font(download_to=None):
    """Caminho de uma fonte com caracteres chineses (baixa a Noto Sans SC se não houver)."""
    for path in CJK_FONTS:
        if Path(path).exists():
            return path
    target = Path(download_to or Path.home() / ".cache" / "gif_factory" / "NotoSansSC.ttf")
    if target.exists():
        return str(target)
    try:
        import requests
        response = requests.get(CJK_FONT_URL, timeout=120)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return str(target)
    except Exception:
        return None


def prepare_map_rendering(lang=None):
    """Ajusta o pipeline para o idioma dos mapas: textos fixos e, em mandarim, uma fonte CJK."""
    from mapbiomas_data.core import frame_processor as fp
    lang = lang or _state["maps"]
    texts = getattr(fp, "TEXTS", None)
    if isinstance(texts, dict):
        for original in PIPELINE_TEXTS:
            texts[original] = map_text(original, lang)
    if not hasattr(fp, "_DEFAULT_FONT"):
        fp._DEFAULT_FONT = fp.FrameProcessor.FONT_PATH
    if lang == "zh":
        font = cjk_font()
        if font:
            fp.FrameProcessor.FONT_PATH = font
    else:
        fp.FrameProcessor.FONT_PATH = fp._DEFAULT_FONT


# ---------------------------------------------------------------------- exportação dos CSVs
def interface_strings(package_dir=None):
    """[(texto, onde)] de todas as chamadas _("...") do pacote, na ordem em que aparecem."""
    package_dir = Path(package_dir or Path(__file__).resolve().parent)
    found = {}
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("_", "N_")
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                found.setdefault(node.args[0].value, f"{path.name}:{node.lineno}")
    return sorted(found.items(), key=lambda item: (item[1].split(":")[0], int(item[1].split(":")[1])))


def map_strings():
    """[(texto, tipo, exemplo)] dos textos que vão nos mapas, sem repetição."""
    from .catalog import catalog_layers, config_loader
    from .legends import THEME_LABELS
    found = {}

    def add(text, kind, example):
        if isinstance(text, str) and text.strip() and not text.strip().replace(".", "").replace("-", "").isdigit():
            found.setdefault(text.strip(), (kind, example))

    config = config_loader()
    for layer in catalog_layers(("validado", "beta")):
        add(layer.get("name"), "produto", layer["id"])
        add(layer.get("initiative"), "iniciativa", layer.get("initiative_code", ""))
        add(layer.get("legend_label"), "titulo_legenda", layer["id"])
        add(layer.get("collection_title"), "colecao", layer["id"])
    for key, viz in sorted(config.visualizations.items()):
        add(viz.get("label"), "titulo_legenda", key)
        add(viz.get("name"), "nome_visualizacao", key)
        for label in viz.get("discrete_labels") or []:
            add(label, "classe", key)
        for item in viz.get("classes") or []:
            if isinstance(item, dict):
                add(item.get("label"), "classe", key)
    from .territories import sources_for
    for base in ("brasil", "peru", "paraguay", "drc"):
        for source in sources_for(base):
            add(source.get("label"), "tipo_territorio", base)
    for theme, labels in THEME_LABELS.items():
        add(labels.get("pt"), "tema", theme)
    for text in PIPELINE_TEXTS:
        add(text, "texto_fixo", "pipeline")
    return [(text, kind, example) for text, (kind, example) in found.items()]


def _write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(interface_path=INTERFACE_CSV, maps_path=MAPS_CSV):
    """Regrava os CSVs com os textos atuais, mantendo as traduções que já existem."""
    old = _read(Path(interface_path), "pt")
    rows = []
    for text, where in interface_strings():
        row = {"pt": text, "onde": where}
        row.update({code: (old.get(text) or {}).get(code, "") for code in CODES if code != "pt"})
        rows.append(row)
    _write(Path(interface_path), INTERFACE_COLUMNS, rows)
    old_maps = _read(Path(maps_path), "original")
    map_rows = []
    for text, kind, example in map_strings():
        row = {"original": text, "tipo": kind, "exemplo": example}
        row.update({code: (old_maps.get(text) or {}).get(code, "") for code in CODES})
        map_rows.append(row)
    _write(Path(maps_path), MAPS_COLUMNS, map_rows)
    reload()
    return len(rows), len(map_rows)


if __name__ == "__main__":
    counts = export()
    print(f"interface.csv: {counts[0]} textos · mapas.csv: {counts[1]} textos", file=sys.stdout)
