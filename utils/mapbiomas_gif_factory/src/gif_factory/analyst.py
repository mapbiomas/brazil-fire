"""Modo analista: transformar um asset beta em um mapa da Fábrica sem escrever código.

- `inspect_asset`: tipo, bandas e anos detectados no asset;
- `style_choices`: os estilos do catálogo (paleta + legenda) para reaproveitar;
- `build_layer`: monta a camada a partir de um estilo existente ou de cores próprias;
- `catalog_snippet`: o trecho YAML para propor o mapa ao catálogo oficial.
"""
import hashlib
import re

import yaml

from .catalog import config_loader, legend_from_visualization
from .engine import load_image, prepare_layer
from .temporal import temporal_key
from .territories import plain

THEMES = [("Fogo", "fire"), ("Uso e cobertura", "lulc"), ("Degradação", "degradation"), ("Solo", "soil"),
          ("Água", "water"), ("Outros", "outros")]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", plain(text)).strip("_")[:40] or "mapa"


def layer_id(asset, name):
    digest = hashlib.sha256(f"{asset}|{name}".encode("utf-8")).hexdigest()[:8]
    return f"meu_{slug(name)}_{digest}"


def inspect_asset(asset):
    """{'kind', 'bands', 'years', 'axis'} do asset, como o motor vai lê-lo."""
    import ee
    asset = str(asset or "").strip()
    if not asset:
        raise ValueError("Cole o caminho do asset (ex.: projects/meu-projeto/assets/minha_imagem).")
    info = ee.data.getAsset(asset)
    kind = info.get("type", "")
    if kind not in ("IMAGE", "IMAGE_COLLECTION"):
        raise ValueError(f"O asset precisa ser uma imagem ou coleção de imagens (é {kind or 'desconhecido'}).")
    layer = prepare_layer({"asset": asset})
    bands = load_image(layer).bandNames().getInfo()
    keys = [temporal_key(b) for b in bands]
    years = (min(k[0] for k in keys), max(k[0] for k in keys)) if keys and all(keys) else None
    return {"kind": kind, "bands": bands, "years": years, "axis": layer.get("axis", "time")}


def style_choices():
    """[(rótulo, chave)] dos estilos do catálogo que têm legenda (servem para mapas de classes ou escalas)."""
    from mapbiomas_data.core import VisualizationManager
    manager = VisualizationManager(config_loader())
    choices = []
    for key, raw in config_loader().visualizations.items():
        if not isinstance(raw, dict) or raw.get("random_viz") or raw.get("rgb_legend") or not raw.get("palette"):
            continue
        try:
            viz = manager.get_viz_params(key)
        except Exception:
            continue
        if legend_from_visualization(viz):
            choices.append((f"{viz.get('name') or key}  ·  {key}", key))
    return sorted(choices, key=lambda item: plain(item[0]))


def style_params(key):
    from mapbiomas_data.core import VisualizationManager
    viz = VisualizationManager(config_loader()).get_viz_params(key)
    return viz, legend_from_visualization(viz)


def parse_colors(text):
    colors = [c.strip().lstrip("#") for c in re.split(r"[,\s;]+", str(text or "")) if c.strip()]
    bad = [c for c in colors if not re.fullmatch(r"[0-9a-fA-F]{6}", c)]
    if bad:
        raise ValueError(f"Cores inválidas: {', '.join(bad)}. Use hexadecimais de 6 dígitos (ex.: ff0000).")
    if len(colors) < 2:
        raise ValueError("Informe pelo menos duas cores.")
    return colors


def build_layer(asset, name, theme, collection, style_key=None, minimum=None, maximum=None,
                colors="", labels="", legend_title=""):
    """Camada pronta para o motor: com estilo do catálogo (`style_key`) ou cores próprias."""
    asset, name = str(asset or "").strip(), str(name or "").strip()
    if not asset or not name:
        raise ValueError("Informe o asset e um nome para o mapa.")
    layer = {"id": layer_id(asset, name), "name": name, "asset": asset, "theme": theme or "outros",
             "collection": str(collection or "beta").strip() or "beta", "status": "experimental",
             "initiative": "Meus mapas"}
    if style_key:
        viz, legend = style_params(style_key)
        layer.update(visualization={k: viz[k] for k in ("min", "max", "palette", "cmap_type", "discrete_labels",
                                                        "label", "nodata_color") if viz.get(k) is not None},
                     legend=legend, legend_label=legend_title or viz.get("label", ""), style_key=style_key)
    else:
        palette = parse_colors(colors)
        names = [s.strip() for s in str(labels or "").split(",")]
        if names == [""]:
            names = []
        if names and len(names) != len(palette):
            raise ValueError(f"São {len(palette)} cores e {len(names)} rótulos; informe um rótulo por cor ou nenhum.")
        try:
            low, high = float(minimum), float(maximum)
        except (TypeError, ValueError):
            raise ValueError("Informe os valores mínimo e máximo.")
        if low >= high:
            raise ValueError("O mínimo precisa ser menor que o máximo.")
        low, high = (int(low) if low.is_integer() else low), (int(high) if high.is_integer() else high)
        legend = ([{"label": label, "color": color} for label, color in zip(names, palette)] if names else
                  [{"label": str(low), "color": palette[0]}, {"label": str(high), "color": palette[-1]}])
        layer.update(visualization={"min": low, "max": high, "palette": palette,
                                    "cmap_type": "discrete" if names else "sequential",
                                    **({"discrete_labels": names} if names else {})},
                     legend=legend, legend_label=legend_title)
    return layer


def detect_range(asset):
    """(mín, máx) aproximados do último período, para sugerir a escala de cores."""
    import ee
    layer = prepare_layer({"asset": asset})
    image = load_image(layer)
    band = image.bandNames().getInfo()[-1]
    selected = image.select(band)
    stats = selected.reduceRegion(ee.Reducer.minMax(), geometry=selected.geometry().bounds(), scale=5000,
                                  bestEffort=True, maxPixels=1e8).getInfo()
    values = [v for v in stats.values() if v is not None]
    if len(values) != 2:
        raise ValueError("Não foi possível calcular os valores (imagem vazia ou sem área definida).")
    return min(values), max(values)


def catalog_snippet(layer, initiative="brasil"):
    """Trecho YAML para colar em config/initiatives/<iniciativa>/collections/<coleção>.yaml."""
    key = slug(layer["name"])
    product = {"name": layer["name"], "asset": layer["asset"]}
    if layer.get("style_key"):
        product["visualization"] = layer["style_key"]
        extra = ""
    else:
        viz_key = f"{key}_beta"
        product["visualization"] = viz_key
        viz = layer["visualization"]
        style = {viz_key: {"name": layer["name"], "min": viz["min"], "max": viz["max"], "palette": viz["palette"],
                           "label": layer.get("legend_label") or layer["name"], "cmap_type": viz.get("cmap_type", "sequential")}}
        if viz.get("discrete_labels"):
            style[viz_key]["discrete_labels"] = viz["discrete_labels"]
        extra = ("\n# --- e em config/visualization.yaml, dentro de `visualizations:` ---\n"
                 + _indent(yaml.safe_dump(style, allow_unicode=True, sort_keys=False), 2))
    body = _indent(yaml.safe_dump({key: product}, allow_unicode=True, sort_keys=False), 2)
    return (f"# --- em config/initiatives/{initiative}/collections/<coleção>.yaml, dentro de `products:` ---\n"
            f"{body}{extra}"
            "# --- e em config/catalog_status.yaml, enquanto estiver em revisão ---\n"
            f"#   <dataset>: {{products: {{{key}: {{status: beta, reason: \"Proposto por analista\"}}}}}}\n")


def _indent(text, spaces):
    pad = " " * spaces
    return "".join(pad + line if line.strip() else line for line in text.splitlines(True))
