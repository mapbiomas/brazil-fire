"""Colhe paletas/legendas da Fábrica anterior para o catálogo unificado.

Fonte: config/visualization.yaml da Fábrica de GIFs anterior (raiz do repo).
Resolve o token `auto_bg` para `ffffff` (fundo claro), pois este pacote não
implementa tema escuro automático.

Gera `src/gif_factory/legends.json`, que acompanha o pacote e alimenta as
sugestões de paleta/legenda do catálogo.

Uso (na raiz do repositório que contém config/visualization.yaml):
    python mapbiomas_gif_factory/validation/build_legends.py
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent.parent / "src" / "gif_factory" / "legends.json"

AUTO_BG = {"auto_bg", "auto", "theme_bg", "background"}


def resolve_color(value):
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in AUTO_BG:
        return "ffffff"
    return text.lstrip("#")


def main():
    path = ROOT / "config" / "visualization.yaml"
    if not path.exists():
        raise SystemExit(f"Não encontrado: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))["visualizations"]
    registry = {}
    for key, viz in data.items():
        if not isinstance(viz, dict):
            continue
        palette = viz.get("palette")
        if not isinstance(palette, list) or not palette:
            continue
        entry = {
            "min": viz.get("min", 0),
            "max": viz.get("max", 1),
            "cmap_type": viz.get("cmap_type", "sequential"),
            "palette": [resolve_color(c) for c in palette],
        }
        for field in ("discrete_labels", "legend_order", "nodata_value", "prefix_labels", "label", "unit"):
            if viz.get(field) is not None:
                entry[field] = viz[field]
        if viz.get("nodata_color") is not None:
            entry["nodata_color"] = resolve_color(viz["nodata_color"])
        registry[key] = entry
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(registry)} legendas -> {OUT}")


if __name__ == "__main__":
    main()
