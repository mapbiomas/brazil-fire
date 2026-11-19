"""Catálogo oficial de mapas, lido das configurações em `gif_factory/config`.

Fonte única de produto -> processador -> visualização, trazida do projeto
local (mapbiomas-ia/FABRICA DE GIFS). Substitui a montagem de camadas a partir
do inventário bruto de assets, que não tinha paletas.

Cada item é uma "camada de catálogo": um dicionário com `source: "catalog"`,
`dataset` e `product`. O motor (`engine.py`) usa esses dois campos para rodar o
produto pelo pipeline exatamente como no projeto local.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from .legends import INITIATIVES, canonical_theme, normalize_collection

CONFIG_DIR = Path(__file__).with_name("config")
STATUSES = ("validado", "beta", "pendente")


@lru_cache(maxsize=1)
def config_loader():
    from mapbiomas_data.config import ConfigLoader
    return ConfigLoader(str(CONFIG_DIR))


@lru_cache(maxsize=1)
def _status_rules():
    path = CONFIG_DIR / "catalog_status.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return data or {}


@lru_cache(maxsize=1)
def _generator_rules():
    """Status padrão dos produtos gerados (sync/generator.yaml, quando presente no código-fonte)."""
    path = CONFIG_DIR.parents[2] / "sync" / "generator.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {"status": (data or {}).get("status", "beta"),
            "reason": (data or {}).get("reason", "Gerado da legenda oficial; aguarda revisão.")}


def status_for(dataset_id, product_id):
    """Status e motivo de um produto, com exceções por dataset e por produto."""
    rules = _status_rules()
    status, reason = rules.get("default", "validado"), ""
    if dataset_id.startswith("gen_"):
        # Gerado das fontes oficiais (sync/catalog_build.py): beta até a revisão da equipe.
        generated = _generator_rules()
        status, reason = generated.get("status", "beta"), generated.get("reason", "")
    dataset_rule = (rules.get("datasets") or {}).get(dataset_id) or {}
    if dataset_rule.get("status"):
        status, reason = dataset_rule["status"], dataset_rule.get("reason", "")
    product_rule = (dataset_rule.get("products") or {}).get(product_id) or {}
    if product_rule.get("status"):
        status, reason = product_rule["status"], product_rule.get("reason", reason)
    if status not in STATUSES:
        raise ValueError(f"Status inválido para {dataset_id}/{product_id}: {status}")
    return status, reason


def legend_from_visualization(viz):
    """Legenda (rótulo, cor) para mostrar na interface antes de gerar."""
    palette = [str(c) for c in viz.get("palette") or []]
    labels = viz.get("discrete_labels") or []
    classes = viz.get("classes") or []
    if classes:
        return [{"label": str(c.get("label", "")), "color": str(c.get("color", ""))} for c in classes]
    if labels and len(labels) == len(palette):
        # Estilos indexados pelo código do pixel deixam posições vazias (sem rótulo, cor
        # de fundo): não são classes, ficam fora da legenda.
        return [{"label": str(label), "color": color} for label, color in zip(labels, palette)
                if str(label).strip() and color.lower() not in ("auto_bg", "auto", "theme_bg", "background")]
    if palette:
        # Escala contínua: mostra só os extremos.
        return [{"label": str(viz.get("min", "")), "color": palette[0]},
                {"label": str(viz.get("max", "")), "color": palette[-1]}]
    return []


def catalog_id(dataset_id, product_id):
    return f"{dataset_id}__{product_id}"


@lru_cache(maxsize=1)
def _catalog():
    config = config_loader()
    initiative_of = {}
    for init_id, info in config.initiatives.items():
        for collection in info.get("collections", []):
            initiative_of[collection["dataset_id"]] = init_id
    visualizations = config.visualizations
    layers = []
    for dataset_id, dataset in config.datasets.items():
        init_id = initiative_of.get(dataset_id, "brasil")
        for product_id, product in (dataset.get("products") or {}).items():
            status, reason = status_for(dataset_id, product_id)
            viz_key = product.get("visualization") or "fire"
            viz = visualizations.get(viz_key) or {}
            layers.append({
                "id": catalog_id(dataset_id, product_id),
                "source": "catalog",
                "dataset": dataset_id,
                "product": product_id,
                "name": product.get("name", product_id),
                "initiative": INITIATIVES.get(init_id, config.initiatives[init_id].get("name", init_id)),
                "initiative_code": init_id,
                "theme": canonical_theme(dataset.get("category")),
                "collection": normalize_collection(dataset.get("collection") or "Sem coleção"),
                "collection_title": dataset.get("title_collection", ""),
                "asset": product.get("asset") or "",
                "processor": product.get("processor"),
                "visualization_key": viz_key,
                "legend": legend_from_visualization(viz),
                "legend_label": viz.get("label", ""),
                "status": status,
                "reason": reason,
                "rgb": bool(product.get("rgb")),
            })
    return layers


def catalog_layers(statuses=("validado", "beta")):
    """Camadas do catálogo com os status pedidos (cópias, seguras para editar)."""
    return [dict(layer) for layer in _catalog() if layer["status"] in statuses]


def find(layer_id):
    for layer in _catalog():
        if layer["id"] == layer_id:
            return dict(layer)
    raise KeyError(f"Produto não encontrado no catálogo: {layer_id}")
