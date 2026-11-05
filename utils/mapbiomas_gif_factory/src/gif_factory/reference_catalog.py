"""Catálogo unificado de layers do inventário com paletas/legendas sugeridas.

Delega o mapeamento produto→legenda a `legends.py`. Uma paleta presente não
comprova acesso ao asset, bandas, máscara ou interpretação: todas as layers
sugeridas permanecem `status: "experimental"` até validação visual no GEE.
"""
import hashlib
import math
import re

from .legends import (INITIATIVES, LANGUAGES, canonical_theme, lang_for,
                      style_for, theme_label)

__all__ = ["INITIATIVES", "LANGUAGES", "canonical_theme", "theme_label",
           "build_reference_layers", "layer_for_record", "experimental_layer"]


def layer_for_record(record):
    """Constrói uma layer unificada a partir de um registro do inventário."""
    key = "|".join(str(record.get(k)) for k in ("initiative", "theme", "collection", "product", "asset"))
    base = {"id": "ref_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
            "name": record["product"],
            "initiative": INITIATIVES.get(record["initiative"], record["initiative"]),
            "theme": canonical_theme(record.get("theme")),
            "collection": record["collection"],
            "asset": record.get("asset"),
            "status": "experimental"}
    if not record.get("asset"):
        base.update(enabled=False, reason="Sem asset no inventário.")
        return base
    lang = lang_for(record.get("initiative"))
    style = style_for(record, lang)
    if style is None:
        base.update(enabled=False, reason="Paleta/legenda pendente de fonte.")
        return base
    base.update(style)
    return base


def build_reference_layers(records):
    """Achata o inventário em layers sugeridas para o catálogo."""
    return [layer_for_record(record) for record in records]


def experimental_layer(record, palette_text, labels_text, minimum, maximum):
    """Cria uma layer selecionável com visualização e rótulos explícitos.

    Mantido por compatibilidade; a UI usa `layer_for_record`.
    """
    if not record.get("asset"):
        raise ValueError("Asset não disponível no inventário / Asset not available / Asset no disponible.")
    colors = [s.strip().lstrip("#") for s in palette_text.split(",") if s.strip()]
    labels = [s.strip() for s in labels_text.split(",")]
    if len(colors) < 2 or any(not re.fullmatch(r"(?:[0-9a-fA-F]{6})", c) for c in colors):
        raise ValueError("Informe pelo menos duas cores hexadecimais de 6 dígitos / Enter at least two hex colors.")
    if len(labels) != len(colors) or any(not label for label in labels):
        raise ValueError("Informe um rótulo por cor / Enter one label per color.")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError("Mínimo deve ser menor que máximo / Minimum must be less than maximum.")
    key = "|".join(str(record[k]) for k in ("initiative", "theme", "collection", "product", "asset"))
    identifier = "ref_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return {"id": identifier, "name": record["product"],
            "initiative": INITIATIVES.get(record["initiative"], record["initiative"]),
            "theme": canonical_theme(record.get("theme")),
            "collection": record["collection"], "asset": record["asset"], "status": "experimental",
            "visualization": {"min": minimum, "max": maximum, "palette": colors},
            "legend": [{"label": label, "color": color} for label, color in zip(labels, colors)]}
