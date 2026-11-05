"""Reference records and explicit user-supplied experimental styling."""
import hashlib
import math
import re


INITIATIVES = {
    "brasil": "Brasil", "peru": "Peru", "paraguay": "Paraguai",
    "argentina": "Argentina", "bolivia": "Bolívia", "chile": "Chile",
    "colombia": "Colômbia", "ecuador": "Equador", "indonesia": "Indonésia",
    "venezuela": "Venezuela",
}


def experimental_layer(record, palette_text, labels_text, minimum, maximum):
    """Create a selectable layer only with explicit visualization and labels.

    This does not verify GEE access, raster type, temporal bands or semantics.
    """
    if not record.get("asset"):
        raise ValueError("Asset não disponível no inventário / Asset not available / Asset no disponible.")
    colors = [s.strip().lstrip("#") for s in palette_text.split(",") if s.strip()]
    labels = [s.strip() for s in labels_text.split(",")]
    if len(colors) < 2 or any(not re.fullmatch(r"(?:[0-9a-fA-F]{6})", c) for c in colors):
        raise ValueError("Informe pelo menos duas cores hexadecimais de 6 dígitos / Enter at least two hex colors / Indique al menos dos colores hex.")
    if len(labels) != len(colors) or any(not label for label in labels):
        raise ValueError("Informe um rótulo por cor / Enter one label per color / Indique una etiqueta por color.")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
        raise ValueError("Mínimo deve ser menor que máximo / Minimum must be less than maximum / Mínimo debe ser menor que máximo.")
    key = "|".join(str(record[k]) for k in ("initiative", "theme", "collection", "product", "asset"))
    identifier = "ref_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return {"id": identifier, "name": record["product"],
            "initiative": INITIATIVES.get(record["initiative"], record["initiative"]),
            "theme": record["theme"], "collection": record["collection"],
            "asset": record["asset"], "status": "experimental",
            "visualization": {"min": minimum, "max": maximum, "palette": colors},
            "legend": [{"label": label, "color": color} for label, color in zip(labels, colors)]}
