"""Reference records and suggested styling for the unified product catalog.

Suggested styles follow well-known MapBiomas Fire conventions. They are
EXPERIMENTAL: the inventory alone does not verify asset access, actual bands,
masking, pixel values or palette correctness. Every suggested layer remains
`status: "experimental"` and must be visually confirmed before sharing.
"""
import hashlib
import math
import re

INITIATIVES = {
    "brasil": "Brasil", "peru": "Peru", "paraguay": "Paraguai",
    "argentina": "Argentina", "bolivia": "Bolívia", "chile": "Chile",
    "colombia": "Colômbia", "ecuador": "Equador", "indonesia": "Indonésia",
    "venezuela": "Venezuela",
}

_MONTHS = ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
           "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
_MONTH_COLORS = ["cc00ff", "6600ff", "0000ff", "00ccff", "00ffcc", "ffff00",
                 "ff9900", "ff3300", "cc0000", "00cc00", "009900", "66ff66"]

_SEVERITY_LABELS = ["Muito baixa", "Baixa", "Moderada", "Alta", "Muito alta", "Extrema"]
_SEVERITY_COLORS = ["003f1b", "8ee637", "ffd426", "ed7513", "db2024", "99151d"]

_FREQ_COLORS = ["fdfdfd", "faf3cd", "fdf1b0", "fbe981", "fae258", "f9de44",
                "f8d71f", "f1ca1e", "e9bc1c", "e2af1a", "daa118", "d29316",
                "cc8715", "c57b14", "bd6c12", "b76011", "b0540f", "a8450d",
                "9f360b", "992a0a", "921e08", "8c1307", "810004", "770105",
                "6a0306", "600407", "5a0508", "4b0709", "450709", "3f0708",
                "390608", "330607", "2d0506", "270506", "220505", "1c0404",
                "170404", "120303", "0d0203", "080202", "040101"]


def _binary_fire(label0="Não observado", label1="Fogo"):
    return {"visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
            "legend": [{"label": label0, "color": "fdfdfd"}, {"label": label1, "color": "800000"}]}


def _monthly():
    return {"visualization": {"min": 1, "max": 12, "palette": _MONTH_COLORS,
                              "cmap_type": "categorical", "prefix_labels": False,
                              "nodata_value": 0, "nodata_color": "ffffff"},
            "legend": [{"label": label, "color": color} for label, color in zip(_MONTHS, _MONTH_COLORS)]}


def _severity():
    return {"visualization": {"min": 1, "max": 6, "palette": _SEVERITY_COLORS,
                              "cmap_type": "categorical", "prefix_labels": False,
                              "nodata_value": 0, "nodata_color": "ffffff"},
            "legend": [{"label": label, "color": color} for label, color in zip(_SEVERITY_LABELS, _SEVERITY_COLORS)]}


def _frequency(max_value=40):
    return {"visualization": {"min": 0, "max": max_value, "palette": _FREQ_COLORS, "cmap_type": "sequential"},
            "legend": [{"label": "0", "color": _FREQ_COLORS[0]}, {"label": str(max_value), "color": _FREQ_COLORS[-1]}]}


def suggest_style(product):
    """Return a suggested {visualization, legend} for a known Fire product name.

    Returns None when no convention is known. Never returns a suggestion for a
    product whose values, coverage classes or collection specifics are unknown.
    """
    name = (product or "").lower().replace("_", " ").strip()
    if name in {"annual burned", "annual burned area", "accumulated burned", "accumulated burned area"}:
        return _binary_fire()
    if name in {"monthly burned", "monthly burned area"}:
        return _monthly()
    if name in {"severity", "severity class", "potential severity"}:
        return _severity()
    if name in {"fire frequency", "frequency burned", "frequency"}:
        return _frequency()
    return None


def layer_for_record(record):
    """Build a unified layer from a record, with a suggested style when known.

    Records without an asset or without a known style are produced disabled
    with a reason, so the unified catalog shows them without enabling a run.
    """
    base = {"id": "ref_" + hashlib.sha256(
                "|".join(str(record.get(k)) for k in ("initiative", "theme", "collection", "product", "asset")).encode("utf-8")
            ).hexdigest()[:16],
            "name": record["product"],
            "initiative": INITIATIVES.get(record["initiative"], record["initiative"]),
            "theme": record["theme"], "collection": record["collection"],
            "asset": record.get("asset"), "status": "experimental"}
    if not record.get("asset"):
        base.update(enabled=False, reason="Sem asset no inventário.")
        return base
    style = suggest_style(record["product"])
    if style is None:
        base.update(enabled=False, reason="Sem paleta/legenda validada para esta layer.")
        return base
    base.update(**style)
    return base


def build_reference_layers(records):
    """Flatten the inventory into unified, suggested layers for the catalog."""
    return [layer_for_record(record) for record in records]


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
