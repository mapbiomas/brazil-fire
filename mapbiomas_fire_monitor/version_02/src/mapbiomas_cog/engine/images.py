"""Resolucao de imagens/unidades no EE (import lazy de ``ee``)."""

from __future__ import annotations

from ..domain.models import unit_key_from_time_start


def _resolve_ic_image(assetid: str, unit_key: str, kind: str):
    """Resolve a imagem de uma ImageCollection cuja unidade == unit_key."""
    import ee

    col = ee.ImageCollection(assetid)
    try:
        idx = col.aggregate_array("system:index").getInfo() or []
        times = col.aggregate_array("system:time_start").getInfo() or []
    except Exception:
        return None
    for index, t in zip(idx, times):
        if t and unit_key_from_time_start(kind, t) == unit_key:
            return col.filter(ee.Filter.eq("system:index", index)).first()
    return None


def resolve_image(assetid: str, unit_key: str, kind: str):
    """Imagem da unidade: banda (IMAGE multibanda) ou imagem (IC)."""
    import ee

    try:
        atype = ee.data.getAsset(assetid).get("type")
    except Exception:
        atype = None
    if atype == "IMAGE":
        return ee.Image(assetid).select(unit_key).rename(unit_key)
    return _resolve_ic_image(assetid, unit_key, kind)


def list_units(assetid: str, kind: str) -> list:
    """Unidades de um produto: bandas (IMAGE) ou chaves YYYY[_MM] (IC)."""
    import ee

    try:
        atype = ee.data.getAsset(assetid).get("type")
    except Exception:
        atype = None
    if atype == "IMAGE":
        return list(ee.Image(assetid).bandNames().getInfo() or [])
    col = ee.ImageCollection(assetid)
    times = col.aggregate_array("system:time_start").getInfo() or []
    keys = {unit_key_from_time_start(kind, t) for t in times if t}
    return sorted(keys, reverse=True)
