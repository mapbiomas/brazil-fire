"""Descoberta e metadados dos produtos de fogo do MapBiomas (multipais).

Baseado em `config.OBJ` (pais -> tema -> colecao -> produtos). Para cada produto
com `assetid`, inspeciona no GEE: tipo (IMAGE/IMAGE_COLLECTION), bandas, dtype,
max observado, temporal e **units** (bandas p/ imagem multibanda; imagens p/
IMAGE_COLLECTION). O `type` declarado no OBJ e mantido (nao sobrescrito).

Uso (onde houver GEE, ex.: Colab):
    from export_and_vectorization.catalog import build_inventory
    inv = build_inventory(["brasil", "indonesia"], refresh=True)
"""

import datetime
import json
import re

from . import config

CACHE_FILE = "catalog_cache.json"


def short_name(name):
    n = re.sub(r"^mapbiomas_(?:\w+_)?fire_", "", name or "")
    n = re.sub(r"^collection\d+_?", "", n)
    n = re.sub(r"_v\d+$", "", n)
    return n


def detect_kind(product):
    n = (product or "").lower()
    if "monthly" in n:
        return "monthly"
    if "annual" in n:
        return "annual"
    if any(k in n for k in ("frequency", "accumulated", "recurrence", "time_after", "year_last", "interval")):
        return "period"
    return "other"


def is_vectorizable(product):
    return detect_kind(product) == "annual" and short_name(product) == "annual_burned"


def save_for_type(ptype):
    t = (ptype or "byte").lower()
    if t in ("float32", "float64", "float"):
        return {"ot": "Float32", "nodata": 0, "predictor": 3, "compression": "DEFLATE"}
    if t in ("int16", "uint16", "int32"):
        return {"ot": "Int16", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}
    return {"ot": "Byte", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}


def _dtype_from_data_type(dt):
    bits = dt.get("bits", 8)
    prec = dt.get("precision", "")
    if prec == "float":
        return "Float32" if bits <= 32 else "Float64"
    if prec == "unsigned int":
        return f"Uint{bits}"
    return f"Int{bits}"


def _observed_max(image):
    import ee
    try:
        red = image.reduceRegion(reducer=ee.Reducer.max(), scale=1000, bestEffort=True).getInfo()
        vals = [v for v in red.values() if isinstance(v, (int, float))]
        return max(vals) if vals else None
    except Exception:
        return None


def _bands_from_image(image):
    bands = []
    try:
        g = image.getInfo()
        for b in g.get("bands", []):
            dt = b.get("data_type", {})
            bands.append({"name": b.get("id"), "dtype": _dtype_from_data_type(dt)})
    except Exception:
        pass
    return bands


def inspect_asset(asset_id):
    """Retorna tipo, bandas, dtype, max, temporal e units de um produto."""
    import ee
    info = {"assetid": asset_id}
    try:
        ameta = ee.data.getAsset(asset_id)
        atype = ameta.get("type")
    except Exception as e:
        info["error"] = str(e)
        return info
    info["type"] = atype

    if atype == "IMAGE":
        img = ee.Image(asset_id)
        bands = _bands_from_image(img)
        info["bands"] = [b["name"] for b in bands]
        info["dtype"] = bands[0]["dtype"] if bands else None
        info["units"] = [{"key": b["name"], "label": b["name"]} for b in bands]
        info["max"] = _observed_max(img)
        return info

    if atype == "IMAGE_COLLECTION":
        col = ee.ImageCollection(asset_id)
        try:
            info["n_images"] = col.size().getInfo()
        except Exception:
            info["n_images"] = None
        bands = _bands_from_image(col.first())
        info["bands"] = [b["name"] for b in bands]
        info["dtype"] = bands[0]["dtype"] if bands else None
        try:
            info["max"] = _observed_max(col.first())
        except Exception:
            info["max"] = None
        # A unit publica e temporal; system:index fica somente para resolver a imagem.
        try:
            idx = col.aggregate_array("system:index").getInfo() or []
            ts = col.aggregate_array("system:time_start").getInfo() or []
            units = []
            for i, ix in enumerate(idx):
                key = ix
                if i < len(ts) and ts[i]:
                    try:
                        key = datetime.datetime.utcfromtimestamp(ts[i] / 1000).strftime("%Y_%m")
                    except Exception:
                        pass
                units.append({"key": key, "label": key, "image_id": ix})
            info["units"] = units
        except Exception:
            info["units"] = []
        try:
            info["t_start"] = col.aggregate_min("system:time_start").getInfo()
            info["t_end"] = col.aggregate_max("system:time_start").getInfo()
        except Exception:
            pass
        return info

    return info


def load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(inv):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(inv, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not save catalog cache: {e}")


def build_inventory(countries=None, refresh=False):
    """Inventario dos produtos do OBJ, enriquecido pelo discovery GEE (cacheado)."""
    countries = countries or list(config.OBJ)
    cache = {} if refresh else load_cache()
    inv = {}
    for country in countries:
        if country in cache and cache[country] and not refresh:
            inv[country] = cache[country]
            continue
        country_inv = {}
        for theme, collections in config.OBJ.get(country, {}).items():
            for coll, prods in collections.items():
                entries = []
                for p in prods:
                    if not p.get("visible", True):
                        continue
                    meta = ({"assetid": p["assetid"], "kind": "GCS_PREFIX"}
                            if p["assetid"].startswith("gcs://")
                            else inspect_asset(p["assetid"]))
                    rec = {
                        "name": p["product"],
                        "assetid": p["assetid"],
                        "declared_type": p.get("type", "byte"),
                        "kind": detect_kind(p["product"]),
                        "vectorize": p.get("vectorize", is_vectorizable(p["product"])),
                        "unit": p.get("unit", ""),
                    }
                    rec.update(meta)
                    rec["save"] = save_for_type(p.get("type", "byte"))
                    entries.append(rec)
                if entries:
                    country_inv.setdefault(theme, {})[coll] = entries
        inv[country] = country_inv
    save_cache(inv)
    return inv
