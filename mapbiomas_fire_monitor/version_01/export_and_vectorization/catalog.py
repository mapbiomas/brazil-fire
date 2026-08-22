"""Descoberta e metadados dos produtos de fogo do MapBiomas (multipais).

Etapa de metadados: inventaria os produtos sob
projects/mapbiomas-public/assets/{country}/fire/** (todas as colecoes),
inspeciona tipo, bandas, dtype, max observado e temporal, classifica por
`kind` (monthly/annual/period) e sugere como salvar no mosaico
(-ot/nodata/predictor). A flag `vectorize` indica se o produto gera
vetorizacao (apenas annual_burned).

Uso (onde houver GEE/GCS, ex.: Colab):
    from export_and_vectorization.catalog import build_inventory, load_cache
    inv = build_inventory(["brazil", "indonesia"], refresh=True)
"""

import json
import os
import re
import time

from . import config

CACHE_FILE = "catalog_cache.json"

# Seed: kind/unit/vectorize por produto conhecido. dtype/max/bandas/temporal
# sao obtidos do GEE pelo discovery.
PRODUCT_SEED = {
    "brazil": {
        "monitor": {
            "mapbiomas_fire_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
        },
        "collection4": {
            "mapbiomas_fire_collection4_annual_burned_v1": {"kind": "annual", "unit": "burned (0/1)", "vectorize": True},
            "mapbiomas_fire_collection4_annual_burned_coverage_v1": {"kind": "annual", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection4_annual_burned_scar_size_range_v1": {"kind": "annual", "unit": "scar size class", "vectorize": False},
            "mapbiomas_fire_collection4_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection4_accumulated_burned_v1": {"kind": "period", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection4_fire_frequency_v1": {"kind": "period", "unit": "frequency (count)", "vectorize": False},
            "mapbiomas_fire_collection4_time_after_fire_v1": {"kind": "period", "unit": "years since fire", "vectorize": False},
            "mapbiomas_fire_collection4_year_last_fire_v1": {"kind": "period", "unit": "last fire year", "vectorize": False},
        },
        "collection4_1": {
            "mapbiomas_fire_collection41_annual_burned_v1": {"kind": "annual", "unit": "burned (0/1)", "vectorize": True},
            "mapbiomas_fire_collection41_annual_burned_coverage_v1": {"kind": "annual", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection41_annual_burned_scar_size_range_v1": {"kind": "annual", "unit": "scar size class", "vectorize": False},
            "mapbiomas_fire_collection41_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection41_accumulated_burned_v1": {"kind": "period", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection41_fire_frequency_v1": {"kind": "period", "unit": "frequency (count)", "vectorize": False},
            "mapbiomas_fire_collection41_time_after_fire_v1": {"kind": "period", "unit": "years since fire", "vectorize": False},
            "mapbiomas_fire_collection41_year_last_fire_v1": {"kind": "period", "unit": "last fire year", "vectorize": False},
        },
        "collection5": {
            "mapbiomas_fire_collection5_annual_burned_v1": {"kind": "annual", "unit": "burned (0/1)", "vectorize": True},
            "mapbiomas_fire_collection5_annual_burned_coverage_v1": {"kind": "annual", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection5_annual_burned_scar_size_range_v1": {"kind": "annual", "unit": "scar size class", "vectorize": False},
            "mapbiomas_fire_collection5_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection5_accumulated_burned_v1": {"kind": "period", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection5_accumulated_burned_coverage_v1": {"kind": "period", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection5_fire_frequency_v1": {"kind": "period", "unit": "frequency (count)", "vectorize": False},
        },
    },
    "indonesia": {
        "monitor": {
            "mapbiomas_fire_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
        },
        "collection1": {
            "mapbiomas_fire_collection1_annual_burned_v1": {"kind": "annual", "unit": "burned (0/1)", "vectorize": True},
            "mapbiomas_fire_collection1_annual_burned_coverage_v1": {"kind": "annual", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection1_monthly_burned_v1": {"kind": "monthly", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection1_accumulated_burned_v1": {"kind": "period", "unit": "burned (0/1)", "vectorize": False},
            "mapbiomas_fire_collection1_accumulated_burned_coverage_v1": {"kind": "period", "unit": "coverage (%)", "vectorize": False},
            "mapbiomas_fire_collection1_fire_frequency_v1": {"kind": "period", "unit": "frequency (count)", "vectorize": False},
        },
    },
}


def short_name(name):
    n = re.sub(r"^mapbiomas_fire_", "", name or "")
    n = re.sub(r"^collection\d+_?", "", n)
    n = re.sub(r"_v\d+$", "", n)
    return n


def detect_kind(name):
    n = (name or "").lower()
    if "monthly" in n:
        return "monthly"
    if "annual" in n:
        return "annual"
    if any(k in n for k in ("frequency", "accumulated", "recurrence", "time_after", "year_last")):
        return "period"
    return "other"


def is_vectorizable(name):
    """Vetorizacao apenas para o produto anual de cicatriz 0/1 (annual_burned)."""
    return detect_kind(name) == "annual" and short_name(name) == "annual_burned"


def suggest_save(dtype, max_value):
    """Sugere como salvar o mosaico a partir do dtype e do max observado."""
    dtype = (dtype or "").lower()
    if "float" in dtype or "double" in dtype:
        return {"ot": "Float32", "nodata": 0, "predictor": 3, "compression": "DEFLATE"}
    if max_value is None or max_value <= 255:
        return {"ot": "Byte", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}
    return {"ot": "Int16", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}


def _dtype_from_data_type(dt):
    bits = dt.get("bits", 8)
    prec = dt.get("precision", "")
    if prec == "float":
        return "Float32" if bits <= 32 else "Float64"
    if prec == "unsigned int":
        return f"Uint{bits}"
    return f"Int{bits}"


def _list_all(parent):
    import ee
    items = []
    res = ee.data.listAssets({"parent": parent})
    items.extend(res.get("assets", []))
    token = res.get("nextPageToken")
    while token:
        res = ee.data.listAssets({"parent": parent, "pageToken": token})
        items.extend(res.get("assets", []))
        token = res.get("nextPageToken")
    return items


def list_products(country, theme="fire"):
    """Retorna {colecao: [{'name','type'}, ...]} sob .../{country}/{theme}."""
    import ee
    root = f"projects/mapbiomas-public/assets/{country}/{theme}"
    out = {}
    try:
        top = _list_all(root)
    except Exception:
        return {}
    for a in top:
        if a.get("type") == "FOLDER":
            coll = a["name"].split("/")[-1]
            try:
                children = _list_all(a["name"])
            except Exception:
                continue
            prods = [{"name": c["name"].split("/")[-1], "type": c["type"]}
                     for c in children if c.get("type") in ("IMAGE", "IMAGE_COLLECTION")]
            if prods:
                out[coll] = prods
    return out


def _observed_max(image):
    import ee
    try:
        red = image.reduceRegion(reducer=ee.Reducer.max(), scale=1000, bestEffort=True).getInfo()
        vals = [v for v in red.values() if isinstance(v, (int, float))]
        return max(vals) if vals else None
    except Exception:
        return None


def inspect_asset(asset_id, asset_type="IMAGE_COLLECTION"):
    """Retorna bandas, dtype, max observado e temporal de um produto."""
    import ee
    info = {}
    if asset_type == "IMAGE":
        img = ee.Image(asset_id)
        try:
            meta = img.getInfo()
            bands = []
            for b in meta.get("bands", []):
                dt = b.get("data_type", {})
                bands.append({"name": b.get("id"), "dtype": _dtype_from_data_type(dt)})
            info["bands"] = [b["name"] for b in bands]
            info["dtype"] = bands[0]["dtype"] if bands else None
        except Exception as e:
            info["error"] = str(e)
        info["max"] = _observed_max(img)
        return info

    col = ee.ImageCollection(asset_id)
    try:
        info["n_images"] = col.size().getInfo()
    except Exception:
        info["n_images"] = None
    try:
        first = col.first()
        meta = first.getInfo()
        bands = []
        for b in meta.get("bands", []):
            dt = b.get("data_type", {})
            bands.append({"name": b.get("id"), "dtype": _dtype_from_data_type(dt)})
        info["bands"] = [b["name"] for b in bands]
        info["dtype"] = bands[0]["dtype"] if bands else None
        info["max"] = _observed_max(first)
    except Exception as e:
        info["error"] = str(e)
    try:
        info["t_start"] = col.aggregate_min("system:time_start").getInfo()
        info["t_end"] = col.aggregate_max("system:time_start").getInfo()
    except Exception:
        pass
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
    """Descobre os produtos de cada pais e monta o inventario (cacheado)."""
    countries = countries or list(config.COUNTRIES)
    cache = {} if refresh else load_cache()
    inv = {}
    for country in countries:
        if country in cache and cache[country] and not refresh:
            inv[country] = cache[country]
            continue
        products = list_products(country)
        country_inv = {}
        for coll, prods in products.items():
            country_inv[coll] = []
            for p in prods:
                asset_id = f"projects/mapbiomas-public/assets/{country}/fire/{coll}/{p['name']}"
                meta = inspect_asset(asset_id, p["type"])
                seed = PRODUCT_SEED.get(country, {}).get(coll, {}).get(p["name"], {})
                kind = seed.get("kind") or detect_kind(p["name"])
                rec = {
                    "name": p["name"],
                    "type": p["type"],
                    "kind": kind,
                    "unit": seed.get("unit", ""),
                    "vectorize": seed.get("vectorize", is_vectorizable(p["name"])),
                }
                rec.update(meta)
                rec["save"] = suggest_save(rec.get("dtype"), rec.get("max"))
                rec["short"] = short_name(p["name"])
                country_inv[coll].append(rec)
        inv[country] = country_inv
    save_cache(inv)
    return inv
