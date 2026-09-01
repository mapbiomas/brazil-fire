"""Descoberta e metadados dos produtos de fogo do MapBiomas (multipais).

Baseado em `config.OBJ` (pais -> tema -> colecao -> produtos). Para cada produto
com `assetid`, inspeciona no GEE: tipo (IMAGE/IMAGE_COLLECTION), bandas, dtype,
max observado, temporal e **units** (bandas p/ imagem multibanda; imagens p/
IMAGE_COLLECTION). O `type` declarado no OBJ e mantido (nao sobrescrito).

Uso (onde houver GEE, ex.: Colab):
    from export_and_vectorization.catalog import build_inventory
    inv = build_inventory(["brasil", "indonesia"], refresh=True)
"""

import argparse
import datetime
import json
import os
import re

from . import config

# Arquivo versionado (memoria persistente entre sessoes). Resolve relativo ao
# pacote (e nao ao CWD) para que, no Colab, o cache commitado no repo seja
# encontrado mesmo com CWD=/content.
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog_cache.json")

# Memoria em sessao do inventario por pais (evita re-parsing do JSON a cada
# ativacao de produto). Invalidada por build_inventory(..., refresh=True).
_MEMO = {}


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


def inspect_asset(asset_id, kind=None):
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
        # Formato da chave acompanha o kind (annual -> %Y; demais -> %Y_%m),
        # em linha com config.unit_key_for_image usado no export.
        fmt = "%Y" if kind == "annual" else "%Y_%m"
        try:
            idx = col.aggregate_array("system:index").getInfo() or []
            ts = col.aggregate_array("system:time_start").getInfo() or []
            units = []
            for i, ix in enumerate(idx):
                key = ix
                if i < len(ts) and ts[i]:
                    try:
                        key = datetime.datetime.utcfromtimestamp(ts[i] / 1000).strftime(fmt)
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


def _load_inventory(country):
    """Inventario de um pais: memo -> cache em disco -> None."""
    if country in _MEMO and _MEMO[country]:
        return _MEMO[country]
    cached = load_cache()
    if cached.get(country):
        _MEMO[country] = cached[country]
        return _MEMO[country]
    return None


def _build_country_inventory(country):
    """Descoberta completa (GEE) de TODOS os produtos visiveis de um pais."""
    country_inv = {}
    for theme, collections in config.OBJ.get(country, {}).items():
        for coll, prods in collections.items():
            entries = []
            for p in prods:
                if not p.get("visible", True):
                    continue
                meta = ({"assetid": p["assetid"], "kind": "GCS_PREFIX"}
                        if p["assetid"].startswith("gcs://")
                        else inspect_asset(p["assetid"], kind=detect_kind(p["product"])))
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
    return country_inv


def build_inventory(countries=None, refresh=False):
    """Inventario dos produtos do OBJ, enriquecido pelo discovery GEE (cacheado).

    Usa a memo em sessao e so grava em disco quando algo foi realmente
    descoberto/atualizado (evita reescrever o JSON a cada chamada).
    """
    countries = countries or list(config.OBJ)
    inv = {}
    changed = False
    for country in countries:
        if not refresh:
            existing = _load_inventory(country)
            if existing:
                inv[country] = existing
                continue
        country_inv = _build_country_inventory(country)
        _MEMO[country] = country_inv
        inv[country] = country_inv
        changed = True
    if changed:
        merged = dict(load_cache())
        merged.update(inv)
        save_cache(merged)
    return inv


def inventory_units(country, theme, collection, product, logger=None):
    """Units (bandas/imagens) de UM produto, descobrindo so quando necessario.

    Chamado no Load Data. Se o produto ja esta no catalogo (memo/disco) COM
    units, devolve direto; caso contrario, inspeciona APENAS esse produto no
    GEE e atualiza/substitui a entrada no cache — evita encher o cache com
    produtos nao carregados.
    """
    current = _load_inventory(country)
    if current is None:
        current = {}
        _MEMO[country] = current
    prods = current.get(theme, {}).get(collection, [])
    existing_idx = None
    for i, p in enumerate(prods):
        if p.get("name") == product:
            existing_idx = i
            units = p.get("units") or []
            if units:
                return [u.get("key") for u in units]
            break  # entrada existe mas sem units -> redescobre

    meta = config.find_product(country, theme, collection, product)
    if not meta:
        return []
    info = inspect_asset(meta["assetid"], kind=detect_kind(product))
    rec = {
        "name": product,
        "assetid": meta["assetid"],
        "declared_type": meta.get("type", "byte"),
        "kind": detect_kind(product),
        "vectorize": meta.get("vectorize", is_vectorizable(product)),
        "unit": meta.get("unit", ""),
    }
    rec.update(info)
    rec["save"] = save_for_type(meta.get("type", "byte"))
    if existing_idx is not None:
        prods[existing_idx] = rec
    else:
        prods.append(rec)

    # Espelha a atualizacao no arquivo em disco (replace na mesma posicao).
    disk = dict(load_cache())
    disk_prods = disk.setdefault(country, {}).setdefault(theme, {}).setdefault(collection, [])
    disk_idx = next((i for i, p in enumerate(disk_prods) if p.get("name") == product), None)
    if disk_idx is not None:
        disk_prods[disk_idx] = rec
    else:
        disk_prods.append(rec)
    save_cache(disk)

    n_units = len(info.get("units") or [])
    if logger:
        logger(f"[CATALOG] Descoberto: {country}/{theme}/{collection}/{product} "
               f"({n_units} unidade(s))")
    return [u.get("key") for u in (info.get("units") or [])]


def download_cache(countries=None, logger=None):
    """Salva o cache com tudo que foi carregado na sessao e o exporta/baixa.

    Colab: dispara files.download('catalog_cache.json'). Fora do Colab:
    grava o arquivo localmente e imprime o caminho. Serve para versionar a
    "memoria" no GitHub quando novos dados forem adicionados ao config.py.
    """
    countries = countries or list(config.OBJ)
    merged = dict(load_cache())
    for c in countries:
        if _MEMO.get(c):
            merged[c] = _MEMO[c]
    save_cache(merged)
    path = os.path.abspath(CACHE_FILE)
    try:
        from google.colab import files as colab_files  # noqa: F401
        colab_files.download(CACHE_FILE)
        if logger:
            logger("[CATALOG] Download do catalog_cache.json iniciado.")
    except Exception:
        if logger:
            logger(f"[CATALOG] Catalog cache salvo em: {path}")
        else:
            print(f"Catalog cache saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", default=None,
                        help="codigo do pais (pode repetir). Default: todos")
    parser.add_argument("--all", action="store_true", help="todos os paises configurados")
    parser.add_argument("--refresh", action="store_true",
                        help="re-escaneia o GEE (ignora memo/cache)")
    args = parser.parse_args()
    if args.all or not args.country:
        countries = list(config.OBJ)
    else:
        countries = [c for c in args.country if c in config.OBJ]
    inv = build_inventory(countries, refresh=args.refresh)
    for country in countries:
        colls = inv.get(country, {})
        n_prod = sum(len(prods) for prods in colls.values())
        n_units = sum(len(p.get("units") or [])
                      for prods in colls.values() for p in prods)
        print(f"{country}: {n_prod} produto(s), {n_units} unidade(s)")
    print("Cache:", os.path.abspath(CACHE_FILE))


if __name__ == "__main__":
    main()
