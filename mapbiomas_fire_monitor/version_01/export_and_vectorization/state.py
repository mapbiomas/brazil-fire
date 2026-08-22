import os
import json
import time
import gcsfs
import urllib.request
from . import config
_fs = None


def _get_fs():
    global _fs
    if _fs is None:
        _fs = gcsfs.GCSFileSystem(token='google_default')
    return _fs


def list_months_in_collection():
    import ee
    import datetime
    try:
        col = ee.ImageCollection(image_collection())
        times = col.aggregate_array('system:time_start').getInfo()
        months = set()
        for t in times:
            dt = datetime.datetime.utcfromtimestamp(t / 1000)
            months.add(f"{dt.year}_{dt.month:02d}")
        return sorted(months, reverse=True)
    except Exception:
        return []


def _new_entry():
    return {
        "exported": False,
        "mosaiced": False,
        "vectorized_gcs": False,
        "vectorized_gee": False,
        "published_mosaic": False,
        "published_vector": False,
        "temp_cleaned": False,
    }


def _tile_unit(basename, context=None):
    """Extrai a unidade de um tile: fire_monitor_v1_{product}_{country}_{unit}_{tileid}.tif."""
    context = context or config.processing_context()
    prefix = f"fire_monitor_v1_{context['product']}_{context['storage_country']}_"
    if not basename.startswith(prefix):
        return None
    rest = basename[len(prefix):].removesuffix(".tif")
    # Tile IDs are appended after the canonical unit. Keep YYYY_MM and bands intact.
    parts = rest.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else rest


def _unit_from_name(basename, prefix_strip, suffix):
    name = basename.removeprefix(prefix_strip).removesuffix(suffix)
    return name or None


def _object_exists(fs, bucket, path):
    """Verifica um objeto por listagem e, se necessario, por direct link."""
    try:
        if fs.exists(f"{bucket}/{path}"):
            return True
    except Exception:
        pass
    try:
        with urllib.request.urlopen(config.gcs_object_url(bucket, path), timeout=8) as response:
            return response.status == 200
    except Exception:
        return False


def _merge_unit(state, unit, **flags):
    entry = state.setdefault(unit, _new_entry())
    entry.update(flags)
    return entry


def _fallback_months():
    """Gera meses candidatos sem transformar o fallback em uma varredura enorme."""
    current_year = time.gmtime().tm_year
    return [f"{year}_{month:02d}" for year in range(current_year - 2, current_year + 1)
            for month in range(1, 13)]


def scan_gcs(context=None, logger=None):
    context = context or config.processing_context()
    fs = _get_fs()
    state = {}
    tiles_present = set()
    cog_present = set()

    def _log(msg):
        if logger:
            logger(msg)

    product = context["product"]
    storage_country = context["storage_country"]
    root = context["root"]
    tiles_path = f"{root}/temp"
    mosaic_path = root
    vector_path = f"{root}_vectors"
    tile_prefix = f"fire_monitor_v1_{product}_{storage_country}_"
    art_prefix = f"{product}-{storage_country}_"

    _log(f"[DEBUG] context={context}")

    _log(f"Scanning GCS: gs://{config.BUCKET}/{tiles_path}/{tile_prefix}*.tif ...")
    try:
        tile_files = fs.glob(f"{config.BUCKET}/{tiles_path}/{tile_prefix}*.tif")
        _log(f"[DEBUG] tile matches={len(tile_files)}")
        for f in tile_files:
            unit = _tile_unit(f.split('/')[-1], context)
            if unit:
                _log(f"[FOUND] Export unit={unit} path={f}")
                tiles_present.add(unit)
                state.setdefault(unit, _new_entry())["exported"] = True
    except Exception as e:
        _log(f"Error scanning tiles: {e}")

    # Fallback para exports conhecidos quando a listagem GCS falha ou esta vazia.
    if context["collection"] == "monitor" and "monthly" in product:
        for unit in _fallback_months():
            tile_glob = f"{tiles_path}/{tile_prefix}{unit}_*.tif"
            matches = fs.glob(f"{config.BUCKET}/{tile_glob}")
            if matches:
                _log(f"[FOUND] Export unit={unit} path={matches[0]}")
                tiles_present.add(unit)
                _merge_unit(state, unit, exported=True)

    _log(f"Scanning GCS: gs://{config.BUCKET}/{mosaic_path}/{art_prefix}*.tif ...")
    try:
        mosaic_files = fs.glob(f"{config.BUCKET}/{mosaic_path}/{art_prefix}*.tif")
        _log(f"[DEBUG] mosaic matches={len(mosaic_files)}")
        for f in mosaic_files:
            unit = _unit_from_name(f.split('/')[-1], art_prefix, ".tif")
            if unit:
                _log(f"[FOUND] Mosaic unit={unit} path={f}")
                cog_present.add(unit)
                entry = state.setdefault(unit, _new_entry())
                entry["exported"] = True
                entry["mosaiced"] = True
    except Exception as e:
        _log(f"Error scanning mosaics: {e}")

    if context["collection"] == "monitor" and "monthly" in product:
        for unit in _fallback_months():
            object_path = f"{mosaic_path}/{art_prefix}{unit}.tif"
            if _object_exists(fs, config.BUCKET, object_path):
                _log(f"[FOUND] Mosaic unit={unit} path={config.BUCKET}/{object_path}")
                cog_present.add(unit)
                _merge_unit(state, unit, exported=True, mosaiced=True)

    _log(f"Scanning GCS: gs://{config.BUCKET}/{vector_path}/{art_prefix}*.zip ...")
    try:
        vector_files = fs.glob(f"{config.BUCKET}/{vector_path}/{art_prefix}*.zip")
        for f in vector_files:
            unit = _unit_from_name(f.split('/')[-1], art_prefix, ".zip")
            if unit:
                state.setdefault(unit, _new_entry())["vectorized_gcs"] = True
    except Exception as e:
        _log(f"Error scanning vectors: {e}")

    _log(f"Scanning public GCS: gs://{config.PUBLIC_BUCKET}/{mosaic_path}/{art_prefix}*.tif ...")
    try:
        pub_mosaic_files = fs.glob(f"{config.PUBLIC_BUCKET}/{mosaic_path}/{art_prefix}*.tif")
        for f in pub_mosaic_files:
            unit = _unit_from_name(f.split('/')[-1], art_prefix, ".tif")
            if unit:
                state.setdefault(unit, _new_entry())["published_mosaic"] = True
    except Exception as e:
        _log(f"Error scanning public mosaics: {e}")

    _log(f"Scanning public GCS: gs://{config.PUBLIC_BUCKET}/{vector_path}/{art_prefix}*.zip ...")
    try:
        pub_vector_files = fs.glob(f"{config.PUBLIC_BUCKET}/{vector_path}/{art_prefix}*.zip")
        for f in pub_vector_files:
            unit = _unit_from_name(f.split('/')[-1], art_prefix, ".zip")
            if unit:
                state.setdefault(unit, _new_entry())["published_vector"] = True
    except Exception as e:
        _log(f"Error scanning public vectors: {e}")

    for unit in cog_present:
        entry = state.setdefault(unit, _new_entry())
        entry["temp_cleaned"] = unit not in tiles_present

    _log(f"[DEBUG] state units={sorted(state)}")

    return state


def scan_gee(context=None, logger=None):
    context = context or config.processing_context()
    import ee
    state = {}

    def _log(msg):
        if logger:
            logger(msg)

    gee_coll = context.get('gee_collection', context['collection'])
    prefix = (f"projects/mapbiomas-public/assets/{context['storage_country']}/"
              f"{context['theme']}/{gee_coll}/{context['product']}_vectors_v01")
    art_prefix = f"{context['product']}-{context['storage_country']}_"
    _log(f"Scanning GEE assets: {prefix} ...")
    try:
        assets = ee.data.listAssets({"parent": prefix})
    except Exception as e:
        _log(f"[WARN] GEE assets folder not found or no access: {prefix} ({e})")
        return state

    def _collect(assets_list):
        for a in assets_list.get("assets", []):
            asset_name = a["name"].split("/")[-1]
            unit = _unit_from_name(asset_name, art_prefix, "")
            if unit:
                state.setdefault(unit, _new_entry())["vectorized_gee"] = True

    try:
        _collect(assets)
        page_token = assets.get("nextPageToken")
        while page_token:
            assets = ee.data.listAssets({"parent": prefix, "pageToken": page_token})
            _collect(assets)
            page_token = assets.get("nextPageToken")
    except Exception as e:
        _log(f"Error scanning GEE: {e}")

    return state


def merge_states(gcs_state, gee_state, units_from_collection):
    result = {}
    all_keys = set(list(gcs_state.keys()) + list(gee_state.keys()) + units_from_collection)
    for key in all_keys:
        result[key] = {
            "exported": gcs_state.get(key, {}).get("exported", False),
            "mosaiced": gcs_state.get(key, {}).get("mosaiced", False),
            "vectorized_gcs": gcs_state.get(key, {}).get("vectorized_gcs", False),
            "vectorized_gee": gee_state.get(key, {}).get("vectorized_gee", False),
            "published_mosaic": gcs_state.get(key, {}).get("published_mosaic", False),
            "published_vector": gcs_state.get(key, {}).get("published_vector", False),
            "temp_cleaned": gcs_state.get(key, {}).get("temp_cleaned", False),
        }
    return result


def build_state(country=None, theme=None, collection=None, product=None, logger=None):
    context = config.processing_context(country, theme, collection, product)
    gcs_state = scan_gcs(context=context, logger=logger)
    gee_state = scan_gee(context=context, logger=logger)
    months = list_months_in_collection()
    full = merge_states(gcs_state, gee_state, months)

    sorted_state = {}
    for key in sorted(full.keys(), reverse=True):
        sorted_state[key] = full[key]

    sorted_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(sorted_state, country=country, theme=theme, collection=collection, product=product)
    return sorted_state


def load_state(country=None, theme=None, collection=None, product=None):
    path = config.state_file(country, theme, collection, product)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state, country=None, theme=None, collection=None, product=None):
    path = config.state_file(country, theme, collection, product)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save state: {e}")


def get_state():
    state = load_state()
    if not state or len(state) <= 1:
        return build_state()
    return state
