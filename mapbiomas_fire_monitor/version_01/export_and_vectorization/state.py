import os
import json
import time
import gcsfs
from . import config
from .config import (
    tiles_prefix,
    mosaic_prefix,
    vector_prefix,
    vector_asset_prefix,
    image_collection,
    tile_pattern,
    mosaic_name,
    vector_name,
)

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


def _month_from_basename(basename, prefix_strip, suffix):
    name = basename.replace(prefix_strip, "").replace(suffix, "")
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def scan_gcs(logger=None):
    fs = _get_fs()
    state = {}
    tiles_present = set()
    cog_present = set()

    def _log(msg):
        if logger:
            logger(msg)

    # Tiles (temp)
    _log(f"Scanning GCS: gs://{config.BUCKET}/{tiles_prefix()}/ ...")
    try:
        tile_files = fs.glob(f"{config.BUCKET}/{tiles_prefix()}/fire_monitor_v1_monthly_burned_{config.COUNTRY}_*.tif")
        for f in tile_files:
            key = _month_from_basename(f.split('/')[-1], f'fire_monitor_v1_monthly_burned_{config.COUNTRY}_', '.tif')
            if key:
                tiles_present.add(key)
                state.setdefault(key, _new_entry())["exported"] = True
    except Exception as e:
        _log(f"Error scanning tiles: {e}")

    # COGs (bucket de processamento)
    _log(f"Scanning GCS: gs://{config.BUCKET}/{mosaic_prefix()}/ ...")
    try:
        mosaic_files = fs.glob(f"{config.BUCKET}/{mosaic_prefix()}/monthly_burned-{config.COUNTRY}_*.tif")
        for f in mosaic_files:
            key = _month_from_basename(f.split('/')[-1], f'monthly_burned-{config.COUNTRY}_', '.tif')
            if key:
                cog_present.add(key)
                entry = state.setdefault(key, _new_entry())
                # COG so existe se houve export + mosaic; tiles podem ja ter sido limpos
                entry["exported"] = True
                entry["mosaiced"] = True
    except Exception as e:
        _log(f"Error scanning mosaics: {e}")

    # Vetores ZIP (bucket de processamento)
    _log(f"Scanning GCS: gs://{config.BUCKET}/{vector_prefix()}/ ...")
    try:
        vector_files = fs.glob(f"{config.BUCKET}/{vector_prefix()}/monthly_burned-{config.COUNTRY}_*.zip")
        for f in vector_files:
            key = _month_from_basename(f.split('/')[-1], f'monthly_burned-{config.COUNTRY}_', '.zip')
            if key:
                state.setdefault(key, _new_entry())["vectorized_gcs"] = True
    except Exception as e:
        _log(f"Error scanning vectors: {e}")

    # Publico (espelho no bucket publico) — COG (mosaico)
    _log(f"Scanning public GCS: gs://{config.PUBLIC_BUCKET}/{mosaic_prefix()}/ ...")
    try:
        pub_mosaic_files = fs.glob(f"{config.PUBLIC_BUCKET}/{mosaic_prefix()}/monthly_burned-{config.COUNTRY}_*.tif")
        for f in pub_mosaic_files:
            key = _month_from_basename(f.split('/')[-1], f'monthly_burned-{config.COUNTRY}_', '.tif')
            if key:
                state.setdefault(key, _new_entry())["published_mosaic"] = True
    except Exception as e:
        _log(f"Error scanning public mosaics: {e}")

    # Publico (espelho no bucket publico) — ZIP (vetor)
    _log(f"Scanning public GCS: gs://{config.PUBLIC_BUCKET}/{vector_prefix()}/ ...")
    try:
        pub_vector_files = fs.glob(f"{config.PUBLIC_BUCKET}/{vector_prefix()}/monthly_burned-{config.COUNTRY}_*.zip")
        for f in pub_vector_files:
            key = _month_from_basename(f.split('/')[-1], f'monthly_burned-{config.COUNTRY}_', '.zip')
            if key:
                state.setdefault(key, _new_entry())["published_vector"] = True
    except Exception as e:
        _log(f"Error scanning public vectors: {e}")

    # Temp limpo = COG existe e nao ha tiles de temp para o mes
    for key in cog_present:
        entry = state.setdefault(key, _new_entry())
        entry["temp_cleaned"] = key not in tiles_present

    return state


def scan_gee(logger=None):
    import ee
    state = {}

    def _log(msg):
        if logger:
            logger(msg)

    prefix = vector_asset_prefix()
    _log(f"Scanning GEE assets: {prefix} ...")
    try:
        assets = ee.data.listAssets({"parent": prefix})
    except Exception as e:
        _log(f"[WARN] GEE assets folder not found or no access: {prefix} ({e})")
        return state

    def _collect(assets_list):
        for a in assets_list.get("assets", []):
            asset_name = a["name"].split("/")[-1]
            key = _month_from_basename(asset_name, f"monthly_burned-{config.COUNTRY}_", "")
            if key:
                state.setdefault(key, _new_entry())["vectorized_gee"] = True

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


def merge_states(gcs_state, gee_state, months_from_collection):
    result = {}
    all_keys = set(list(gcs_state.keys()) + list(gee_state.keys()) + months_from_collection)
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


def build_state(logger=None):
    gcs_state = scan_gcs(logger=logger)
    gee_state = scan_gee(logger=logger)
    months = list_months_in_collection()
    full = merge_states(gcs_state, gee_state, months)

    sorted_state = {}
    for key in sorted(full.keys(), reverse=True):
        sorted_state[key] = full[key]

    sorted_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_state(sorted_state)
    return sorted_state


def load_state():
    try:
        with open(config.STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    try:
        with open(config.STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save state: {e}")


def get_state():
    state = load_state()
    if not state or len(state) <= 1:
        return build_state()
    return state
