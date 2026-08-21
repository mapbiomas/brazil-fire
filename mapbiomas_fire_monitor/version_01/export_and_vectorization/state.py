import os
import json
import time
import gcsfs
from . import config
from .config import (
    BUCKET,
    PUBLIC_BUCKET,
    STATE_FILE,
    COUNTRY,
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
    return {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}


def scan_gcs(logger=None):
    fs = _get_fs()
    state = {}

    def _log(msg):
        if logger:
            logger(msg)

    _log(f"Scanning GCS: gs://{BUCKET}/{tiles_prefix()}/ ...")
    try:
        tile_files = fs.glob(f"{BUCKET}/{tiles_prefix()}/fire_monitor_v1_monthly_burned_{COUNTRY}_*.tif")
        for f in tile_files:
            basename = f.split('/')[-1]
            parts = basename.replace(f'fire_monitor_v1_monthly_burned_{COUNTRY}_', '').split('_')
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1][:2]}"
                state.setdefault(key, _new_entry())["exported"] = True
    except Exception as e:
        _log(f"Error scanning tiles: {e}")

    _log(f"Scanning GCS: gs://{BUCKET}/{mosaic_prefix()}/ ...")
    try:
        mosaic_files = fs.glob(f"{BUCKET}/{mosaic_prefix()}/monthly_burned-{COUNTRY}_*.tif")
        for f in mosaic_files:
            basename = f.split('/')[-1]
            name = basename.replace(f'monthly_burned-{COUNTRY}_', '').replace('.tif', '')
            parts = name.split('_')
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                entry = state.setdefault(key, _new_entry())
                # COG so existe se houve export + mosaic; tiles podem ja ter sido limpos
                entry["exported"] = True
                entry["mosaiced"] = True
    except Exception as e:
        _log(f"Error scanning mosaics: {e}")

    _log(f"Scanning GCS: gs://{BUCKET}/{vector_prefix()}/ ...")
    try:
        vector_files = fs.glob(f"{BUCKET}/{vector_prefix()}/monthly_burned-{COUNTRY}_*.zip")
        for f in vector_files:
            basename = f.split('/')[-1]
            name = basename.replace(f'monthly_burned-{COUNTRY}_', '').replace('.zip', '')
            parts = name.split('_')
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                state.setdefault(key, _new_entry())["vectorized_gcs"] = True
    except Exception as e:
        _log(f"Error scanning vectors: {e}")

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
        _log(f"[WARN] Pasta de assets GEE nao encontrada ou sem acesso: {prefix} ({e})")
        return state

    def _collect(assets_list):
        for a in assets_list.get("assets", []):
            asset_name = a["name"].split("/")[-1]
            parts = asset_name.replace(f"monthly_burned-{COUNTRY}_", "").split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
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
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save state: {e}")


def get_state():
    state = load_state()
    if not state or len(state) <= 1:
        return build_state()
    return state
