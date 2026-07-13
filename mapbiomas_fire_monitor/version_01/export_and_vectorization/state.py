import os
import json
import time
import gcsfs

BUCKET = "mapbiomas-fire"
GEE_PROJECT = "ee-ipam"
IMAGE_COLLECTION = "projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1"
TILES_PREFIX = "monitor/monthly_images/temp"
MOSAIC_PREFIX = "monitor/monthly_images/monthly_burned"
VECTOR_PREFIX = "monitor/monthly_vectors/monthly_burned"
VECTOR_ASSET_PREFIX = "projects/mapbiomas-workspace/FOGO/MONITORAMENTO/fire_monitor_v1_monthly_burned_brazil_vectors"

STATE_FILE = "monitor_state.json"
START_YEAR = 2019

_fs = None

def _get_fs():
    global _fs
    if _fs is None:
        _fs = gcsfs.GCSFileSystem(token='google_default')
    return _fs


def tile_pattern(year, month):
    return f"fire_monitor_v1_monthly_burned_brazil_{year}_{month:02d}"


def mosaic_name(year, month):
    return f"monthly_burned-brazil_{year}_{month:02d}"


def vector_name(year, month):
    return f"monthly_burned-brazil_{year}_{month:02d}"


def list_months_in_collection():
    import ee
    try:
        col = ee.ImageCollection(IMAGE_COLLECTION)
        dates = col.aggregate_array('system:index').getInfo()
        months = set()
        for d in dates:
            d_str = str(d)
            y, m = None, None
            if len(d_str) == 6 and d_str.isdigit():
                y, m = d_str[:4], d_str[4:]
            elif '_' in d_str:
                parts = d_str.split('_')
                for p in parts:
                    if len(p) == 6 and p.isdigit():
                        y, m = p[:4], p[4:]
                        break
                    elif len(p) == 4 and p.isdigit():
                        y = p
                    elif len(p) == 2 and p.isdigit():
                        m = p
            if y and m and len(y) == 4 and 1 <= int(m) <= 12:
                months.add(f"{y}_{m}")
        return sorted(months, reverse=True)
    except Exception:
        return []


def scan_gcs(logger=None):
    fs = _get_fs()
    state = {}

    def _log(msg):
        if logger:
            logger(msg)

    _log(f"Scanning GCS: gs://{BUCKET}/{TILES_PREFIX}/ ...")
    try:
        tile_files = fs.glob(f"{BUCKET}/{TILES_PREFIX}/fire_monitor_v1_monthly_burned_brazil_*.tif")
        for f in tile_files:
            basename = f.split('/')[-1]
            parts = basename.replace('fire_monitor_v1_monthly_burned_brazil_', '').split('_')
            if len(parts) >= 2:
                year, month = parts[0], parts[1][:2]
                key = f"{year}_{month}"
                if key not in state:
                    state[key] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
                state[key]["exported"] = True
    except Exception as e:
        _log(f"Error scanning tiles: {e}")

    _log(f"Scanning GCS: gs://{BUCKET}/{MOSAIC_PREFIX}/ ...")
    try:
        mosaic_files = fs.glob(f"{BUCKET}/{MOSAIC_PREFIX}/monthly_burned-brazil_*.tif")
        for f in mosaic_files:
            basename = f.split('/')[-1]
            name = basename.replace('monthly_burned-brazil_', '').replace('.tif', '')
            parts = name.split('_')
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                if key not in state:
                    state[key] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
                state[key]["mosaiced"] = True
    except Exception as e:
        _log(f"Error scanning mosaics: {e}")

    _log(f"Scanning GCS: gs://{BUCKET}/{VECTOR_PREFIX}/ ...")
    try:
        vector_files = fs.glob(f"{BUCKET}/{VECTOR_PREFIX}/monthly_burned-brazil_*.shp")
        for f in vector_files:
            basename = f.split('/')[-1]
            name = basename.replace('monthly_burned-brazil_', '').replace('.shp', '')
            parts = name.split('_')
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                if key not in state:
                    state[key] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
                state[key]["vectorized_gcs"] = True
    except Exception as e:
        _log(f"Error scanning vectors: {e}")

    return state


def scan_gee(logger=None):
    import ee
    state = {}

    def _log(msg):
        if logger:
            logger(msg)

    _log("Scanning GEE assets...")
    try:
        assets = ee.data.listAssets({"parent": VECTOR_ASSET_PREFIX})
        for a in assets.get("assets", []):
            asset_name = a["name"].split("/")[-1]
            parts = asset_name.replace("monthly_burned-brazil_", "").split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                if key not in state:
                    state[key] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
                state[key]["vectorized_gee"] = True

        page_token = assets.get("nextPageToken")
        while page_token:
            assets = ee.data.listAssets({"parent": VECTOR_ASSET_PREFIX, "pageToken": page_token})
            for a in assets.get("assets", []):
                asset_name = a["name"].split("/")[-1]
                parts = asset_name.replace("monthly_burned-brazil_", "").split("_")
                if len(parts) >= 2:
                    key = f"{parts[0]}_{parts[1]}"
                    if key not in state:
                        state[key] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
                    state[key]["vectorized_gee"] = True
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
