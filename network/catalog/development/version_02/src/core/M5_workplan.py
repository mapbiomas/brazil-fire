import os
import json
import time
import datetime
from M0_auth_config import CONFIG, GLOBAL_OPTS, _get_fs, gcs_classifications_path, gcs_models_path, get_model_scale

# --- Stale lock cleanup on startup ---
_lock_path_stale = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'm5_workplan.json.lock'
)
if os.path.exists(_lock_path_stale):
    try:
        os.remove(_lock_path_stale)
        print(f"[INFO] Removed stale lock: {_lock_path_stale}")
    except OSError:
        pass

def _lock_path():
    return get_workplan_file() + '.lock'

def _acquire_lock(timeout=5.0):
    lock = _lock_path()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(0.05)
    return False

def _release_lock():
    lock = _lock_path()
    try:
        os.remove(lock)
    except FileNotFoundError:
        pass

def get_workplan_file():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, 'm5_workplan.json')

def load_workplan():
    plan_file = get_workplan_file()
    if not _acquire_lock():
        print("[WARN] Workplan busy, returning empty")
        return []
    try:
        if os.path.exists(plan_file):
            with open(plan_file, 'r') as f:
                return json.load(f)
        return []
    finally:
        _release_lock()

def save_workplan(plan):
    plan_file = get_workplan_file()
    if not _acquire_lock():
        print("[WARN] Workplan busy, save skipped")
        return False
    try:
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        return True
    finally:
        _release_lock()

def make_job_id(model, region, period, campaign=None):
    parts = [model, region, period]
    if campaign:
        parts.insert(0, campaign)
    return " | ".join(parts)

def new_job(model, region, period, task_name=''):
    campaign = CONFIG.get('campaign', 'MONITOR_01')
    return {
        'id': make_job_id(model, region, period, campaign),
        'model': model,
        'region': region,
        'period': period,
        'task_name': task_name,
        'campaign': campaign,
        'status': 'PENDING',
        'enabled': True,
        'upload_gee': False,
        'progress': '0%'
    }

def classifications_base(model_id, campaign=None):
    if campaign is None or campaign == CONFIG.get('campaign', 'MONITOR_01'):
        base = gcs_classifications_path()
    else:
        base = gcs_classifications_path(campaign)
    return f"{base}/{model_id}" if model_id else f"{base}/"

def classified_tiles_dir(model_id, campaign=None):
    return f"{classifications_base(model_id, campaign)}/CLASSIFIED_TILES"

def tile_path(model_id, region, cell_id, period, campaign=None):
    return f"{classified_tiles_dir(model_id, campaign)}/tile_{region}_{cell_id}_{period}.tif"

def classified_region_dir(model_id, campaign=None):
    return f"{classifications_base(model_id, campaign)}/CLASSIFIED_REGION"

def region_path(model_id, region, period, campaign=None):
    return f"{classified_region_dir(model_id, campaign)}/region_{region}_{model_id}_{period}.tif"

def stats_dir(model_id, campaign=None):
    return f"{classifications_base(model_id, campaign)}/STATS"

def tile_stats_path(model_id, campaign=None):
    return f"{stats_dir(model_id, campaign)}/stats_tile.csv"

def region_stats_path(model_id, campaign=None):
    return f"{stats_dir(model_id, campaign)}/stats_region.csv"

def consolidated_stats_path(campaign=None):
    return f"{classifications_base('', campaign)}consolidated_stats.csv"

def gcs_full(relative_path):
    """Path completo con bucket para operaciones GCSFS."""
    return f"{CONFIG['bucket']}/{relative_path}"


# --- TAREAS COMPARTIDAS (GCS) ---

def tareas_dir():
    """Directorio GCS donde se almacenan las tareas compartidas."""
    return f"{CONFIG['gcs_cache']}/tareas"

def tarea_path(model_id):
    """Path GCS relativo para la tarea de un modelo."""
    return f"{tareas_dir()}/{model_id}.json"

def save_tarea(model_id, regions, periods, fs=None):
    """Guarda una tarea compartida en GCS."""
    import json
    import datetime
    from M_gcs import write_json, mkdir, exists
    tarea = {
        'model': model_id,
        'regions': sorted(set(regions)),
        'periods': sorted(set(periods)),
        'created_at': datetime.datetime.now().isoformat(timespec='seconds')
    }
    path = gcs_full(tarea_path(model_id))
    dir_path = path.rsplit('/', 1)[0]
    if not exists(dir_path):
        mkdir(dir_path)
    write_json(path, tarea)
    return path

def delete_tarea(model_id, fs=None):
    """Elimina una tarea compartida de GCS."""
    from M_gcs import exists, rm
    path = gcs_full(tarea_path(model_id))
    try:
        if exists(path):
            rm(path)
    except Exception:
        pass

def list_tareas(fs=None):
    """Lista todas las tareas compartidas en GCS. Retorna lista de dicts."""
    import json
    from M0_auth_config import _get_fs
    if fs is None:
        fs = _get_fs()
    dir_path = gcs_full(tareas_dir())
    results = []
    try:
        files = fs.glob(f"{dir_path}/*.json")
    except Exception:
        return results
    for fp in sorted(files):
        try:
            with fs.open(fp, 'r') as f:
                data = json.load(f)
            results.append(data)
        except Exception:
            pass
    return results


# --- GCS WORKPLAN (PENDING / ARCHIVED POR MODELO) ---

def _workplan_dir(model_id):
    """GCS relativo: {campaign}/LIBRARY_MODELS/<model_id>/workplan/"""
    return f"{gcs_models_path()}/{model_id}/workplan"

def _pending_dir(model_id):
    return f"{_workplan_dir(model_id)}/pending"

def _archived_dir(model_id):
    return f"{_workplan_dir(model_id)}/archived"

def _pending_job_filename(region, period):
    return f"pend_{period}_{region}.json"

def _archived_job_filename(region, period, timestamp):
    return f"arch_{period}_{region}_{timestamp}.json"

def _ensure_dir(fs, path):
    """Cria diretório GCS se não existir."""
    from M_gcs import exists, mkdir as gcs_mkdir
    full = gcs_full(path)
    if not fs.exists(full):
        dir_parent = full.rsplit('/', 1)[0]
        if not fs.exists(dir_parent):
            gcs_mkdir(dir_parent)
        gcs_mkdir(full)

def save_pending_job_to_gcs(job, fs=None):
    """Salva um job em pending/ no GCS.

    Args:
        job: dict com pelo menos model, region, period, id, task_name, campaign, ...
        fs: opcional, gcsfs instance.
    Returns:
        str: caminho GCS completo onde foi salvo, ou None se erro.
    """
    from M_gcs import write_json
    if fs is None:
        fs = _get_fs()
    model_id = job['model']
    region = job['region']
    period = job['period']
    rel_path = _pending_dir(model_id) + '/' + _pending_job_filename(region, period)
    full = gcs_full(rel_path)
    _ensure_dir(fs, _pending_dir(model_id))
    payload = dict(job)
    payload['_saved_at'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        write_json(full, payload)
        return full
    except Exception:
        return None

def delete_pending_job_gcs(model_id, region, period, fs=None):
    """Remove um job de pending/ no GCS."""
    from M_gcs import exists, rm
    rel = _pending_dir(model_id) + '/' + _pending_job_filename(region, period)
    full = gcs_full(rel)
    try:
        if exists(full):
            rm(full)
            return True
    except Exception:
        pass
    return False

def archive_job_on_gcs(job, tile_results, fs=None):
    """Move de pending/ → archived/ com metadados de conclusão.

    Se o job não existir em pending/ (nunca foi salvo), apenas retorna False.
    """
    import json
    from M_gcs import write_json, rm as gcs_rm, exists as gcs_exists
    if fs is None:
        fs = _get_fs()
    model_id = job['model']
    region = job['region']
    period = job['period']
    pend_rel = _pending_dir(model_id) + '/' + _pending_job_filename(region, period)
    pend_full = gcs_full(pend_rel)
    if not fs.exists(pend_full):
        return False
    try:
        with fs.open(pend_full, 'r') as f:
            payload = json.load(f)
    except Exception:
        return False
    total_pixels = sum(tr.get('total_pixels', 0) for tr in tile_results)
    burned_pixels = sum(tr.get('burned_pixels', 0) for tr in tile_results)
    confidences = [tr.get('mean_confidence', 0) for tr in tile_results if tr.get('burned_pixels', 0) > 0]
    resolution_m = get_model_scale(model_id)
    pixel_area_m2 = resolution_m ** 2
    burned_area_km2 = burned_pixels * pixel_area_m2 / 1_000_000
    timestamp = datetime.datetime.utcnow().strftime('T%Y%m%dT%H%M%SZ')
    payload['status'] = 'FINISHED'
    payload['progress'] = '100%'
    payload['_finished_at'] = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    payload['_completed_tiles'] = len(tile_results)
    payload['_total_pixels'] = total_pixels
    payload['_burned_pixels'] = burned_pixels
    payload['_burned_area_km2'] = round(burned_area_km2, 4)
    payload['_mean_confidence'] = round(float(sum(confidences) / len(confidences)), 4) if confidences else 0.0
    arch_rel = _archived_dir(model_id) + '/' + _archived_job_filename(region, period, timestamp)
    arch_full = gcs_full(arch_rel)
    _ensure_dir(fs, _archived_dir(model_id))
    try:
        write_json(arch_full, payload)
    except Exception:
        return False
    try:
        gcs_rm(pend_full)
    except Exception:
        pass
    return True

def load_pending_from_gcs(model_id, fs=None):
    """Lista todos os jobs em pending/ de um modelo. Retorna list[dict]."""
    import json
    if fs is None:
        fs = _get_fs()
    pattern = gcs_full(_pending_dir(model_id)) + '/*.json'
    results = []
    try:
        files = fs.glob(pattern)
    except Exception:
        return results
    for fp in sorted(files):
        try:
            with fs.open(fp, 'r') as f:
                job = json.load(f)
            job['_saved'] = True
            results.append(job)
        except Exception:
            pass
    return results

def load_all_pending_from_gcs(fs=None):
    """Varre todos os modelos, retorna dict {model_id: [jobs...]}."""
    import json
    if fs is None:
        fs = _get_fs()
    # Ex: gs://bucket/sudamerica/peru/CATALOG_01/LIBRARY_MODELS/*/workplan/pending/*.json
    pattern = gcs_full(gcs_models_path()) + '/*/workplan/pending/*.json'
    results = {}
    try:
        files = fs.glob(pattern)
    except Exception:
        return results
    for fp in sorted(files):
        try:
            with fs.open(fp, 'r') as f:
                job = json.load(f)
            model = job.get('model', '')
            if model:
                job['_saved'] = True
                results.setdefault(model, []).append(job)
        except Exception:
            pass
    return results

def list_archived_jobs(model_id, fs=None):
    """Lista jobs em archived/ de um modelo. Retorna list[dict]."""
    import json
    if fs is None:
        fs = _get_fs()
    pattern = gcs_full(_archived_dir(model_id)) + '/*.json'
    results = []
    try:
        files = fs.glob(pattern)
    except Exception:
        return results
    for fp in sorted(files):
        try:
            with fs.open(fp, 'r') as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results

def delete_archived_job(model_id, gcs_path, fs=None):
    """Remove um job específico de archived/."""
    from M_gcs import exists, rm
    try:
        if exists(gcs_path):
            rm(gcs_path)
            return True
    except Exception:
        pass
    return False

def sync_gcs_to_local_workplan(fs=None):
    """Sincroniza jobs do GCS pending/ para o m5_workplan.json local.

    Para cada modelo com pendentes no GCS, adiciona ao m5_workplan.json
    jobs que ainda não existem (por id). Marca como _saved=True.
    """
    if fs is None:
        fs = _get_fs()
    plan = load_workplan()
    existing_ids = set(j['id'] for j in plan)
    added = 0
    all_pending = load_all_pending_from_gcs(fs=fs)
    for model, jobs in all_pending.items():
        for pj in jobs:
            if pj['id'] not in existing_ids:
                pj['_saved'] = True
                plan.append(pj)
                existing_ids.add(pj['id'])
                added += 1
    if added > 0:
        save_workplan(plan)
    return added
