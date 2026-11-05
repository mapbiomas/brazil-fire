import os
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, HTML
from M0_auth_config import CONFIG, _get_fs, _gcs_models_base, _gcs_download, _gcs_upload, get_temp_dir
from M_cache import CacheManager
from M5_workplan import load_workplan, save_workplan, make_job_id, tile_path, gcs_full, archive_job_on_gcs, delete_pending_job_gcs
from M5_inference import load_model_from_gcs, classify_cell_with_cogs, build_band_paths
from M_lang import L as Lang
from M_periods import parse_period
from M_regions import REGION_NAME_PROPERTY
from M_ui_components import THEME, _fmt_time

_C_INFO = THEME['INFO']
_C_ERROR = THEME['ERROR']

_log_lock = threading.Lock()

def _log(out, msg):
    """Exibe mensagem no Output widget. [ERROR] em vermelho, [WARN] em laranja."""
    from IPython.display import display, HTML
    with _log_lock, out:
        if '[ERROR]' in msg:
            display(HTML('<span style="color:{};">{}</span>'.format(THEME['ERROR'], msg)))
        elif '[WARN' in msg:
            display(HTML('<span style="color:{};">{}</span>'.format(THEME['WARNING'], msg)))
        else:
            print(msg)

def _auto_workers():
    return max(1, os.cpu_count() or 1)

def run_m5_workplan(progress_callback=None, n_workers=None):
    """Motor de clasificacion M5.

    Procesa todos los jobs PENDING del workplan, agrupados por (modelo, periodo).
    No incluye mosaicado ni publicacion — use M6_publisher.run_m6_publish().

    Args:
        progress_callback: opcional, llamado tras cada tile.
            firma: progress_callback(model, region, cell_id, i, total, status)
        n_workers: workers paralelos (default: auto detecta via os.cpu_count())
    """
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets

    if n_workers is None:
        n_workers = _auto_workers()
    n_workers = max(1, n_workers)

    out = widgets.Output()
    display(out)
    print(f"Workers: {n_workers}")

    plan = load_workplan()
    _run_classification(plan, out, progress_callback, n_workers)


def _run_classification(plan, out, progress_callback=None, n_workers=None):
    """Fase 1: agrupa jobs por (modelo, periodo) y clasifica compartiendo COGs."""
    pending = [j for j in plan if j['status'] == 'PENDING' and j.get('enabled', True)]

    if not pending:
        with out:
            clear_output()
            display(HTML(f"<b style='color:{THEME['SUCCESS']};'>{Lang.NO_PENDING_JOBS}</b>"))
        return

    with out:
        print("--- Phase 1: Classification ---")

    groups = defaultdict(list)
    for j in pending:
        groups[(j['model'], j['period'])].append(j)

    total_groups = len(groups)
    
    # Calcula total de tiles em todos os grupos
    total_tiles_all_groups = 0
    groups_cells_count = {}
    for (model_id, period), group_jobs in groups.items():
        group_total = 0
        for job in group_jobs:
            region = job['region']
            try:
                cells = _get_region_cells(region)
                group_total += len(cells)
            except Exception as e:
                with out:
                    print(f"  [WARN] Tile count failed for {region}: {e}")
        groups_cells_count[(model_id, period)] = group_total
        total_tiles_all_groups += group_total
    with out:
        print(f"  Total tiles across all groups: {total_tiles_all_groups}")
    
    global_start_time = time.time()
    global_completed = 0
    
    for g_idx, ((model_id, period), group) in enumerate(groups.items()):
        with out:
            print(f"\nGroup {g_idx+1}/{total_groups}: model '{model_id}' | period {period} | {len(group)} region(s)")

        try:
            _process_period(model_id, period, group, out, progress_callback=progress_callback, 
                           n_workers=n_workers,
                           g_idx=g_idx, total_groups=total_groups,
                           global_start_time=global_start_time, 
                           global_completed=global_completed,
                           total_tiles_all_groups=total_tiles_all_groups)
            
            # Atualiza progresso global
            global_completed += groups_cells_count.get((model_id, period), 0)
        except Exception as e:
            import traceback
            _log(out, f"[ERROR] [FATAL] Group '{model_id}' | {period} failed:")
            if 'metadata.json not found' in str(e):
                _log(out, f"[ERROR] Model '{model_id}' not found in GCS. It may have been deleted. Click 'Sync Data' to refresh the model list.")
            else:
                _log(out, f"[ERROR] {e}")
                _log(out, traceback.format_exc())
            p = load_workplan()
            for job in group:
                for pj in p:
                    if pj['id'] == job['id']:
                        pj['status'] = 'FAILED'
            save_workplan(p)
            global_completed += groups_cells_count.get((model_id, period), 0)
            continue

    # A finished classification run changes what M6 should discover next.
    try:
        from M6_ui import invalidate_discovery_cache
        invalidate_discovery_cache()
    except Exception:
        pass


def _classify_one_tile(cell, model_id, region_name, period, campaign,
                        predict_fn, bands_config, norm_stats, band_paths, band_order,
                        out, worker_id, fs):
    cell_id = cell['name']
    out_rel = tile_path(model_id, region_name, cell_id, period, campaign)
    out_full = gcs_full(out_rel)

    if fs.exists(out_full):
        _log(out, f"    [W{worker_id}] {cell_id}: already exists, skipped")
        return ('skipped', cell_id, region_name, None, worker_id)

    _log(out, f"    [W{worker_id}] >>> {cell_id}")
    try:
        stats = classify_cell_with_cogs(
            cell_id, predict_fn, bands_config, norm_stats,
            out_full, band_paths, band_order,
            logger=lambda m: _log(out, m), worker_id=worker_id
        )
    except Exception as e:
        _log(out, f"    [W{worker_id}] {cell_id}: [ERROR] {e}")
        return ('error', cell_id, region_name, None, worker_id)

    if stats:
        stats['tile_id'] = cell_id
        stats['region'] = region_name
        stats['period'] = period
        return ('done', cell_id, region_name, stats, worker_id)
    else:
        _log(out, f"    [W{worker_id}] <<< {cell_id} warn: no stats")
        return ('warn', cell_id, region_name, None, worker_id)


def _process_period(model_id, period, group_jobs, out, progress_callback=None, n_workers=None, g_idx=0, total_groups=1,
                    global_start_time=None, global_completed=0, total_tiles_all_groups=0):
    """Procesa todas las regiones de un (modelo, periodo) compartiendo COGs.

    Los COGs se descargan una unica vez y se reusan para todos los tiles
    de todas las regiones del grupo.
    """
    import ee
    from M0_auth_config import authenticate, _gcs_models_base

    authenticate()
    fs = _get_fs()

    # Extrai campanha do primeiro job (todos no grupo compartilham o mesmo periodo)
    campaign = group_jobs[0].get('campaign', '') if group_jobs else ''

    period_info = parse_period(period)

    # 1. Cargar modelo + metadatos (una vez)
    with out:
        print(f"  Loading model '{model_id}'...")

    model_dir = f"gs://{CONFIG['bucket']}/{_gcs_models_base()}/{model_id}"
    predict_fn, meta, bands_config, norm_stats, band_order = load_model_from_gcs(
        model_dir, fs, logger=lambda m: _log(out, m)
    )

    # 2. Construir paths e verificar COGs via CacheManager primeiro
    band_paths = build_band_paths(bands_config, period)

    known_cogs = set(CacheManager.cogs_for(period_info.periodicity))

    with out:
        print(f"  Checking COGs for period '{period}'...")
    for b, p in band_paths.items():
        # Extrai o nome base (ex: image_peru_fire_sentinel2_minnbr_2025_08_blue)
        cog_basename = os.path.basename(p).lower().replace('_cog.tif', '')
        in_cache = cog_basename in known_cogs
        exists = "OK (cache)" if in_cache else ("OK" if fs.exists(f"gs://{p}") else "MISSING")
        with out:
            print(f"    {b}: {exists}")

    first_full = f"gs://{band_paths[band_order[0]]}"
    first_basename = os.path.basename(first_full).lower().replace('_cog.tif', '')
    if first_basename not in known_cogs and not fs.exists(first_full):
        raise FileNotFoundError(f"COG not found at: {first_full}")

    with out:
        print(f"  COGs opened via streaming ({len(band_paths)} bands)")

    # 4. Pre-computar celdas de todas as regiões para progresso cumulativo
    region_cells_map = {}
    for job in group_jobs:
        region_cells_map[job['region']] = _get_region_cells(job['region'])
    total_cells_group = sum(len(cells) for cells in region_cells_map.values())
    processed = 0
    _t0 = time.time()
    _done = 0
    _completed_cards = 0
    _total_cards = len(group_jobs)
    
    # Rastreamento de pixels para ETA mais preciso
    _pixels_processed = 0
    _avg_pixels_per_tile = 0
    _pixels_remaining_estimate = 0

    # 5. Procesar cada region del grupo
    for job_idx, job in enumerate(group_jobs):
        region_name = job['region']

        job['status'] = 'RUNNING'
        p = load_workplan()
        for pj in p:
            if pj['id'] == job['id']:
                pj['status'] = 'RUNNING'
        save_workplan(p)

        with out:
            print(f"\n  Region {job_idx+1}/{len(group_jobs)}: {region_name}")

        cells = region_cells_map[region_name]
        if not cells:
            with out:
                print(f"  [WARN] No cells found for {region_name}")
            continue

        total = len(cells)
        with out:
            print(f"  {total} tiles to process.")

        tile_results = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for i, cell in enumerate(cells):
                worker_id = (i % n_workers) + 1
                future = executor.submit(
                    _classify_one_tile,
                    cell, model_id, region_name, period, campaign,
                    predict_fn, bands_config, norm_stats, band_paths, band_order,
                    out, worker_id, fs
                )
                futures[future] = cell['name']

            completed_in_region = 0
            _last_progress_log = time.time()
            for future in as_completed(futures):
                status, cell_id, reg, stats, wid = future.result()
                completed_in_region += 1
                now = time.time()
                tile_elapsed = (stats.get('tile_elapsed', 0) if stats else 0)

                if status == 'skipped':
                    _done += 1
                    if progress_callback:
                        progress_callback(model_id, reg, cell_id, completed_in_region, total, 'skipped')
                elif status == 'error':
                    _done += 1
                    if progress_callback:
                        progress_callback(model_id, reg, cell_id, completed_in_region, total, 'error')
                elif status == 'warn':
                    if progress_callback:
                        progress_callback(model_id, reg, cell_id, completed_in_region, total, 'error')
                else:
                    # status == 'done'
                    _done += 1
                    tile_results.append(stats)
                    if progress_callback:
                        progress_callback(model_id, reg, cell_id, completed_in_region, total, 'done')
                    
                    # Atualiza rastreamento de pixels
                    tile_pixels = stats.get('total_pixels', 0) if stats else 0
                    _pixels_processed += tile_pixels
                    if _done > 0:
                        _avg_pixels_per_tile = _pixels_processed / _done
                    
                    group_elapsed = now - _t0
                    pixels_per_sec = _pixels_processed / max(group_elapsed, 1)
                    pixels_remaining = max(0, (_avg_pixels_per_tile * (total_cells_group - _done))) if _avg_pixels_per_tile > 0 else 0
                    group_remaining = pixels_remaining / pixels_per_sec if pixels_per_sec > 0 else 0
                    group_total = group_elapsed + group_remaining
                    
                    # Calcula tempo global considerando TODOS os períodos
                    if global_start_time is not None and total_tiles_all_groups > 0:
                        global_elapsed = now - global_start_time
                        # Estima pixels globais
                        global_pixels_processed = (_avg_pixels_per_tile * global_completed) + _pixels_processed if _avg_pixels_per_tile > 0 else 0
                        global_pixels_per_sec = global_pixels_processed / max(global_elapsed, 1)
                        global_pixels_remaining = (_avg_pixels_per_tile * (total_tiles_all_groups - (global_completed + _done))) if _avg_pixels_per_tile > 0 else 0
                        global_remaining = global_pixels_remaining / global_pixels_per_sec if global_pixels_per_sec > 0 else 0
                        global_total = global_elapsed + global_remaining
                        remaining_str = f" | total global ~{_fmt_time(global_total)} | remaining ~{_fmt_time(global_remaining)}"
                    else:
                        remaining_str = f" | total ~{_fmt_time(group_total)} | remaining ~{_fmt_time(group_remaining)}"
                    
                    px_rate = f" [{int(pixels_per_sec):,} px/s]" if pixels_per_sec > 0 else ""
                    _log(out, f"    [W{wid}] <<< {cell_id} done in {_fmt_time(tile_elapsed)} ({tile_pixels:,} px){px_rate}")
                    _log(out, f"    [W{wid}]     tile {completed_in_region}/{total}"
                          f" | period {g_idx+1}/{total_groups}"
                          f" | region {job_idx+1}/{_total_cards}"
                          f" | group {_done}/{total_cells_group}")
                    _log(out, f"    [W{wid}]     elapsed group {_fmt_time(group_elapsed)}"
                          + remaining_str)

                # Log de progresso a cada 5 tiles ou no último
                if completed_in_region == total or now - _last_progress_log >= 15 or completed_in_region % 5 == 0:
                    _last_progress_log = now
                    group_elapsed = now - _t0
                    pixels_per_sec = _pixels_processed / max(group_elapsed, 1)
                    pixels_remaining = max(0, (_avg_pixels_per_tile * (total_cells_group - _done))) if _avg_pixels_per_tile > 0 else 0
                    remaining = pixels_remaining / pixels_per_sec if pixels_per_sec > 0 else 0
                    total_proj = group_elapsed + remaining
                    
                    # Global time estimate
                    if global_start_time is not None and total_tiles_all_groups > 0:
                        global_elapsed = now - global_start_time
                        global_pixels_processed = (_avg_pixels_per_tile * global_completed) + _pixels_processed if _avg_pixels_per_tile > 0 else 0
                        global_pixels_per_sec = global_pixels_processed / max(global_elapsed, 1)
                        global_pixels_remaining = (_avg_pixels_per_tile * (total_tiles_all_groups - (global_completed + _done))) if _avg_pixels_per_tile > 0 else 0
                        global_remaining = global_pixels_remaining / global_pixels_per_sec if global_pixels_per_sec > 0 else 0
                        global_total = global_elapsed + global_remaining
                        with out:
                            display(HTML(f'<span style="color:{_C_INFO};">  > Group progress: {_done}/{total_cells_group} tiles ({_pixels_processed:,} px @ {int(pixels_per_sec):,} px/s) | elapsed {_fmt_time(group_elapsed)} | total ~{_fmt_time(total_proj)} | remaining ~{_fmt_time(remaining)}</span>'))
                            display(HTML(f'<span style="color:{_C_INFO};">  > GLOBAL: {global_completed + _done}/{total_tiles_all_groups} tiles ({int(global_pixels_processed):,} px @ {int(global_pixels_per_sec):,} px/s) | elapsed {_fmt_time(global_elapsed)} | total ~{_fmt_time(global_total)} | remaining ~{_fmt_time(global_remaining)}</span>'))
                    else:
                        with out:
                            display(HTML(f'<span style="color:{_C_INFO};">  > Group progress: {_done}/{total_cells_group} tiles ({_pixels_processed:,} px @ {int(pixels_per_sec):,} px/s) | elapsed {_fmt_time(group_elapsed)} | total ~{_fmt_time(total_proj)} | remaining ~{_fmt_time(remaining)}</span>'))

            # Final remaining for region (guaranteed)
            elapsed = time.time() - _t0
            pixels_per_sec = _pixels_processed / max(elapsed, 1)
            pixels_remaining = max(0, (_avg_pixels_per_tile * (total_cells_group - _done))) if _avg_pixels_per_tile > 0 else 0
            remaining = pixels_remaining / pixels_per_sec if pixels_per_sec > 0 else 0
            total_proj = elapsed + remaining
            
            # Global final time estimate
            if global_start_time is not None and total_tiles_all_groups > 0:
                global_elapsed = time.time() - global_start_time
                global_pixels_processed = (_avg_pixels_per_tile * global_completed) + _pixels_processed if _avg_pixels_per_tile > 0 else 0
                global_pixels_per_sec = global_pixels_processed / max(global_elapsed, 1)
                global_pixels_remaining = (_avg_pixels_per_tile * (total_tiles_all_groups - (global_completed + _done))) if _avg_pixels_per_tile > 0 else 0
                global_remaining = global_pixels_remaining / global_pixels_per_sec if global_pixels_per_sec > 0 else 0
                global_total = global_elapsed + global_remaining
                with out:
                    display(HTML(f'<span style="color:{_C_INFO};">  > Group progress: {_done}/{total_cells_group} tiles ({_pixels_processed:,} px @ {int(pixels_per_sec):,} px/s) | elapsed {_fmt_time(elapsed)} | total ~{_fmt_time(total_proj)} | remaining ~{_fmt_time(remaining)}</span>'))
                    display(HTML(f'<span style="color:{_C_INFO};">  > GLOBAL: {global_completed + _done}/{total_tiles_all_groups} tiles ({int(global_pixels_processed):,} px @ {int(global_pixels_per_sec):,} px/s) | elapsed {_fmt_time(global_elapsed)} | total ~{_fmt_time(global_total)} | remaining ~{_fmt_time(global_remaining)}</span>'))
            else:
                with out:
                    display(HTML(f'<span style="color:{_C_INFO};">  > Group progress: {_done}/{total_cells_group} tiles ({_pixels_processed:,} px @ {int(pixels_per_sec):,} px/s) | elapsed {_fmt_time(elapsed)} | total ~{_fmt_time(total_proj)} | remaining ~{_fmt_time(remaining)}</span>'))

            # Salva progresso no workplan
            p = load_workplan()
            pct = (processed + completed_in_region) / total_cells_group if total_cells_group else 0
            for pj in p:
                if pj['id'] == job['id']:
                    pj['progress'] = f"{_done}/{total_cells_group} ({pct:.1%})"
            save_workplan(p)

        processed += total

        if tile_results:
            with out:
                print(f"  Archiving job for {len(tile_results)} tiles...")
            archive_job_on_gcs(job, tile_results, fs=fs)
        else:
            with out:
                print(f"  No tiles classified for {region_name}")
            delete_pending_job_gcs(model_id, region_name, period, fs=fs)

        _completed_cards += 1
        p = load_workplan()
        for pj in p:
            if pj['id'] == job['id']:
                pj['status'] = 'COMPLETED'
                pj['progress'] = '100%'
        save_workplan(p)

        with out:
            total_elapsed = time.time() - _t0
            print(f"  OK: {region_name} completed"
                  f" | period {g_idx+1}/{total_groups}"
                  f" | region {_completed_cards}/{_total_cards}"
                  f" | group {_done}/{total_cells_group} tiles"
                  f" | elapsed {_fmt_time(total_elapsed)}")

    total_time = time.time() - _t0
    with out:
        print(f"  --- Group {g_idx+1}/{total_groups} | {model_id} | {period} done in {_fmt_time(total_time)} ---"
              f" | {_done}/{total_cells_group} tiles | {_completed_cards}/{_total_cards} regions")


def _get_region_cells(region_name):
    """Busca celdas cim-world que intersectan la region."""
    import ee
    cim = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")

    region_fc = ee.FeatureCollection(CONFIG['asset_regions']).filter(ee.Filter.eq(REGION_NAME_PROPERTY, region_name))

    intersected = cim.filterBounds(region_fc.geometry())
    try:
        names = intersected.aggregate_array('name').getInfo()
        return [{'name': str(n)} for n in names]
    except Exception as e:
        print(f"[ERROR] Grid search failed: {e}")
        return []
