import os
import csv
import json
import time
import shutil
import subprocess
from datetime import datetime
import numpy as np
import rasterio
from osgeo import gdal
from M0_auth_config import CONFIG, _get_fs, _gcs_download, _gcs_upload, get_temp_dir, gcs_classifications_path, get_model_scale
from M_periods import parse_period
from M5_workplan import (
    classified_tiles_dir, classified_region_dir, region_path,
    consolidated_stats_path, classifications_base,
    tile_path, gcs_full
)
from M_regions import REGION_NAME_PROPERTY
from M_cache import CacheManager
from M_ui_components import _fmt_time, ProgressTracker


def _get_resolution_from_model(model_id):
    """Resolve pixel size from the sensor encoded in the trained model ID."""
    return get_model_scale(model_id)


def _period_export_metadata(period):
    """Return GEE time properties for a period key (YYYY, YYYY_MM, ...)."""
    info = parse_period(period)
    month_s = f"{info.index:02d}" if info.periodicity == 'monthly' else '0'

    return {
        'year': str(info.year),
        'month': month_s,
        'time_start': int(info.start.timestamp() * 1000),
        'time_end': int(info.end.timestamp() * 1000),
        'periodicity': info.periodicity,
        'asset_type': info.asset_type,
    }


def _gcs_exists(path, timeout=15):
    """Check GCS existence via gsutil stat com timeout. Fallback fs.exists()."""
    try:
        ret = subprocess.run(['gsutil', '-q', 'stat', f"gs://{path}"],
                             capture_output=True, timeout=timeout)
        return ret.returncode == 0
    except Exception:
        fs = _get_fs()
        try:
            return fs.exists(path)
        except Exception:
            return False


def merge_region_tiles(model_id, region, period, fs=None, logger=None, campaign=None):
    """Junta todos os tiles de uma regiao em um mosaico regional."""
    if fs is None:
        fs = _get_fs()

    tiles_dir = gcs_full(classified_tiles_dir(model_id, campaign))
    match_prefix = f"tile_{region}_"
    match_suffix = f"_{period}.tif"
    tile_paths = sorted([
        p for p in fs.glob(f"{tiles_dir}/*.tif")
        if os.path.basename(p).startswith(match_prefix)
        and os.path.basename(p).endswith(match_suffix)
        and not p.endswith('.aux.xml')
    ])

    if not tile_paths:
        if logger:
            logger(f"    [WARN] No tiles found for {region}")
        return None

    if logger:
        logger(f"    Merging {len(tile_paths)} tiles for {region}...")

    out_gcs = gcs_full(region_path(model_id, region, period, campaign))
    tmpdir = os.path.join(get_temp_dir('mosaics'), f"merge_{model_id}_{region}_{period}_{int(time.time())}")
    os.makedirs(tmpdir, exist_ok=True)

    try:
        local_tiles = []
        for i, tp in enumerate(tile_paths):
            local = os.path.join(tmpdir, f"tile_{i:04d}.tif")
            _gcs_download(tp, local)
            local_tiles.append(local)

        vrt_path = os.path.join(tmpdir, "mosaic.vrt")
        gdal.BuildVRT(vrt_path, local_tiles,
            options=gdal.BuildVRTOptions(resampleAlg='nearest'))

        mosaic_local = os.path.join(tmpdir, "mosaic.tif")
        gdal.Translate(mosaic_local, vrt_path,
            options=gdal.TranslateOptions(
                creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES']))

        # Nomeia as bandas no GeoTIFF — GEE respeita BandDescription
        try:
            ds = gdal.Open(mosaic_local, 1)
            ds.GetRasterBand(1).SetDescription('probability')
            ds.GetRasterBand(2).SetDescription('dayOfYear')
            ds = None
        except Exception:
            pass

        _gcs_upload(mosaic_local, out_gcs)

        if logger:
            logger(f"    Mosaic saved: {out_gcs}")
        return out_gcs

    finally:
        shutil.rmtree(tmpdir)


def compute_region_stats_from_mosaic(model_id, region, period, fs=None, logger=None, campaign=None):
    """Le o mosaico e computa estatisticas da regiao."""
    if fs is None:
        fs = _get_fs()

    mosaic_gcs = gcs_full(region_path(model_id, region, period, campaign))
    local_tmp = os.path.join(get_temp_dir('stats'), "mosaic_stats.tif")
    try:
        _gcs_download(mosaic_gcs, local_tmp)
    except Exception:
        if logger:
            logger(f"    [WARN] Mosaic not found for stats: {mosaic_gcs}")
        return None

    with rasterio.open(local_tmp) as src:
        prob_band = src.read(1)  # int16, 0-1000
        doy_band = src.read(2)   # int16, 1-366, -9999 nodata

    os.remove(local_tmp)

    total_valid = int((prob_band != -9999).sum())
    # Product convention: any positive probability counts as burned (same rule
    # as the dayOfYear band in M5 and the GEE filters).
    burned_mask = prob_band > 0
    n_burned = int(burned_mask.sum())

    resolution_m = _get_resolution_from_model(model_id)
    pixel_area_m2 = resolution_m ** 2
    burned_area_km2 = n_burned * pixel_area_m2 / 1_000_000
    total_area_km2 = total_valid * pixel_area_m2 / 1_000_000 if total_valid else 0

    if n_burned > 0:
        mean_conf = float(prob_band[burned_mask].mean()) / 1000.0
    else:
        mean_conf = 0.0

    row = {
        'model_id': model_id,
        'region': region,
        'period': period,
        'tiles_total': 0,
        'tiles_processed': 1,
        'total_pixels': total_valid,
        'burned_pixels': n_burned,
        'burned_area_km2': f"{burned_area_km2:.2f}",
        'total_area_km2': f"{total_area_km2:.2f}",
        'burned_percentage': f"{(n_burned / total_valid * 100):.2f}" if total_valid else "0.00",
        'mean_confidence': f"{mean_conf:.4f}",
    }
    return row


def update_consolidated_stats(row, fs=None, logger=None, campaign=None):
    """Adiciona/atualiza uma linha no consolidated_stats.csv da campanha."""
    from M_gcs import mkdir

    consolidated_path = gcs_full(consolidated_stats_path(campaign))
    fieldnames = list(row.keys())

    cons_rows = []
    try:
        local_cons = os.path.join(get_temp_dir('stats'), "consolidated.csv")
        _gcs_download(consolidated_path, local_cons)
        with open(local_cons, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['region'] == row['region'] and r['period'] == row['period'] and r['model_id'] == row['model_id']:
                    continue
                cons_rows.append(r)
        os.remove(local_cons)
    except Exception:
        pass

    local_cons = os.path.join(get_temp_dir('stats'), "consolidated.csv")
    with open(local_cons, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in cons_rows:
            writer.writerow(r)
        writer.writerow(row)

    dir_path = consolidated_path.rsplit('/', 1)[0]
    mkdir(dir_path)
    _gcs_upload(local_cons, consolidated_path)
    os.remove(local_cons)

    if logger:
        logger(f"    Consolidated stats updated for {row['model_id']} | {row['region']} | {row['period']}")


def upload_to_gee(model_id, region, period, fs=None, logger=None, campaign=None, scale=10):
    """Envia o mosaico regional como Image para o GEE via earthengine CLI."""
    import ee

    mosaic_gcs = gcs_full(region_path(model_id, region, period, campaign))
    gcs_uri = f"gs://{mosaic_gcs}"

    # Usa gsutil stat (nao gcsfs.exists) para evitar cache stale
    ret = subprocess.run(['gsutil', '-q', 'stat', gcs_uri], capture_output=True)
    if ret.returncode != 0:
        if logger:
            logger(f"    [WARN] Mosaic not found for GEE upload")
        return False

    if logger:
        logger(f"    Setting up GEE folders...")

    base = CONFIG['gee_campaigns_prefix']
    campaign = campaign or CONFIG.get('campaign', 'MONITOR_01')
    for sub in [f'{campaign}/LIBRARY_CLASSIFICATIONS', f'{campaign}/LIBRARY_CLASSIFICATIONS/REGIONAL']:
        try:
            ee.data.createAsset({'type': 'Folder'}, f"{base}/{sub}")
        except Exception:
            pass

    parent = f"{base}/{campaign}/LIBRARY_CLASSIFICATIONS/REGIONAL/{model_id}"
    try:
        ee.data.createAsset({'type': 'ImageCollection'}, parent)
    except Exception:
        pass

    asset_id = f"{parent}/{region}_{period}"

    # Deletar se ja existir
    try:
        ee.data.getAsset(asset_id)
        if logger:
            logger(f"    Asset already exists, deleting: {asset_id}")
        ee.data.deleteAsset(asset_id)
        time.sleep(2)
    except Exception:
        pass

    time_metadata = _period_export_metadata(period)

    ee_project = CONFIG.get('gee_project', 'mapbiomas-peru')
    country = CONFIG.get('country', 'peru')
    version = CONFIG.get('version', 'version_01')

    cmd = (
        f'earthengine --project {ee_project} upload image '
        f'--asset_id={asset_id} '
        f'--pyramiding_policy=mode '
        f'--time_start {time_metadata["time_start"]} '
        f'--time_end {time_metadata["time_end"]} '
        f'--property source=mapbiomas-fire '
        f'--property model_id={model_id} '
        f'--property region={region} '
        f'--property period={period} '
        f'--property year={time_metadata["year"]} '
        f'--property month={time_metadata["month"]} '
        f'--property campaign={campaign or ""} '
        f'--property country={country} '
        f'--property version={version} '
        f'--property type={time_metadata["asset_type"]} '
        f'--property periodicity={time_metadata["periodicity"]} '
        f'{gcs_uri}'
    )

    if logger:
        logger(f"    Submitting earthengine upload...")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode == 0:
        if logger:
            logger(f"    GEE upload task submitted! Asset: {asset_id}")
        return True
    else:
        if logger:
            logger(f"    [ERROR] GEE upload failed (exit {ret.returncode})")
        return False


def gee_asset_exists(model_id, region, period, campaign=None):
    """Check if GEE asset ja existe."""
    import ee
    camp = campaign or CONFIG.get('campaign', 'MONITOR_01')
    parent = f"{CONFIG['gee_campaigns_prefix']}/{camp}/LIBRARY_CLASSIFICATIONS/REGIONAL/{model_id}"
    asset_id = f"{parent}/{region}_{period}"
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def load_gee_assets(model_id, campaign=None):
    """Retorna set de region_period que ja existem no GEE para este modelo. 1 chamada GEE."""
    import ee
    camp = campaign or CONFIG.get('campaign', 'MONITOR_01')
    parent = f"{CONFIG['gee_campaigns_prefix']}/{camp}/LIBRARY_CLASSIFICATIONS/REGIONAL/{model_id}"
    assets = set()
    try:
        result = ee.data.listAssets({'parent': parent})
        for a in result.get('assets', []):
            name = a['name'].split('/')[-1]
            assets.add(name)
    except Exception:
        pass
    return assets


def load_stats_done(groups, fs=None):
    """Baixa consolidated_stats.csv de cada campanha e retorna set de (model, region, period, campaign) que existem.
    groups: iteravel de (model_id, region, period, campaign)."""
    if fs is None:
        fs = _get_fs()
    done = set()
    for campaign in set(c for _, _, _, c in groups):
        path = gcs_full(consolidated_stats_path(campaign))
        if not fs.exists(path):
            continue
        local_csv = os.path.join(get_temp_dir('stats'), "load_stats_done.csv")
        try:
            _gcs_download(path, local_csv)
            with open(local_csv, 'r') as f:
                for r in csv.DictReader(f):
                    done.add((r['model_id'], r['region'], r['period'], campaign))
        except Exception:
            pass
        finally:
            if os.path.exists(local_csv):
                os.remove(local_csv)
    return done


def compute_region_stats_from_tiles(model_id, region, period, fs=None, logger=None, campaign=None):
    """Agrega stats dos tiles (1 por vez, RAM baixa). Unidades em hectares."""
    if fs is None:
        fs = _get_fs()

    tiles_dir = gcs_full(classified_tiles_dir(model_id, campaign))
    match_prefix = f"tile_{region}_"
    match_suffix = f"_{period}.tif"
    tile_paths = sorted([
        p for p in fs.glob(f"{tiles_dir}/*.tif")
        if os.path.basename(p).startswith(match_prefix)
        and os.path.basename(p).endswith(match_suffix)
        and not p.endswith('.aux.xml')
    ])

    if not tile_paths:
        if logger:
            logger(f"    [WARN] No tiles for stats: {region} {period}")
        return None

    total_valid = 0
    total_burned = 0
    sum_conf = 0.0
    n_tiles = len(tile_paths)

    tmpdir = os.path.join(get_temp_dir('stats'), f"tile_stats_{model_id}_{region}_{period}_{int(time.time())}")
    os.makedirs(tmpdir, exist_ok=True)

    try:
        for i, tp in enumerate(tile_paths):
            local = os.path.join(tmpdir, f"tile_{i:04d}.tif")
            _gcs_download(tp, local)

            with rasterio.open(local) as src:
                prob_band = src.read(1)  # int16, 0-1000

            os.remove(local)

            valid_mask = prob_band != -9999
            n_valid = int(valid_mask.sum())
            # Product convention: any positive probability counts as burned
            # (same rule as the dayOfYear band in M5 and the GEE filters).
            burned_mask = prob_band > 0
            n_burned = int(burned_mask.sum())

            total_valid += n_valid
            total_burned += n_burned
            if n_burned > 0:
                sum_conf += float(prob_band[burned_mask].sum())

            if logger and (i + 1) % 5 == 0:
                logger(f"    Stats tile {i+1}/{n_tiles} ({n_burned} burned / {n_valid} px)")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    resolution_m = _get_resolution_from_model(model_id)
    pixel_area_m2 = resolution_m ** 2
    burned_area_ha = total_burned * pixel_area_m2 / 10000
    total_area_ha = total_valid * pixel_area_m2 / 10000 if total_valid else 0
    mean_conf = sum_conf / total_burned / 1000.0 if total_burned > 0 else 0.0

    row = {
        'model_id': model_id,
        'region': region,
        'period': period,
        'tiles_total': n_tiles,
        'tiles_processed': n_tiles,
        'total_pixels': total_valid,
        'burned_pixels': total_burned,
        'burned_area_ha': f"{burned_area_ha:.2f}",
        'total_area_ha': f"{total_area_ha:.2f}",
        'burned_percentage': f"{(total_burned / total_valid * 100):.2f}" if total_valid else "0.00",
        'mean_confidence': f"{mean_conf:.4f}",
    }
    return row


def stats_row_exists(model_id, region, period, campaign=None, fs=None):
    """Check if consolidated_stats.csv ja tem linha para (model, region, period)."""
    path = gcs_full(consolidated_stats_path(campaign))
    if not _gcs_exists(path):
        return False
    local_csv = os.path.join(get_temp_dir('stats'), "check_stats.csv")
    try:
        _gcs_download(path, local_csv)
        with open(local_csv, 'r') as f:
            for r in csv.DictReader(f):
                if r['region'] == region and r['period'] == period and r['model_id'] == model_id:
                    return True
    except Exception:
        return False
    finally:
        if os.path.exists(local_csv):
            os.remove(local_csv)
    return False


def discover_classified_groups(fs=None, logger=None):
    """Scan GCS e descobre grupos (modelo, regiao, periodo, campaign) com tiles classificados."""
    if fs is None:
        fs = _get_fs()

    # Scan ALL campaigns under the prefix, not just CONFIG['campaign']
    bucket = CONFIG.get('gcs_bucket', 'mapbiomas-fire')
    gcs_prefix = CONFIG.get('gcs_campaigns_prefix', 'sudamerica/peru/CATALOG_01')
    base = f"gs://{bucket}/{gcs_prefix}"
    patterns = [
        f"{base}/*/LIBRARY_CLASSIFICATIONS/*/CLASSIFIED_TILES/tile_*.tif",
    ]

    groups = set()
    for pattern in patterns:
        try:
            all_tiles = fs.glob(pattern)
        except Exception as e:
            if logger:
                logger(f"    [ERROR] scanning GCS: {e}")
            continue

        for tp in all_tiles:
            basename = os.path.basename(tp)
            parts = basename.replace('tile_', '', 1).split('_')
            if len(parts) < 3:
                continue
            # Period: mensal (YYYY_MM) ou anual (YYYY)
            last = parts[-1].replace('.tif', '')
            if (len(last) == 2 and last.isdigit() and 1 <= int(last) <= 12
                and len(parts) >= 3 and len(parts[-2]) == 4 and parts[-2].isdigit()):
                period = f"{parts[-2]}_{last}"
                region = '_'.join(parts[:-3])
            else:
                period = last
                region = '_'.join(parts[:-2]) if len(parts) >= 2 else parts[0]
            model_dir = tp.split('/CLASSIFIED_TILES')[0]
            model_id = os.path.basename(model_dir)
            # Extract campaign from actual GCS path, not from current CONFIG
            # (CONFIG may have changed since tiles were written by M5)
            campaign = model_dir.split('/LIBRARY_CLASSIFICATIONS')[0].rstrip('/').split('/')[-1]
            groups.add((model_id, region, period, campaign))

    return groups


def cleanup_old_m5_stats(fs=None, logger=print):
    """Remove stats antigos do M5 (STATS/ + GERAL_STATS/) do GCS."""
    from M_gcs import rm as gcs_rm, exists as gcs_exists
    if fs is None:
        fs = _get_fs()
    base = f"{CONFIG['bucket']}/{gcs_classifications_path()}"

    for model_dir in fs.glob(f"{base}/*/"):
        stats_dir = f"{model_dir.rstrip('/')}/STATS"
        if gcs_exists(stats_dir):
            for f in fs.glob(f"{stats_dir}/*"):
                gcs_rm(f)
            gcs_rm(stats_dir)
            logger(f"  Removed {stats_dir}")

    geral = f"{base}/GERAL_STATS"
    if gcs_exists(geral):
        for f in fs.glob(f"{geral}/*"):
            gcs_rm(f)
        gcs_rm(geral)
        logger(f"  Removed {geral}")

    logger("  Cleanup done.")


def generate_region_thumbnail(region_name, size=128):
    """Retorna base64 PNG estilo M5: Peru outline + grid + regiao destacada."""
    import ee
    import base64
    import requests
    from M0_auth_config import authenticate
    if not getattr(ee.data, '_credentials', None):
        authenticate()

    all_regions = ee.FeatureCollection(CONFIG['asset_regions'])
    sel_region = all_regions.filter(
        ee.Filter.eq(REGION_NAME_PROPERTY, region_name))
    grid = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")

    region_lines = all_regions.style(**{'color': '2980b9', 'width': 1, 'fillColor': '00000000'})
    sel_fill = sel_region.style(**{'color': 'f39c12', 'width': 2, 'fillColor': 'f39c1260'})
    grid_lines = grid.style(**{'color': 'e0e0e0', 'width': 0.3, 'fillColor': '00000000'})

    overlay = ee.ImageCollection([region_lines, sel_fill, grid_lines]).mosaic()
    bounds = all_regions.geometry().bounds(1, 'EPSG:4326')
    try:
        url = overlay.getThumbURL({'region': bounds, 'dimensions': size, 'format': 'png'})
        resp = requests.get(url, timeout=60)
        return base64.b64encode(resp.content).decode('ascii')
    except Exception:
        return None


def run_m6_publish(upload_gee=True, groups=None, ui=None, logger=None):
    """Entry point M6: mosaic, stats, GEE upload.
    
    groups: lista de (model_id, region, period, campaign). Se None, descobre todos.
    ui: M6WorkplanUI opcional. Se fornecido e groups=None, le os checkboxes da UI.
    Se ui=None, tenta auto-detectar _M6_UI_INSTANCE.
    """
    if logger is None:
        logger = print

    fs = _get_fs()

    # --- Descobrir grupos: UI checkboxes > cache > scan GCS ---
    if groups is not None:
        groups = list(groups)
    else:
        if ui is None:
            try:
                from M6_ui import _M6_UI_INSTANCE
                ui = _M6_UI_INSTANCE
            except Exception:
                ui = None

        if ui is not None and getattr(ui, '_publish_checks', None):
            checked = [g for g, cb in ui._publish_checks.items() if cb.value]
            if not checked:
                logger("  No groups checked in 'To Publish'. Select at least one card and try again.")
                return
            groups = checked
            logger(f"  Using {len(groups)} groups from UI selection.")

        if groups is None:
            from M6_ui import _M6_DISCOVERY_CACHE
            if _M6_DISCOVERY_CACHE is not None:
                cached = _M6_DISCOVERY_CACHE
                raw = cached['groups']
                done = cached['mosaics'] & cached['stats_done'] & cached['gee_assets']
                groups = [g for g in raw if g not in done]
                logger(f"  Using {len(groups)} pending groups from cache ({len(raw)} total, {len(done)} complete).")
            else:
                groups = discover_classified_groups(fs=fs, logger=logger)
                logger(f"  Discovered {len(groups)} groups via GCS scan.")

    if not groups:
        logger("  No groups to process.")
        return

    groups = sorted(groups)
    _t0 = time.time()
    prog = ProgressTracker(len(groups), "M6 Publish")

    for idx, (model_id, region, period, campaign) in enumerate(groups):
        campaign_str = f" [{campaign}]" if campaign else ""
        logger(f"\n  [{idx+1}/{len(groups)}] {model_id} | {region} | {period}{campaign_str}")

        # 1. Mosaic
        mosaic_gcs = gcs_full(region_path(model_id, region, period, campaign))
        mosaic_exists = _gcs_exists(mosaic_gcs)
        if mosaic_exists:
            logger(f"    Mosaic already exists.")

        if not mosaic_exists:
            mosaic_path = merge_region_tiles(model_id, region, period, fs=fs, logger=logger, campaign=campaign)
            if not mosaic_path:
                logger(f"    [WARN] No tiles to mosaic for {region}.")
                continue

        # 2. Stats (per tile, RAM-friendly)
        if not stats_row_exists(model_id, region, period, campaign, fs):
            row = compute_region_stats_from_tiles(model_id, region, period, fs=fs, logger=logger, campaign=campaign)
            if row:
                update_consolidated_stats(row, fs=fs, logger=logger, campaign=campaign)
        else:
            logger(f"    Stats already exist.")

        # 3. GEE upload
        if upload_gee:
            if not gee_asset_exists(model_id, region, period, campaign):
                upload_to_gee(model_id, region, period, fs=fs, logger=logger, campaign=campaign)
            else:
                logger(f"    Already in GEE.")

        prog.passo()
        if prog.completed < prog.total:
            logger(f"  {prog.summary()}")

    total = time.time() - _t0
    logger(f"\n  --- M6 publish done in {_fmt_time(total)} ---")

    # Refresh the M6 cache/UI so the freshly published groups appear at once.
    try:
        from M6_ui import invalidate_discovery_cache, _M6_UI_INSTANCE
        invalidate_discovery_cache()
        if _M6_UI_INSTANCE is not None:
            _M6_UI_INSTANCE.refresh()
    except Exception:
        pass
