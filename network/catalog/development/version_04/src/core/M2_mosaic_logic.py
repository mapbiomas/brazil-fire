"""M2 download and mosaic logic for country-wide COGs."""
import os
import subprocess
import glob
import tempfile
from functools import lru_cache

from M0_auth_config import (
    CONFIG, mosaic_name, mosaic_name_for_period, _get_fs,
    monthly_chunk_path, monthly_cog_path,
    yearly_chunk_path, yearly_cog_path,
    period_chunk_path, period_cog_path,
    get_temp_dir, check_command_exists
)
from M_periods import parse_period
from M_gcs import upload, download, rm_files

# M2 output is 8-bit for every band except dayOfYear, which needs Int16.
BAND_DATATYPES = {
    'blue': 'Byte', 'green': 'Byte', 'red': 'Byte',
    'nir': 'Byte', 'swir1': 'Byte', 'swir2': 'Byte',
    'nbr': 'Int16', 'dayOfYear': 'Int16',
    'classification': 'Byte', 'burned_area': 'Byte',
    'probability': 'Float32', 'score': 'Float32'
}

# The COG is the final product; chunks are only intermediate files. Deleting
# them after a successful upload keeps the bucket from holding two copies.
DELETE_CHUNKS_AFTER_COG = True


def list_gcs_files(prefix, logger=None):
    """Lista shards no GCS usando gcsfs."""
    fs = _get_fs()
    clean_prefix = prefix.replace(f"gs://{CONFIG['bucket']}/", "").strip("/") + "/"
    
    if logger: logger(f"Searching shards in: gs://{CONFIG['bucket']}/{clean_prefix}", "info")
    
    try:
        files = fs.find(f"gs://{CONFIG['bucket']}/{clean_prefix}")
        tif_files = [
            f for f in files
            if f.endswith('.tif') and '/.m1_updates/' not in f.lower()
        ]
        return [f"gs://{f}" for f in tif_files]
    except Exception as e:
        if logger: logger(f"Error accessing GCS: {e}", "error")
        return []

def check_m2_dependencies():
    """Valida se utilitarios estao integrados."""
    from M0_auth_config import ensure_gdal_path
    ensure_gdal_path()
    
    deps = {
        'gsutil':  'gsutil' if os.name != 'nt' else 'gsutil.cmd',
        'gdalbuildvrt': 'gdalbuildvrt',
        'gdal_translate': 'gdal_translate'
    }
    missing = [name for name, cmd in deps.items() if not check_command_exists(cmd)]
    try:
        import rasterio  # noqa: F401
    except ImportError:
        missing.append('rasterio')
    return missing

def run_cmd(args, label="Comando"):
    """Envolve comando subprocess com traces mais seguros"""
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error in {label}: {e.stderr.strip() if e.stderr else e.stdout.strip()}")


def mask_raster_to_country(source_path, output_path, country_geometry, window_size=512):
    """Rasterize the country mask window-by-window on the unchanged source grid.

    Data and validity masks are read/written in bounded windows so a large
    country COG never needs to be materialized as a full-image NumPy array.
    """
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom
    from rasterio.windows import Window, transform as window_transform

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    with rasterio.open(source_path) as source:
        geometry = country_geometry
        if source.crs and str(source.crs).upper() not in ('EPSG:4326', 'OGC:CRS84'):
            geometry = transform_geom('EPSG:4326', source.crs, country_geometry)
        profile = source.profile.copy()
        profile.update(driver='GTiff', compress='LZW', tiled=True)
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        output_block_size = max(16, ((window_size + 15) // 16) * 16)
        profile['blockxsize'] = output_block_size
        profile['blockysize'] = output_block_size

        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(output_path, 'w', **profile) as destination:
                for row in range(0, source.height, window_size):
                    height = min(window_size, source.height - row)
                    for col in range(0, source.width, window_size):
                        width = min(window_size, source.width - col)
                        window = Window(col, row, width, height)
                        country_mask = geometry_mask(
                            [geometry],
                            out_shape=(height, width),
                            transform=window_transform(window, source.transform),
                            all_touched=False,
                            invert=True,
                        )
                        source_mask = source.dataset_mask(window=window) > 0
                        valid = country_mask & source_mask
                        for band_index in range(1, source.count + 1):
                            data = source.read(band_index, window=window)
                            destination.write(data, band_index, window=window)
                        destination.write_mask(
                            np.where(valid, 255, 0).astype('uint8'),
                            window=window,
                        )
    return output_path


@lru_cache(maxsize=16)
def _country_geometry_geojson(asset_regions):
    """Fetch a configured country boundary once per asset per notebook session."""
    import ee
    return ee.FeatureCollection(asset_regions).geometry().getInfo()


def mask_existing_country_cog(
    period_key, band, sensor, mosaic_method, logger=None, update_manifest=True,
):
    """Mask an existing GCS COG in place without creating an Earth Engine task.

    The old object remains untouched until the masked COG has been built
    successfully; the final upload replaces the same GCS object key.
    """
    import shutil
    from M_gcs import blob_exists

    info = parse_period(period_key)
    cog_dir = period_cog_path(period_key, mosaic=mosaic_method, sensor=sensor)
    band_name = 'dayOfYear' if band.lower() == 'dayofyear' else band
    cog_name = (
        f"{mosaic_name_for_period(period_key, band=band_name, sensor=sensor, mosaic=mosaic_method)}_cog.tif"
    )
    remote_uri = f"gs://{CONFIG['bucket']}/{cog_dir}/{cog_name}"
    if not blob_exists(remote_uri):
        return False

    missing = [dep for dep in check_m2_dependencies() if dep != 'gsutil']
    if missing:
        raise RuntimeError(f"Missing M2 dependencies: {missing}")

    work_dir = tempfile.mkdtemp(prefix='m1_country_mask_', dir=get_temp_dir())
    try:
        source_path = os.path.join(work_dir, cog_name)
        masked_path = os.path.join(work_dir, f"masked_{cog_name}")
        output_path = os.path.join(work_dir, f"updated_{cog_name}")
        download(remote_uri, source_path)
        geometry = _country_geometry_geojson(CONFIG['asset_regions'])
        mask_raster_to_country(source_path, masked_path, geometry)
        data_type = BAND_DATATYPES.get(band_name, 'Int16')
        run_cmd([
            'gdal_translate', '-of', 'COG', '-ot', data_type,
            '-co', 'COMPRESS=LZW', '-co', 'PREDICTOR=2',
            '-co', 'NUM_THREADS=2', '-co', 'BIGTIFF=YES',
            masked_path, output_path,
        ], label=f"COG country-mask update ({band_name})")
        upload(output_path, f"{CONFIG['bucket']}/{cog_dir}/{cog_name}")

        if update_manifest:
            try:
                from M1_update_manifest import complete_gcs_item
                complete_gcs_item(
                    CONFIG['country'], sensor, info.periodicity,
                    mosaic_method, period_key, band_name,
                )
            except Exception as e:
                if logger:
                    logger(f"[WARN] Masked COG uploaded but manifest completion failed: {e}", "warning")
        if logger:
            logger(f"Updated existing COG in place: {remote_uri}", "success")
        return True
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

def assemble_country_mosaic(year, month=None, period='monthly', bands=None, sensor=None, mosaic_method='minnbr', logger=None, progress_idx=0, progress_total=0, start_time=None, period_key=None, update_manifest=True):
    import shutil
    from M0_auth_config import GLOBAL_OPTS, _single

    s = sensor or _single(GLOBAL_OPTS['SENSOR'])
    m_method = mosaic_method or 'minnbr'

    info = parse_period(period_key, periodicity=period) if period_key else None
    if info:
        year = info.year
        month = info.index if info.periodicity == 'monthly' else None
        period = info.periodicity
        chunk_prefix  = period_chunk_path(info.key, mosaic=m_method, sensor=s)
        mosaic_prefix = period_cog_path(info.key, mosaic=m_method, sensor=s)
        base_name = mosaic_name_for_period(info.key, mosaic=m_method, sensor=s)
        label = info.key
    elif period == 'monthly':
        chunk_prefix  = monthly_chunk_path(year, month, mosaic=m_method, sensor=s)
        mosaic_prefix = monthly_cog_path(year, month, mosaic=m_method, sensor=s)
        base_name = mosaic_name(year, month, 'monthly', mosaic=m_method, sensor=s)
        label = f"{year}-{month:02d}"
    else:
        chunk_prefix  = yearly_chunk_path(year, mosaic=m_method, sensor=s)
        mosaic_prefix = yearly_cog_path(year, mosaic=m_method, sensor=s)
        base_name = mosaic_name(year, period='yearly', mosaic=m_method, sensor=s)
        label = f"{year} Annual"

    missing = check_m2_dependencies()
    # Remove gsutil from required deps — now using gcsfs
    missing = [d for d in missing if d != 'gsutil']
    if missing:
        msg = f" Missing vital dependencies: {missing}"
        if logger: logger(msg, "error")
        return []

    work_root = get_temp_dir()
    # Parallel M2 workers can share a period/mosaic but process different bands.
    # Allocate atomically unique work dirs so one worker cannot remove another's
    # temporary VRT, shards, or COG during cleanup.
    tmp_path = tempfile.mkdtemp(prefix=f"m2_{base_name}_", dir=work_root)

    results = []

    try:
        # Busca direta no GCS
        gcs_files = list_gcs_files(chunk_prefix, logger=logger)
        
        if not gcs_files:
            if logger: logger(f"No shard found in GCS for {label}.", "warning")
            return []
        country_geometry = _country_geometry_geojson(CONFIG['asset_regions'])
            
        found_bands = {}
        for f in gcs_files:
            fname = os.path.basename(f)
            # Match mais flexível: busca a banda cercada por separadores
            for b_name in (bands or CONFIG['bands_all']):
                # Procuramos a banda como um "token" isolado no nome do arquivo
                # Ex: _blue_ ou _blue.tif ou _blue-
                if f"_{b_name}_" in fname or f"_{b_name}." in fname:
                    if b_name not in found_bands: found_bands[b_name] = []
                    found_bands[b_name].append(f)
                    break
        
        # Log para conferência
        if logger:
            for b, f_list in found_bands.items():
                logger(f"[DEBUG] Band {b}: {len(f_list)} shards found.", "info")
        
        target_bands = bands or CONFIG['bands_all']
        bands_to_process = {b: found_bands[b] for b in target_bands if b in found_bands}
        
        if not bands_to_process:
            if logger: logger(f"Target bands {target_bands} not detected in GCS for {base_name}.", "warning")
            return []

        current_step = progress_idx

        for b_name, remote_shards in bands_to_process.items():
            clean_b_name = "dayOfYear" if b_name.lower() == "dayofyear" else b_name
            current_step += 1
            
            if logger: logger(f"Processing [{b_name}] ({len(remote_shards)} shards)...", "info")
            
            band_tmp = os.path.join(tmp_path, b_name)
            os.makedirs(band_tmp, exist_ok=True)

            try:
                for shard_url in remote_shards:
                    local_shard = os.path.join(band_tmp, os.path.basename(shard_url))
                    download(shard_url, local_shard)

                local_shards = glob.glob(os.path.join(band_tmp, '*.tif'))
                if not local_shards: continue

                vrt_path = os.path.join(tmp_path, f"{base_name}_{b_name}.vrt")
                run_cmd(['gdalbuildvrt', vrt_path] + local_shards, label=f"VRT ({b_name})")
                masked_path = os.path.join(tmp_path, f"{base_name}_{b_name}_country_mask.tif")
                mask_raster_to_country(vrt_path, masked_path, country_geometry)

                cog_base = (
                    mosaic_name_for_period(info.key, band=clean_b_name, sensor=s, mosaic=m_method)
                    if info else mosaic_name(year, month, period, clean_b_name, sensor=s, mosaic=m_method)
                )
                cog_remote_name = f"{cog_base}_cog.tif"
                cog_local_path = os.path.join(tmp_path, cog_remote_name)
                
                dt = BAND_DATATYPES.get(clean_b_name, 'Int16')
                
                run_cmd([
                    'gdal_translate', '-of', 'COG', '-ot', dt,
                    '-co', 'COMPRESS=LZW', '-co', 'PREDICTOR=2',
                    '-co', 'NUM_THREADS=2', '-co', 'BIGTIFF=YES', masked_path, cog_local_path
                ], label=f"COG Conversion ({clean_b_name})")

                dest = f"{CONFIG['bucket']}/{mosaic_prefix}/{cog_remote_name}"
                upload(cog_local_path, dest)

                if update_manifest:
                    try:
                        from M1_update_manifest import complete_gcs_item
                        complete_gcs_item(
                            CONFIG['country'], s, period, m_method, label, clean_b_name,
                        )
                    except Exception as e:
                        if logger:
                            logger(f"[WARN] COG uploaded but update manifest was not marked complete: {e}", "warning")
                
                if logger: logger(f" Success: {cog_remote_name}", "success")
                if DELETE_CHUNKS_AFTER_COG:
                    try:
                        rm_files(remote_shards)
                        if logger:
                            logger(
                                f" Cleaned {len(remote_shards)} source chunks for {clean_b_name}.",
                                "info",
                            )
                    except Exception as e:
                        if logger:
                            logger(
                                f"[WARN] COG built but source chunks may remain for {clean_b_name}: {e}",
                                "warning",
                            )
                results.append(f"gs://{dest}")
            finally:
                if os.path.exists(band_tmp): shutil.rmtree(band_tmp)

    finally:
        if os.path.exists(tmp_path): shutil.rmtree(tmp_path)

    return results
