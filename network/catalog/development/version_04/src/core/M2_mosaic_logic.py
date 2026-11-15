"""M2 download and mosaic logic for country-wide COGs."""
import os
import subprocess
import glob
import tempfile
import json
import hashlib
import threading
import logging
import base64
import math
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
        'gdal_translate': 'gdal_translate',
        'gdal_rasterize': 'gdal_rasterize',
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


def _polygon_features(geometry):
    """Flatten GEE/GeoJSON collections without burning line or point features."""
    kind = geometry.get('type')
    if kind == 'FeatureCollection':
        for feature in geometry.get('features', []):
            yield from _polygon_features(feature)
    elif kind == 'Feature':
        if geometry.get('geometry'):
            yield from _polygon_features(geometry['geometry'])
    elif kind == 'GeometryCollection':
        for part in geometry.get('geometries', []):
            yield from _polygon_features(part)
    elif kind in ('Polygon', 'MultiPolygon'):
        yield {'type': 'Feature', 'properties': {}, 'geometry': geometry}


_MASK_CACHE_VERSION = 1
_mask_cache_guard = threading.Lock()
_mask_cache_dir = None
_mask_cache_locks = {}
_mask_upload_active = set()
_MASK_UPLOAD_RETRY_DELAYS = (0.25, 1.0)
_MASK_GCS_REQUEST_TIMEOUT = 30


def _mask_gcs_request_timeout():
    """Per-request socket timeout, not a deadline for a complete transfer."""
    import M0_auth_config
    value = (M0_auth_config.CONFIG or {}).get('m2_country_mask_cache_gcs_timeout',
                                             _MASK_GCS_REQUEST_TIMEOUT)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('m2_country_mask_cache_gcs_timeout must be a finite positive number')
    return value


def _mask_file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _remote_mask_blob(key):
    """Resolve the active M0 configuration, including after set_global_opts."""
    import M0_auth_config
    config = M0_auth_config.CONFIG or {}
    prefix = config.get('m2_country_mask_cache_gcs_prefix')
    if 'm2_country_mask_cache_gcs_prefix' not in config and config.get('bucket'):
        prefix = f"gs://{config['bucket']}/temp/registro_de_testes_de_mosaico/m2_country_masks/v1"
    if not prefix:
        return None
    if not isinstance(prefix, str) or not prefix.startswith('gs://'):
        raise ValueError('m2_country_mask_cache_gcs_prefix must be gs://bucket/prefix')
    bucket, separator, name = prefix[5:].strip('/').partition('/')
    if not bucket or not separator or not name.strip('/'):
        raise ValueError('A dedicated GCS cache prefix is required')
    from M_gcs import _storage_client
    return _storage_client().bucket(bucket).blob(name.rstrip('/') + '/' + key + '.tif')


def _verify_uploaded_mask(blob, path):
    """Compare GCS CRC32C for the uploaded generation without deleting it."""
    import google_crc32c
    checksum = google_crc32c.Checksum()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    if blob.generation is None:
        raise ValueError('Uploaded country mask generation unavailable')
    blob.reload(if_generation_match=blob.generation, timeout=_mask_gcs_request_timeout(), retry=None)
    if blob.crc32c != base64.b64encode(checksum.digest()).decode('ascii'):
        raise ValueError('Uploaded country mask CRC32C mismatch')


def _upload_country_mask(blob, path, key):
    """Publish once plus bounded retries of transient errors, never overwrite."""
    from google.api_core.exceptions import (
        ServiceUnavailable, TooManyRequests, GatewayTimeout, DeadlineExceeded,
    )
    from requests.exceptions import ConnectionError as RequestConnectionError, Timeout
    import google_crc32c  # noqa: F401 - require verification before upload

    blob.metadata = {'m2-mask-key': key, 'm2-mask-version': str(_MASK_CACHE_VERSION),
                     'm2-mask-sha256': _mask_file_sha256(path)}
    transient = (OSError, ServiceUnavailable, TooManyRequests, GatewayTimeout,
                 DeadlineExceeded, RequestConnectionError, Timeout)
    for attempt in range(len(_MASK_UPLOAD_RETRY_DELAYS) + 1):
        try:
            # Disable the SDK's retry and automatic checksum deletion: retries
            # are bounded here and every write retains the create-only condition.
            blob.upload_from_filename(path, if_generation_match=0, checksum=None,
                                      content_type='image/tiff', retry=None, timeout=_mask_gcs_request_timeout())
        except transient:
            if attempt == len(_MASK_UPLOAD_RETRY_DELAYS):
                raise
            threading.Event().wait(_MASK_UPLOAD_RETRY_DELAYS[attempt])
        else:
            _verify_uploaded_mask(blob, path)
            return


def _start_mask_upload(blob, path, key):
    """One daemon publisher per local mask; never wait for network in callers."""
    with _mask_cache_guard:
        if path in _mask_upload_active:
            return
        _mask_upload_active.add(path)

    def worker():
        try:
            _upload_country_mask(blob, path, key)
        except Exception as exc:
            logging.getLogger(__name__).warning('M2 mask cache upload skipped or failed: %s', exc)
        finally:
            with _mask_cache_guard:
                _mask_upload_active.discard(path)

    try:
        threading.Thread(target=worker, name='m2-mask-cache-upload', daemon=True).start()
    except Exception:
        with _mask_cache_guard:
            _mask_upload_active.discard(path)
        raise


def _download_country_mask(blob, path, source, key):
    """Download one immutable generation and validate metadata, bytes and grid."""
    import rasterio
    blob.reload(timeout=_mask_gcs_request_timeout(), retry=None)
    metadata = blob.metadata or {}
    if (metadata.get('m2-mask-key') != key or
            metadata.get('m2-mask-version') != str(_MASK_CACHE_VERSION) or
            not metadata.get('m2-mask-sha256') or blob.generation is None):
        raise ValueError('Invalid country mask cache metadata')
    blob.download_to_filename(path, if_generation_match=blob.generation, checksum='auto',
                              timeout=_mask_gcs_request_timeout(), retry=None)
    if _mask_file_sha256(path) != metadata['m2-mask-sha256']:
        raise ValueError('Country mask checksum mismatch')
    with rasterio.open(path) as mask:
        if (mask.width, mask.height, mask.crs, mask.transform, mask.count, mask.dtypes) != (
            source.width, source.height, source.crs, source.transform, 1, ('uint8',)
        ):
            raise ValueError('Country mask grid or type mismatch')
        for _, window in mask.block_windows(1):
            data = mask.read(1, window=window)
            if ((data != 0) & (data != 1)).any():
                raise ValueError('Invalid country mask values')


def _cached_country_mask(source, features, block_size):
    """Build once per complete grid and polygon payload, shared by threads.

    The private temporary directory lives for the Python session and is removed
    at interpreter exit. GCS persistence defaults to the dedicated bucket prefix;
    set m2_country_mask_cache_gcs_prefix to False to disable it. Failures fall
    back to local rasterization. Uploads run in background with bounded transient
    retries; process exit may interrupt them. No eviction is performed.
    """
    global _mask_cache_dir
    import rasterio

    payload = {'version': _MASK_CACHE_VERSION, 'crs': source.crs.to_wkt() if source.crs else None,
               'transform': list(source.transform), 'width': source.width, 'height': source.height,
               'features': features}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True).encode('utf-8')).hexdigest()
    with _mask_cache_guard:
        if _mask_cache_dir is None:
            _mask_cache_dir = tempfile.TemporaryDirectory(prefix='m2_country_cache_', dir=get_temp_dir())
        directory = _mask_cache_dir.name
        lock = _mask_cache_locks.setdefault(key, threading.Lock())
    raster_path = os.path.join(directory, key + '.tif')
    with lock:
        if os.path.isfile(raster_path):
            return raster_path
        # The staging directory is on the same filesystem, so publication is
        # atomic; neither a failed rasterization nor a partial file is reused.
        with tempfile.TemporaryDirectory(prefix='build_', dir=directory) as staging:
            vector_path = os.path.join(staging, 'country.geojson')
            staged_raster = os.path.join(staging, 'country.tif')
            blob = None
            missing = False
            try:
                blob = _remote_mask_blob(key)
                if blob is not None:
                    try:
                        _download_country_mask(blob, staged_raster, source, key)
                    except Exception as exc:
                        from google.api_core.exceptions import NotFound
                        if isinstance(exc, (NotFound, FileNotFoundError)):
                            missing = True
                        else:
                            logging.getLogger(__name__).warning('M2 mask cache download failed: %s', exc)
                    else:
                        os.replace(staged_raster, raster_path)
                        return raster_path
            except Exception as exc:
                logging.getLogger(__name__).warning('M2 mask cache unavailable: %s', exc)
            with open(vector_path, 'w', encoding='utf-8') as vector:
                json.dump({'type': 'FeatureCollection', 'features': features}, vector)
            mask_profile = {
                'driver': 'GTiff', 'width': source.width, 'height': source.height,
                'count': 1, 'dtype': 'uint8', 'crs': source.crs,
                'transform': source.transform, 'tiled': True,
                'blockxsize': block_size, 'blockysize': block_size,
                'compress': 'LZW', 'BIGTIFF': 'YES', 'SPARSE_OK': 'TRUE',
            }
            with rasterio.open(staged_raster, 'w', **mask_profile):
                pass
            run_cmd(['gdal_rasterize', '-burn', '1', '-optim', 'VECTOR',
                     vector_path, staged_raster], label='Country mask rasterization')
            os.replace(staged_raster, raster_path)
            if blob is not None and missing:
                try:
                    _start_mask_upload(blob, raster_path, key)
                except Exception as exc:
                    logging.getLogger(__name__).warning('M2 mask cache upload skipped or failed: %s', exc)
    return raster_path


def mask_raster_to_country(source_path, output_path, country_geometry, window_size=512):
    """Rasterize once on the unchanged source grid, then combine masks by window.

    Data and validity masks are read/written in bounded windows so a large
    country COG never needs to be materialized as a full-image NumPy array.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.windows import Window

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    features = list(_polygon_features(country_geometry))
    if not features:
        raise ValueError('Country geometry has no polygons')

    with rasterio.open(source_path) as source:
        data_bands = [i for i, color in enumerate(source.colorinterp, 1)
                      if color != ColorInterp.alpha]
        if not data_bands:
            raise ValueError('Source has no data bands')
        profile = source.profile.copy()
        profile.update(driver='GTiff', count=len(data_bands), compress='LZW', tiled=True, BIGTIFF='YES')
        profile.pop('photometric', None)
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        output_block_size = max(16, ((window_size + 15) // 16) * 16)
        profile['blockxsize'] = output_block_size
        profile['blockysize'] = output_block_size

        raster_path = _cached_country_mask(source, features, output_block_size)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(raster_path) as country, rasterio.open(output_path, 'w', **profile) as destination:
            for row in range(0, source.height, window_size):
                height = min(window_size, source.height - row)
                for col in range(0, source.width, window_size):
                    width = min(window_size, source.width - col)
                    window = Window(col, row, width, height)
                    country_mask = country.read(1, window=window) > 0
                    source_mask = source.dataset_mask(window=window) > 0
                    valid = country_mask & source_mask
                    for output_index, band_index in enumerate(data_bands, 1):
                        data = source.read(band_index, window=window)
                        destination.write(data, output_index, window=window)
                    destination.write_mask(
                        np.where(valid, 255, 0).astype('uint8'),
                        window=window,
                    )
    return output_path


@lru_cache(maxsize=16)
def _country_geometry_geojson(asset_regions):
    """Fetch individual region features once per asset per notebook session."""
    import ee
    return ee.FeatureCollection(asset_regions).getInfo()


def _validate_masked_cog(source_path, output_path):
    """Reject incomplete or grid-changing conversions before publishing."""
    import rasterio
    from rasterio.enums import ColorInterp, MaskFlags

    with rasterio.open(source_path) as source, rasterio.open(output_path) as output:
        if (source.width, source.height, source.transform, source.crs, source.count) != (
            output.width, output.height, output.transform, output.crs, output.count
        ):
            raise ValueError('COG grid or band count differs from masked source')
        if ColorInterp.alpha in output.colorinterp:
            raise ValueError('COG must not contain an alpha band')
        if not all(MaskFlags.per_dataset in flags for flags in output.mask_flag_enums):
            raise ValueError('COG is missing its internal validity mask')
        if output.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG':
            raise ValueError('Output is not a COG')
        # Force a read, detecting at least an unreadable first tile before upload.
        output.read(1, window=((0, min(1, output.height)), (0, min(1, output.width))))


def _cog_threads(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('gdal_threads must be a positive integer')
    return value


def mask_existing_country_cog(
    period_key, band, sensor, mosaic_method, logger=None, update_manifest=True, gdal_threads=2,
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
            '-co', f'NUM_THREADS={_cog_threads(gdal_threads)}', '-co', 'BIGTIFF=YES',
            masked_path, output_path,
        ], label=f"COG country-mask update ({band_name})")
        _validate_masked_cog(masked_path, output_path)
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

def assemble_country_mosaic(year, month=None, period='monthly', bands=None, sensor=None, mosaic_method='minnbr', logger=None, progress_idx=0, progress_total=0, start_time=None, period_key=None, update_manifest=True, gdal_threads=2):
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
                    '-co', f'NUM_THREADS={_cog_threads(gdal_threads)}', '-co', 'BIGTIFF=YES', masked_path, cog_local_path
                ], label=f"COG Conversion ({clean_b_name})")

                _validate_masked_cog(masked_path, cog_local_path)
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
