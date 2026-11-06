"""Experimental local computePixels exporter with GCS-backed resumption."""
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import threading

import ee

from M_compute_pixels import compute_pixels, make_tile_request, download_adaptive_tile
from M_export_grid import tiles_for_bounds
from M_export_manifest import completed_tiles
from M_lang import L
from M_progress_eta import progress_message


def export_band(image, geometry, band, band_name, folder, config, scale,
                workers=4, tile_pixels=4096, logger=print):
    """Export tiles for one band; reuse existing objects after interruption.

    A manifest is written before any tile. An incompatible existing manifest
    is never replaced. COG consolidation is left to M2.
    """
    from M_gcs import blob_exists, read_json, write_json, upload, _storage_client
    if workers < 1 or tile_pixels < 1:
        raise ValueError('workers and tile_pixels must be positive')
    bucket = config['bucket']
    manifest_path = f'{bucket}/{folder}/{band_name}.manifest.json'
    current = read_json(manifest_path) if blob_exists(manifest_path) else None
    if current is not None:
        # The first experimental release used 1024 pixels without recording
        # the size. Keep that grid to resume it without deleting any tiles.
        tile_pixels = current.get('tile_pixels', 1024)
    # EPSG:4326 matches the geographic output convention; EE nominal metre
    # scale is expressed in degrees at the equator. This is versioned in the
    # manifest and must be compared with existing products before production.
    resolution = scale / 111319.49079327358
    coords = geometry.bounds().coordinates().getInfo()[0]
    xs, ys = [p[0] for p in coords], [p[1] for p in coords]
    tiles = list(tiles_for_bounds((min(xs), min(ys), max(xs), max(ys)),
                                  resolution, tile_pixels))
    expression = ee.serializer.encode(image.select(band), for_cloud_api=True)
    spec = {'expression': expression, 'band': band, 'scale': scale,
            'crs': 'EPSG:4326', 'resolution': resolution, 'tile_pixels': tile_pixels,
            'project': config['gee_project']}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    objects = [{'id': t.id, 'object': f'gs://{bucket}/{folder}/{band_name}_{t.id}.tif'} for t in tiles]
    planned = {'version': 1, 'status': 'running', 'fingerprint': fingerprint,
               'crs': 'EPSG:4326', 'resolution': resolution,
               'tile_pixels': tile_pixels, 'tiles': objects}
    if current is not None:
        if current.get('fingerprint') != fingerprint or current.get('tiles') != objects:
            raise RuntimeError('Existing manifest has different parameters; refusing to overwrite it')
        if current.get('status') == 'consumed':
            if current.get('cog') and blob_exists(current['cog']):
                logger(f'[GCS] COG already exists: {current["cog"]}')
                return current
            raise RuntimeError('Manifest is consumed but its COG is missing; manual recovery required')
        if current.get('status') not in ('running', 'complete'):
            raise RuntimeError('Unsupported manifest state')
    else:
        cog_folder = folder.rsplit('/', 1)[0] + '/COG'
        if blob_exists(f'{bucket}/{cog_folder}/{band_name}_cog.tif'):
            raise RuntimeError('Existing COG will not be overwritten by the experimental exporter')
        write_json(manifest_path, planned)

    prefix = f'{folder}/{band_name}_'
    blobs = _storage_client().list_blobs(bucket, prefix=prefix)
    existing_paths = {f'gs://{bucket}/{blob.name}' for blob in blobs}
    existing = {entry['object'] for entry in objects if entry['object'] in existing_paths}
    pending = [(tile, entry['object']) for tile, entry in zip(tiles, objects)
               if entry['object'] not in existing]
    start = time.monotonic()
    last_report = start
    logger(f'{band_name} ' + progress_message(L.ETA_M1_PHASE, len(existing), len(tiles), 0, 0, len(pending)))
    try:
        credentials = ee.data._get_state().credentials
    except AttributeError as exc:
        raise RuntimeError('Cannot access the initialized Earth Engine user credentials') from exc
    if credentials is None:
        raise RuntimeError('Earth Engine user credentials are not initialized')

    request_limit = {'pixels': tile_pixels}
    limit_lock = threading.Lock()

    def fetch(item):
        tile, dest = item
        request = make_tile_request(expression, tile, resolution, tile_pixels,
                                    'EPSG:4326', [2 if band == 'dayOfYear' else 1])
        with tempfile.TemporaryDirectory(prefix='m1_tile_') as tmp:
            local = os.path.join(tmp, 'tile.tif')
            with limit_lock:
                max_pixels = request_limit['pixels']

            def on_split(_old, size):
                with limit_lock:
                    changed = size < request_limit['pixels']
                    request_limit['pixels'] = min(size, request_limit['pixels'])
                if changed:
                    logger(L.ETA_TILE_SPLIT.format(tile=tile.id, size=size))

            download_adaptive_tile(
                config['gee_project'], credentials, request, local, fetch=compute_pixels,
                dtype='int16' if band == 'dayOfYear' else 'uint8',
                on_split=on_split, request_max_pixels=max_pixels,
            )
            upload(local, dest)
        return dest

    failures = []
    max_failures = 3
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    iterator = iter(pending)
    futures = {}
    try:
        def submit_next():
            item = next(iterator, None)
            if item is not None:
                futures[pool.submit(fetch, item)] = item
        for _ in range(min(len(pending), workers)):
            submit_next()
        index = 0
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                try:
                    existing.add(future.result())
                except Exception as exc:
                    failures.append((item[0].id, str(exc)))
                    logger(L.ETA_TILE_FAILED.format(tile=item[0].id, error=str(exc)))
                    if len(failures) == max_failures:
                        logger(L.ETA_EXPORT_PAUSED.format(count=max_failures))
                index += 1
                now = time.monotonic()
                if index == 1 or index % 10 == 0 or index == len(pending) or now - last_report >= 30:
                    logger(f'{band_name} ' + progress_message(
                        L.ETA_M1_PHASE, len(existing), len(tiles), now - start,
                        index - len(failures), len(pending), len(failures)))
                    last_report = now
                if len(failures) < max_failures:
                    submit_next()
    except BaseException:
        # Do not enqueue the rest of the country after a Colab interruption.
        # In-flight requests may still finish and can be reused on resume.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    if failures:
        raise RuntimeError(f'{len(failures)} tiles failed; rerun M1 to resume: {failures[0]}')
    planned['status'] = 'complete'
    completed_tiles(planned, existing)
    write_json(manifest_path, planned)
    return planned
