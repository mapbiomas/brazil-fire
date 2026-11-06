"""Experimental local computePixels exporter with GCS-backed resumption."""
import concurrent.futures
import hashlib
import json
import os
import tempfile

import ee

from M_compute_pixels import compute_pixels, make_tile_request
from M_export_grid import tiles_for_bounds
from M_export_manifest import completed_tiles


def export_band(image, geometry, band, band_name, folder, config, scale,
                workers=4, tile_pixels=1024, logger=print):
    """Export tiles for one band; reuse existing objects after interruption.

    A manifest is written before any tile. An incompatible existing manifest
    is never replaced. COG consolidation is left to M2.
    """
    from M_gcs import blob_exists, read_json, write_json, upload, _storage_client
    if workers < 1 or tile_pixels < 1:
        raise ValueError('workers and tile_pixels must be positive')
    bucket = config['bucket']
    manifest_path = f'{bucket}/{folder}/{band_name}.manifest.json'
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
               'crs': 'EPSG:4326', 'resolution': resolution, 'tiles': objects}
    if blob_exists(manifest_path):
        current = read_json(manifest_path)
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
    logger(f'[GCS] {band_name}: {len(existing)}/{len(tiles)} tiles available')
    try:
        credentials = ee.data._get_state().credentials
    except AttributeError as exc:
        raise RuntimeError('Cannot access the initialized Earth Engine user credentials') from exc
    if credentials is None:
        raise RuntimeError('Earth Engine user credentials are not initialized')

    def fetch(item):
        tile, dest = item
        request = make_tile_request(expression, tile, resolution, tile_pixels,
                                    'EPSG:4326', [2 if band == 'dayOfYear' else 1])
        payload = compute_pixels(config['gee_project'], credentials, request)
        with tempfile.TemporaryDirectory(prefix='m1_tile_') as tmp:
            local = os.path.join(tmp, 'tile.tif')
            with open(local, 'wb') as handle:
                handle.write(payload)
            upload(local, dest)
        return dest

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, item): item for item in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                existing.add(future.result())
            except Exception as exc:
                failures.append((futures[future][0].id, str(exc)))
            if index % 25 == 0 or index == len(pending):
                logger(f'[GCS] {band_name}: {len(existing)}/{len(tiles)} tiles available')
    if failures:
        raise RuntimeError(f'{len(failures)} tiles failed; rerun M1 to resume: {failures[0]}')
    planned['status'] = 'complete'
    completed_tiles(planned, existing)
    write_json(manifest_path, planned)
    return planned
