"""Regional asset export primitives for the experimental M1.

The national asset path is unchanged and the image is not clipped. Existing
assets and active tasks are not overwritten or submitted again.
"""
import hashlib
import re
import unicodedata

import ee


def region_asset_suffix(region_name):
    """Stable, EE-safe suffix, including a hash to avoid slug collisions."""
    if not isinstance(region_name, str) or not region_name.strip():
        raise ValueError('Region names must be non-empty strings')
    name = region_name.strip()
    ascii_name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^a-z0-9]+', '_', ascii_name.lower()).strip('_')[:40] or 'region'
    digest = hashlib.sha256(name.encode('utf-8')).hexdigest()[:12]
    return f'region_{slug}_{digest}'


def regional_task_prefix(task_flag, base_name):
    """Return an EE-safe, deterministic prefix with room for a region hash."""
    raw = f'{task_flag}_ASSET_{base_name}'
    ascii_text = unicodedata.normalize('NFKD', raw).encode('ascii', 'ignore').decode('ascii')
    safe = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', ascii_text)
    if len(safe) > 86:
        digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
        safe = safe[:77] + '_' + digest
    return safe


def validate_region_names(values):
    """Check every feature has a unique identifier before launching tasks."""
    if not isinstance(values, list) or not values:
        raise ValueError('Region collection must contain at least one feature')
    suffixes = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError('Region identifiers must be non-empty strings without surrounding spaces')
        if value in seen:
            raise ValueError(f'Duplicate region identifier: {value!r}')
        seen.add(value)
        suffixes.append(region_asset_suffix(value))
    if len(set(suffixes)) != len(suffixes):
        raise ValueError('Regional asset suffix collision')
    return suffixes


def list_collection_asset_names(collection_id):
    """List asset IDs, without masking permission or network errors."""
    names, token = set(), None
    while True:
        params = {'parent': collection_id}
        if token:
            params['pageToken'] = token
        result = ee.data.listAssets(params)
        names.update(item['name'] for item in result.get('assets', []))
        token = result.get('nextPageToken')
        if not token:
            return names


def submit_region_exports(image, collection, collection_id, base_name, band,
                          period_key, properties, scale, task_flag,
                          region_property='region_nam'):
    """Submit one asset task per region; return started and skipped IDs.

    The caller is responsible for creating the image collection and applying
    the intended country mask. All identifiers and EE listings are validated
    before the first task starts. A partial submission error is propagated;
    the next invocation can skip assets and tasks that already exist.
    """
    if not all(isinstance(v, str) and v for v in
               (collection_id, base_name, band, period_key, task_flag, region_property)):
        raise ValueError('Missing regional export configuration')
    if scale <= 0:
        raise ValueError('Scale must be positive')
    # aggregate_array omits null values; compare with feature count so a
    # missing identifier cannot silently drop a region.
    size = collection.size().getInfo()
    names = collection.aggregate_array(region_property).getInfo()
    if not isinstance(names, list) or len(names) != size:
        raise ValueError(f'Every region must have {region_property!r}')
    suffixes = validate_region_names(names)
    existing = list_collection_asset_names(collection_id)
    active = {task.get('description') for task in ee.data.getTaskList()
              if task.get('state') in ('READY', 'PENDING', 'RUNNING')}
    started, skipped = [], []
    for region_name, suffix in zip(names, suffixes):
        name = f'{base_name}_{suffix}'
        asset_id = f'{collection_id}/{name}'
        description = f'{regional_task_prefix(task_flag, base_name)}_{suffix[-12:]}'
        legacy_description = f'{task_flag}_ASSET_{name}'
        if asset_id in existing or description in active or legacy_description in active:
            skipped.append(asset_id)
            continue
        region = collection.filter(ee.Filter.eq(region_property, region_name)).geometry()
        metadata = dict(properties)
        metadata.update({'period_key': period_key, 'region_id': region_name,
                         'regional_export': 1, 'band': band,
                         'expected_region_count': len(names)})
        tile_image = image.select(band).set(metadata)
        task = ee.batch.Export.image.toAsset(
            image=tile_image, description=description, assetId=asset_id,
            region=region, scale=scale, maxPixels=1e13,
            pyramidingPolicy={'.default': 'median'},
        )
        task.start()
        started.append(asset_id)
    return {'started': started, 'skipped': skipped}
