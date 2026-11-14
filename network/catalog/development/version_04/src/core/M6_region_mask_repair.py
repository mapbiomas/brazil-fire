"""Resumable repair of already-published regional classification assets.

The first phase updates existing GEE images in place. GCS outputs are
intentionally a separate follow-up phase.
"""

import datetime
import hashlib
import re

from M_periods import parse_period
from M_regions import REGION_NAME_PROPERTY

REPAIR_VERSION = 'source_region_v1'
MANIFEST_VERSION = 1
MANIFEST_FILENAME = 'CLASSIFICATION_REGION_MASK_REPAIR.json'
REGIONAL_FOLDER = 'REGIONAL'


def _campaign_root(config, campaign):
    return (
        f"{config['gee_campaigns_prefix']}/{campaign}/"
        f"LIBRARY_CLASSIFICATIONS/{REGIONAL_FOLDER}"
    )


def _region_lookup(region_names):
    return {
        _normalize_region_token(name): str(name).strip()
        for name in region_names
    }


def _normalize_region_token(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().casefold()).strip('_')


def resolve_classification_identity(asset_id, properties, region_names, campaign):
    """Resolve model, source region, and period; return None if ambiguous/invalid."""
    region_map = _region_lookup(region_names)
    props = properties or {}
    item_campaign = props.get('campaign')
    if item_campaign and str(item_campaign).casefold() != campaign.casefold():
        return None

    model_id = props.get('model_id')
    region = props.get('region')
    period_key = props.get('period')
    suffix = asset_id.rsplit('/', 1)[-1]
    regional_root = f"/LIBRARY_CLASSIFICATIONS/{REGIONAL_FOLDER}/"
    try:
        relative = asset_id.split(regional_root, 1)[1]
        model_from_path = relative.split('/', 1)[0]
    except IndexError:
        return None
    model_id = str(model_id or model_from_path)

    normalized_region = _normalize_region_token(region) if region else None
    if normalized_region in region_map and period_key:
        region = region_map[normalized_region]
    else:
        # Legacy assets may not carry region/period properties. Match against
        # actual region names, then accept only one valid period parse.
        matches = []
        for candidate in region_map.values():
            prefix = f"{_normalize_region_token(candidate)}_"
            normalized_suffix = _normalize_region_token(suffix)
            if not normalized_suffix.startswith(prefix):
                continue
            possible_period = normalized_suffix[len(prefix):]
            try:
                parsed = parse_period(possible_period)
            except (TypeError, ValueError):
                continue
            matches.append((candidate, parsed.key))
        if len(matches) != 1:
            return None
        region, parsed_key = matches[0]
        period_key = period_key or parsed_key

    try:
        period_key = parse_period(period_key).key
    except (TypeError, ValueError):
        return None

    return {
        'asset_id': asset_id,
        'campaign': campaign,
        'model_id': model_id,
        'region': region,
        'period': period_key,
    }


def _list_asset_children(ee_api, parent):
    """List all pages of children for an Earth Engine asset parent."""
    token = None
    while True:
        request = {'parent': parent, 'pageSize': 1000}
        if token:
            request['pageToken'] = token
        response = ee_api.data.listAssets(request) or {}
        yield from response.get('assets', [])
        token = response.get('nextPageToken')
        if not token:
            break


def discover_gee_classifications(campaign, config=None, ee_api=None):
    """Inventory regional image assets under one configured catalog campaign.

    Returns ``{'items': [...], 'skipped': [...]}``. This function is read-only;
    it does not create export tasks or modify assets.
    """
    if config is None:
        import M0_auth_config
        config = M0_auth_config.CONFIG
    if not config:
        raise RuntimeError('M0 CONFIG must be initialized before inventory.')
    if ee_api is None:
        import ee as ee_api

    from M_regions import REGION_NAME_PROPERTY

    root = _campaign_root(config, campaign)
    regions_fc = ee_api.FeatureCollection(config['asset_regions'])
    region_names = (
        regions_fc.aggregate_array(REGION_NAME_PROPERTY).distinct().getInfo() or []
    )
    if not region_names:
        raise RuntimeError(
            f"No region names found in {config['asset_regions']} "
            f"(property {REGION_NAME_PROPERTY!r})."
        )

    items, skipped = [], []
    pending = [root]
    while pending:
        parent = pending.pop()
        for asset in _list_asset_children(ee_api, parent):
            asset_id = asset.get('name')
            asset_type = asset.get('type')
            if not asset_id:
                continue
            if asset_type in ('FOLDER', 'IMAGE_COLLECTION'):
                pending.append(asset_id)
                continue
            if asset_type != 'IMAGE':
                continue

            try:
                details = ee_api.data.getAsset(asset_id) or asset
            except Exception as error:
                skipped.append({'asset_id': asset_id, 'reason': f'getAsset failed: {error}'})
                continue
            identity = resolve_classification_identity(
                asset_id, details.get('properties'), region_names, campaign,
            )
            if identity is None:
                skipped.append({
                    'asset_id': asset_id,
                    'reason': 'could not unambiguously resolve campaign, source region, and period',
                })
                continue
            identity['already_masked'] = (
                (details.get('properties') or {}).get('region_mask_version') == REPAIR_VERSION
            )
            items.append(identity)

    return {
        'items': sorted(items, key=lambda item: item['asset_id']),
        'skipped': sorted(skipped, key=lambda item: item['asset_id']),
    }


def build_region_masked_image(ee_api, source_image, region_features, item):
    """Mask to all features of one source region and carry image properties."""
    region_mask = (
        ee_api.Image(0).byte().paint(region_features, 1).gt(0)
    )
    masked = source_image.updateMask(region_mask).copyProperties(source_image)

    # copyProperties without an explicit list copies ordinary properties.
    # Preserve the time properties explicitly because they are system fields.
    try:
        available = source_image.propertyNames().getInfo()
    except Exception:
        available = []
    system_time_properties = [
        prop for prop in ('system:time_start', 'system:time_end')
        if prop in available
    ]
    if system_time_properties:
        masked = masked.copyProperties(source_image, system_time_properties)

    return masked.set({
        'region_mask_version': REPAIR_VERSION,
        'region_masked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'region': item['region'],
        'period': item['period'],
        'model_id': item['model_id'],
        'campaign': item['campaign'],
    })


def task_description(item):
    """Create a stable, short task name for restart/reconciliation."""
    readable = '_'.join((item['model_id'], item['region'], item['period']))
    readable = re.sub(r'[^A-Za-z0-9_-]+', '_', readable).strip('_')[:70]
    digest = hashlib.sha1(item['asset_id'].encode('utf-8')).hexdigest()[:10]
    return f"region_mask_{readable}_{digest}"


def build_overwrite_task(item, config=None, ee_api=None):
    """Build (but do not start) a masked export to the existing GEE asset ID."""
    if config is None:
        import M0_auth_config
        config = M0_auth_config.CONFIG
    if ee_api is None:
        import ee as ee_api

    regions = ee_api.FeatureCollection(config['asset_regions']).filter(
        ee_api.Filter.eq(REGION_NAME_PROPERTY, item['region'])
    )
    if int(regions.size().getInfo()) < 1:
        raise ValueError(
            f"No features in {config['asset_regions']} for region {item['region']!r}."
        )
    geometry = regions.geometry()
    source = ee_api.Image(item['asset_id'])
    masked = build_region_masked_image(ee_api, source, regions, item)
    projection = source.select(0).projection().getInfo()
    if not projection.get('crs') or not projection.get('transform'):
        raise ValueError(f"Source asset has no usable projection: {item['asset_id']}")

    return ee_api.batch.Export.image.toAsset(
        image=masked,
        description=task_description(item),
        assetId=item['asset_id'],
        region=geometry.bounds(1),
        crs=projection['crs'],
        crsTransform=projection['transform'],
        maxPixels=1e13,
        pyramidingPolicy={'.default': 'mode'},
        overwrite=True,
    )


def manifest_path(campaign, config=None):
    if config is None:
        import M0_auth_config
        config = M0_auth_config.CONFIG
    return (
        f"gs://{config['bucket']}/{config['gcs_campaigns_prefix']}/"
        f"{campaign}/.CACHE/{MANIFEST_FILENAME}"
    )


def load_manifest(campaign, config=None):
    from M_gcs import exists, read_json

    path = manifest_path(campaign, config)
    if not exists(path):
        return {'version': MANIFEST_VERSION, 'repair_version': REPAIR_VERSION, 'items': {}}
    manifest = read_json(path)
    if manifest.get('version') != MANIFEST_VERSION:
        raise ValueError(f"Unsupported classification repair manifest: {manifest.get('version')}")
    manifest.setdefault('items', {})
    return manifest


def save_manifest(campaign, manifest, config=None):
    from M_gcs import write_json

    write_json(manifest_path(campaign, config), manifest)


def reconcile_manifest(manifest, task_list):
    """Update submitted item states from the current Earth Engine task list."""
    tasks_by_id = {str(task.get('id')): task for task in task_list if task.get('id')}
    for entry in manifest.get('items', {}).values():
        task = tasks_by_id.get(str(entry.get('task_id')))
        if not task:
            continue
        state = task.get('state')
        if state == 'COMPLETED':
            entry['status'] = 'complete'
        elif state in ('FAILED', 'CANCELLED', 'CANCEL_REQUESTED'):
            entry['status'] = 'failed'
            entry['last_error'] = task.get('error_message') or state
        elif state in ('READY', 'RUNNING', 'PENDING'):
            entry['status'] = 'submitted'
    return manifest


def submit_gee_repairs(items, campaign, confirmation=None, config=None, ee_api=None,
                       max_submissions=2, max_active_tasks=2,
                       retry_failed=False, logger=print):
    """Submit selected repair exports; require an explicit campaign confirmation.

    The inventory and task builder are safe/read-only. This is the only helper
    that starts Earth Engine writes.
    """
    if confirmation != f'OVERWRITE {campaign}':
        raise PermissionError(
            f"To start overwriting assets, pass confirmation='OVERWRITE {campaign}'."
        )
    if config is None:
        import M0_auth_config
        config = M0_auth_config.CONFIG
    if ee_api is None:
        import ee as ee_api

    manifest = load_manifest(campaign, config)
    task_list = ee_api.data.getTaskList()
    reconcile_manifest(manifest, task_list)
    save_manifest(campaign, manifest, config)
    active_states = ('READY', 'RUNNING', 'PENDING')
    active_count = sum(task.get('state') in active_states for task in task_list)
    tasks_by_description = {
        task.get('description'): task for task in task_list
        if task.get('description') and task.get('state') in ('READY', 'RUNNING', 'PENDING')
    }
    submitted = 0
    for item in items:
        asset_id = item['asset_id']
        entry = manifest['items'].setdefault(asset_id, {**item, 'status': 'pending'})
        if item.get('already_masked'):
            entry.update(status='complete', already_masked=True)
            save_manifest(campaign, manifest, config)
            continue
        description = task_description(item)
        active_task = tasks_by_description.get(description)
        if active_task:
            entry.update(status='submitted', task_id=active_task.get('id'))
            save_manifest(campaign, manifest, config)
            continue
        if entry.get('status') in ('complete', 'submitted'):
            continue
        if entry.get('status') == 'failed' and not retry_failed:
            logger(f"[SKIP] Failed asset needs retry_failed=True: {asset_id}")
            continue
        if max_active_tasks is not None and active_count >= max_active_tasks:
            logger(f"[PAUSE] Active GEE task limit reached ({active_count}/{max_active_tasks}).")
            break
        if max_submissions is not None and submitted >= max_submissions:
            break

        entry.update(status='starting', last_error=None)
        save_manifest(campaign, manifest, config)
        try:
            task = build_overwrite_task(item, config=config, ee_api=ee_api)
            task.start()
            entry.update(status='submitted', task_id=getattr(task, 'id', None))
            submitted += 1
            active_count += 1
            logger(f"[SUBMITTED] {asset_id} | task={getattr(task, 'id', None)}")
        except Exception as error:
            entry.update(status='failed', last_error=str(error))
            logger(f"[FAILED] {asset_id}: {error}")
        save_manifest(campaign, manifest, config)
    return manifest
