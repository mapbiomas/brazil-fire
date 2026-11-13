"""Temporary, country-wide M1 export correction manifest.

The manifest is stored under the configured library's cache folder so update
progress survives notebook restarts without adding campaign to M1 identity.
"""

MANIFEST_VERSION = 1
COUNTRY_MASK_VERSION = 'country_regions_v1'
IDENTITY_FIELDS = (
    'country', 'sensor', 'period', 'mosaic', 'period_key', 'band', 'type',
)


def item_key(item):
    """Return the existing M1 identity: date/period + band + target context."""
    return tuple(str(item.get(field, '')).lower() for field in IDENTITY_FIELDS)


def new_manifest():
    return {
        'version': MANIFEST_VERSION,
        'mask_version': COUNTRY_MASK_VERSION,
        'items': [],
    }


def pending_item(manifest, item):
    key = item_key(item)
    return any(
        item_key(entry) == key
        and entry.get('status', 'pending') not in ('complete', 'submitted_new')
        for entry in manifest.get('items', [])
    )


def find_item(manifest, item):
    key = item_key(item)
    return next(
        (entry for entry in manifest.get('items', []) if item_key(entry) == key),
        None,
    )


def add_pending_item(manifest, item):
    """Record an existing output as needing the country-mask correction."""
    existing = find_item(manifest, item)
    if existing:
        return existing, False
    entry = {
        **{field: item.get(field) for field in IDENTITY_FIELDS},
        'before': 'existing export without country mask guarantee',
        'after': COUNTRY_MASK_VERSION,
        'status': 'pending',
    }
    manifest.setdefault('items', []).append(entry)
    return entry, True


def set_item_status(manifest, item, status, **metadata):
    entry = find_item(manifest, item)
    if entry is None:
        entry, _ = add_pending_item(manifest, item)
    entry.update(metadata)
    entry['status'] = status
    entry.pop('last_error', None)
    return entry


def mark_item_complete(manifest, item):
    return set_item_status(manifest, item, 'complete')


def manifest_path():
    from M0_auth_config import CONFIG
    return f"{CONFIG['gcs_library_images']}/.CACHE/M1_UPDATE_MANIFEST.json"


def load_manifest():
    from M_gcs import exists, read_json
    path = manifest_path()
    if not exists(path):
        return new_manifest()
    manifest = read_json(path)
    if manifest.get('version') != MANIFEST_VERSION:
        raise ValueError(f"Unsupported M1 update manifest version: {manifest.get('version')}")
    manifest.setdefault('items', [])
    manifest.setdefault('mask_version', COUNTRY_MASK_VERSION)
    return manifest


def save_manifest(manifest):
    from M_gcs import write_json
    write_json(manifest_path(), manifest)


def complete_gcs_item(country, sensor, period, mosaic, period_key, band):
    manifest = load_manifest()
    item = {
        'country': country, 'sensor': sensor, 'period': period,
        'mosaic': mosaic, 'period_key': period_key, 'band': band,
        'type': 'GCS',
    }
    entry = find_item(manifest, item)
    if entry and entry.get('status') == 'chunks_ready':
        mark_item_complete(manifest, item)
        save_manifest(manifest)
