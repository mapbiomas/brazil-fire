"""Validation of an experimental per-band tile manifest.

No network operations occur here. An exporter must persist its manifest before
making tiles visible, and a consumer must verify the actual object list too.
"""


def completed_tiles(manifest, existing_objects):
    """Return the exact ordered object names, refusing incomplete exports.

    ``manifest`` must contain a nonempty ``tiles`` list, each with unique
    ``id`` and ``object`` fields, and ``status`` equal to ``complete``. Object
    names are full canonical GCS URLs. Extra objects are ignored, so consumers
    never accidentally combine tiles from different runs.
    """
    if not isinstance(manifest, dict) or manifest.get('version') != 1:
        raise ValueError('Unsupported export manifest')
    if manifest.get('status') != 'complete':
        raise ValueError('Export is not complete')
    tiles = manifest.get('tiles')
    if not isinstance(tiles, list) or not tiles:
        raise ValueError('Export has no expected tiles')
    ids, objects, object_set = set(), [], set()
    for tile in tiles:
        if not isinstance(tile, dict):
            raise ValueError('Invalid tile entry')
        tile_id, obj = tile.get('id'), tile.get('object')
        if not isinstance(tile_id, str) or not tile_id or tile_id in ids:
            raise ValueError('Missing or duplicate tile ID')
        if not isinstance(obj, str) or not obj.startswith('gs://') or obj in object_set:
            raise ValueError('Invalid or duplicate tile object')
        ids.add(tile_id)
        objects.append(obj)
        object_set.add(obj)
    missing = set(objects) - set(existing_objects)
    if missing:
        raise ValueError(f'Export is missing {len(missing)} expected tile(s)')
    return objects
