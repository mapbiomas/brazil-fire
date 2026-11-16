"""Read-only GCS legacy inventory; output is an audit, not an M1 manifest."""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from M_periods import parse_period

FIELDS = ('country', 'sensor', 'period', 'mosaic', 'period_key', 'band', 'type')


def parse_timestamp(value):
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, dt.datetime) or value.utcoffset() is None:
        raise ValueError('timestamp must include a UTC offset')
    return value.astimezone(dt.timezone.utc)


def timestamp(value):
    return parse_timestamp(value).isoformat().replace('+00:00', 'Z')


def key(item):
    return tuple(str(item.get(field, '')).lower() for field in FIELDS)


def identity(name, country, sensor, method):
    parts = name.split('/')
    if len(parts) < 7:
        raise ValueError('unrecognized library path')
    sensor_dir, folder, method_dir, period_key, kind, filename = parts[-6:]
    if (len(parts) < 7 or parts[-7] != 'LIBRARY_IMAGES'
            or sensor_dir.lower() != sensor.lower()
            or method_dir.lower() != method.lower()):
        raise ValueError('path does not match explicit context')
    info = parse_period(period_key)
    if info.folder != folder or info.key != period_key:
        raise ValueError('period folder/key mismatch')
    prefix = f'image_{country}_fire_{sensor}_{method}_'
    if not filename.lower().startswith(prefix.lower()):
        raise ValueError('filename does not match explicit context')
    tail = filename[len(prefix):]
    if kind == 'CHUNKS':
        match = re.fullmatch(r'(.+)_' + re.escape(period_key) + r'_(\d{10}-\d{10})\.tif', tail, re.I)
    elif kind == 'COG':
        match = re.fullmatch(r'(.+)_' + re.escape(period_key) + r'_cog\.tif', tail, re.I)
    else:
        raise ValueError('not a final CHUNKS or COG path')
    if not match or not re.fullmatch(r'[A-Za-z0-9]+', match.group(1)):
        raise ValueError('unrecognized filename')
    return dict(country=country, sensor=sensor, period=info.periodicity, mosaic=method,
                period_key=period_key, band=match.group(1), type='GCS'), kind


def inventory(client, bucket, prefixes, country, campaigns, sensor, method, cutoff, manifest=None):
    cutoff = parse_timestamp(cutoff)
    if not all(isinstance(x, str) and x.strip() for x in [bucket, country, sensor, method]):
        raise ValueError('bucket and context must be nonempty')
    if not prefixes or any(not isinstance(p, str) or not p.strip() or p.startswith('/') or p.startswith('gs://') for p in prefixes):
        raise ValueError('prefixes must be nonempty bucket-relative object prefixes')
    if not campaigns or any(not isinstance(c, str) or not c.strip() for c in campaigns):
        raise ValueError('campaigns must be nonempty')
    if manifest is not None and (not isinstance(manifest, dict) or manifest.get('version') != 1
                                 or not isinstance(manifest.get('items'), list)
                                 or any(not isinstance(i, dict) for i in manifest['items'])):
        raise ValueError('invalid M1 manifest')
    result = dict(bucket=bucket, prefixes=list(dict.fromkeys(prefixes)),
                  campaign_context=list(dict.fromkeys(campaigns)),
                  cutoff=timestamp(cutoff), scan_complete=True, ready=False,
                  candidates=[], observed_groups=[], unresolved=[], conflicts=[], errors=[])
    groups, seen, names = {}, {}, {}
    manifest_index = {}
    for entry in (manifest or {}).get('items', []):
        manifest_index.setdefault(key(entry), []).append(entry)
    for prefix in result['prefixes']:
        try:
            # Default GCS listing returns current live objects only, not historical versions.
            for obj in client.list_blobs(bucket, prefix=prefix):
                name = getattr(obj, 'name', None)
                generation = getattr(obj, 'generation', None)
                created = getattr(obj, 'time_created', None)
                updated = getattr(obj, 'updated', None)
                metageneration = getattr(obj, 'metageneration', None)
                record = dict(name=name, generation=str(generation) if generation is not None else None,
                              time_created=str(created) if created is not None else None,
                              updated=str(updated) if updated is not None else None,
                              metageneration=str(metageneration) if metageneration is not None else None,
                              current=True)
                try:
                    if not isinstance(name, str) or not name.startswith(prefix) or generation is None:
                        raise ValueError('missing or invalid object name/generation')
                    meta, kind = identity(name, country, sensor, method)
                except (ValueError, TypeError, OverflowError) as exc:
                    result['unresolved'].append(dict(**record, reason=str(exc),
                                                     warning='legacy_or_unrecognized_name'))
                    continue
                try:
                    created_utc = parse_timestamp(created)
                    updated_utc = parse_timestamp(updated)
                    if (not re.fullmatch(r'[1-9]\d*', str(generation))
                            or not re.fullmatch(r'[1-9]\d*', str(metageneration))):
                        raise ValueError('invalid generation or metageneration')
                    if updated_utc < created_utc:
                        raise ValueError('updated precedes creation')
                    record['time_created'] = timestamp(created_utc)
                    record['updated'] = timestamp(updated_utc)
                except (ValueError, TypeError, OverflowError) as exc:
                    result['unresolved'].append(dict(**record, identity=meta, reason=str(exc),
                                                     warning='invalid_or_missing_metadata'))
                    continue
                objkey = (name, str(generation))
                if objkey in seen:
                    if seen[objkey] != record:
                        result['conflicts'].append(dict(name=name, reason='duplicate generation with differing metadata',
                                                        first=seen[objkey], duplicate=record))
                    continue
                seen[objkey] = record
                if name in names and names[name] != str(generation):
                    result['conflicts'].append(dict(name=name, reason='multiple current generations observed'))
                names[name] = str(generation)
                if created_utc > cutoff:
                    result['conflicts'].append(dict(name=name, generation=str(generation),
                                                    reason='current object created after cutoff'))
                if updated_utc > cutoff:
                    result['conflicts'].append(dict(name=name, generation=str(generation),
                                                    reason='object metadata updated after cutoff'))
                if int(metageneration) > 1:
                    result['conflicts'].append(dict(name=name, generation=str(generation),
                                                    reason='object metadata has changed'))
                k = key(meta)
                if k not in groups:
                    matches = manifest_index.get(k, [])
                    if len(matches) > 1:
                        result['conflicts'].append(dict(identity=meta, reason='duplicate manifest identity'))
                    groups[k] = dict(**meta,
                                     manifest_status=matches[0].get('status') if len(matches) == 1 else None,
                                     chunks=[], cogs=[])
                groups[k]['chunks' if kind == 'CHUNKS' else 'cogs'].append(record)
        except Exception as exc:
            result['scan_complete'] = False
            result['errors'].append(dict(prefix=prefix, error=f'{type(exc).__name__}: {exc}'))
    for group in groups.values():
        if len(group['cogs']) > 1:
            result['conflicts'].append(dict(
                identity={field: group[field] for field in FIELDS},
                reason='multiple COGs for identity',
                objects=[record['name'] for record in group['cogs']]))
        tiles = {}
        for record in group['chunks']:
            tile = re.search(r'_(\d{10}-\d{10})\.tif$', record['name'], re.I).group(1)
            tiles.setdefault(tile, []).append(record['name'])
        for tile, objects in tiles.items():
            if len(objects) > 1:
                result['conflicts'].append(dict(
                    identity={field: group[field] for field in FIELDS},
                    reason='multiple chunks for identity and tile',
                    tile=tile, objects=objects))
    result['observed_groups'] = list(groups.values())
    result['already_complete'] = [
        group for group in result['observed_groups']
        if group['manifest_status'] == 'complete'
    ]
    result['ready'] = bool(result['scan_complete'] and not result['errors']
                           and not result['conflicts'] and not result['unresolved'])
    if result['ready']:
        result['candidates'] = [
            group for group in result['observed_groups']
            if group['manifest_status'] != 'complete'
        ]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('bucket', 'country', 'sensor', 'method', 'cutoff', 'output'):
        parser.add_argument('--' + flag, required=True)
    parser.add_argument('--prefix', action='append', required=True)
    parser.add_argument('--campaign', action='append', required=True)
    parser.add_argument('--manifest', help='Optional local JSON snapshot, read only')
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    repo = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists()), None)
    if repo is None or output == repo or repo in output.parents:
        parser.error('output must be outside the repository')
    if args.manifest and output == Path(args.manifest).resolve():
        parser.error('output must not overwrite the manifest')
    if output.exists():
        parser.error('output already exists')
    try:
        cutoff = parse_timestamp(args.cutoff)
        manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8')) if args.manifest else None
        from google.cloud import storage
        result = inventory(storage.Client(), args.bucket, args.prefix, args.country,
                           args.campaign, args.sensor, args.method, cutoff, manifest)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0 if result['ready'] else 1


if __name__ == '__main__':
    sys.exit(main())
