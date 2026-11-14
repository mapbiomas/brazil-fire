# Repairing exported regional classifications

`M6_region_mask_repair.py` is a campaign-parameterized, GEE-first migration
utility for classification images already published under
`LIBRARY_CLASSIFICATIONS/REGIONAL`. It reuses the existing classification; it
does not run M5 again. GCS correction is intentionally deferred to a separate
second phase.

## Safety model

- Inventory is read-only. Only image assets below the configured campaign's
  `LIBRARY_CLASSIFICATIONS/REGIONAL` path are considered.
- Each asset must resolve to exactly one known source region and period, using
  image properties first and a strict asset-name fallback. Ambiguous assets are
  reported as skipped; they are never guessed.
- The region geometry is selected from `CONFIG['asset_regions']` by
  `REGION_NAME_PROPERTY`. The image is masked with
  `ee.Image(0).byte().paint(region_features, 1).gt(0)`.
- The replacement uses the same asset ID, the region's rectangular bounds,
  source CRS/transform, `copyProperties`, and `overwrite=True`. No asset is
  deleted before export.
- Exports are checkpointed in a campaign-scoped GCS manifest. The submit
  function requires the exact confirmation string `OVERWRITE <campaign>` and
  defaults to at most two active/submitted tasks at a time.

## Peru `MONITOR_DEV` pilot

Initialize and authenticate the ordinary v04 configuration for Peru with
`campaign='MONITOR_DEV'` and its matching `asset_regions` FeatureCollection.
Then run an inventory before submitting any work:

```python
from M6_region_mask_repair import discover_gee_classifications

inventory = discover_gee_classifications('MONITOR_DEV')
print('GEE images:', len(inventory['items']))
print('Skipped for review:', len(inventory['skipped']))
for item in inventory['items']:
    print(item['model_id'], item['region'], item['period'], item['asset_id'])
for item in inventory['skipped']:
    print('REVIEW:', item['asset_id'], item['reason'])
```

Review the inventory and skipped entries first. For a single-item pilot, select
one verified item and start only one task:

```python
from M6_region_mask_repair import submit_gee_repairs

pilot = [item for item in inventory['items'] if not item['already_masked']][:1]
manifest = submit_gee_repairs(
    pilot,
    campaign='MONITOR_DEV',
    confirmation='OVERWRITE MONITOR_DEV',
    max_submissions=1,
    max_active_tasks=1,
)
```

The manifest records task IDs and reconciles task states on the next call. To
continue after the pilot succeeds, pass the reviewed inventory with the normal
two-task cap. Failed entries require `retry_failed=True`; an entry already
marked with `region_mask_version='source_region_v1'` is skipped.

Before expanding to another country, confirm its campaign path, region asset,
region-name property, asset properties, source projection, and overwrite
permissions. No authenticated exports are run by the offline tests. GCS objects
are not modified by this utility.
