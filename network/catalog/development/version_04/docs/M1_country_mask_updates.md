# M1 country-mask correction and gradual updates

M1 keeps the existing country-wide output layout: one row per period, band and
target (GEE asset or GCS). The active campaign is not part of M1 output identity.
The country mask is painted from every feature in the configured
`CONFIG['asset_regions']`; export bounds only define the rectangular processing
extent. The mask is applied with `updateMask` (no geometry `clip`). M2 applies
the same country geometry to the COG grid and stores pixels outside it as
invalid/nodata.

## Gradual correction workflow

1. Open M1 for the country and years to be checked. Existing outputs that have
   no manifest record are recorded as pending in
   `gs://<bucket>/<gcs_library_images>/.CACHE/M1_UPDATE_MANIFEST.json`.
2. Select pending **ASSET** checkboxes in M1 and run the notebook's normal M1
   start cell. It re-exports the GEE asset to the same ID with overwrite enabled.
   Existing GCS mosaics are labeled for M2 rather than re-exported by M1.
3. Select the corresponding pending GCS checkbox in M2 and run the regular M2
   assembly cell. If a COG already exists, M2 downloads it to Colab's temporary
   directory, applies the country mask in fixed-size raster windows, writes a
   temporary replacement, and uploads it to the same object key. The old COG
   remains until the replacement upload succeeds. If only chunks exist, M2
   assembles those chunks. If neither COG nor chunks exists, use the normal M1
   export first.
4. Temporary input/output rasters are removed after processing, including on
   failure. Refresh the module to see the completed item return to **OK**; a
   failed mask or assembly remains selectable for retry.

M2 rasterizes the polygon parts of the configured country features on the
original grid using `gdal_rasterize`, caches the raster mask locally for the
Python session, and combines that mask with
the source validity mask in fixed-size windows. Its temporary TIFF is BigTIFF;
M2 checks the output COG's grid, band count and internal mask before upload.
The cache is reused only for matching polygon geometry, CRS, transform, width
and height; different grids (including 10 m and 30 m) get separate masks. Each
distinct grid consumes temporary disk space, with no automatic eviction during
the session. By default M2 uses a dedicated remote cache under the configured
bucket:

```text
gs://<bucket>/temp/registro_de_testes_de_mosaico/m2_country_masks/v1/
```

To override or disable it, set the following **after M0 initialization and
before starting M2**:

```python
import M2_mosaic_logic
M2_mosaic_logic.CONFIG['m2_country_mask_cache_gcs_prefix'] = 'gs://<bucket>/<dedicated-mask-cache-prefix>'
# To disable remote persistence instead:
# M2_mosaic_logic.CONFIG['m2_country_mask_cache_gcs_prefix'] = False
```

The override must be set again after reinitializing M0. Use an authorized prefix
separate from COGs, chunks and manifests. A session-local cache hit avoids a
download; otherwise M2 checks for an
immutable remote entry and builds and conditionally publishes one on a miss.
Cache GCS requests use a 30-second socket timeout by default and disable SDK
retries. Adjust `M0_auth_config.CONFIG['m2_country_mask_cache_gcs_timeout']`
to a finite positive number of seconds after M0 initialization if needed for
large transfers. This is a per-request socket timeout, not a total transfer
deadline; slow transfers that continue making progress may take longer.
Publication runs in a background daemon with at most three create-only attempts
for transient upload errors. It reuses the local mask without delaying bands;
session termination may interrupt publication, and an exhausted retry is not
restarted by later local cache hits. A later session can attempt publication again.
Downloads are checked against metadata, checksum, grid and mask values. An
invalid or inaccessible entry falls back to local rasterization; it is not
automatically deleted or replaced. A bad immutable entry requires administrative
investigation before future sessions can reuse that key. Remote cache permissions and GCS failures can
affect performance, but should not block local masking. Do not assume 10 m or
30 m alone identifies a grid. This cache has no automatic remote eviction; its
storage costs and retention are the operator's responsibility.

M2 detects CPUs available to the process, including affinity and visible cgroup
quotas, as well as memory limits and free temporary disk, and chooses a
conservative number of concurrent products (currently one or two). The CPU budget
is shared with GDAL compression threads (currently at most eight per product);
additional cores do not necessarily speed up a disk-bound job. Each product uses an isolated temporary
directory. The output can require substantial disk space, and the local Peru
benchmark took over an hour per full-country band. Start with one band when
validating a new Colab runtime; actual performance and disk needs vary.

The manifest identity is country, sensor, periodicity, mosaic method, period
key, band and target. Campaign is deliberately excluded. M1's default
`years=None` spans all available years; pass an explicit year range to work in
smaller batches. If the notebook supplies a fixed year list, expand it to cover
the legacy periods that should be included. The manifest is stored in GCS rather
than the cloned notebook, so progress survives Colab restarts.
