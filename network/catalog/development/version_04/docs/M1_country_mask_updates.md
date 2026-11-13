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
2. Each pending M1 cell displays a localized **Update** button. It operates on
   exactly that period, band and target; other selections are cleared.
3. GEE asset updates use the existing asset ID with Earth Engine overwrite
   enabled. For GCS, M1 exports to a unique staging prefix. After the export
   task completes, refresh M1 to promote the staged shards over the existing
   shard keys. Old shards are not removed before replacement.
4. For a GCS row, run the corresponding M2 assembly after M1 reports the
   updated chunks. M2 rasterizes the country geometry on the existing raster
   grid and uploads the corrected COG to its existing object key. Only after a
   successful COG upload are source chunks cleaned up and the manifest item
   marked complete.
5. Refresh M1 to see the completed item return to **OK**. A failed or incomplete
   task remains eligible for another update.

The manifest identity is country, sensor, periodicity, mosaic method, period
key, band and target. Campaign is deliberately excluded. M1's default
`years=None` spans all available years; pass an explicit year range to work in
smaller batches. If the notebook supplies a fixed year list, expand it to cover
the legacy periods that should be included. The manifest is stored in GCS rather
than the cloned notebook, so progress survives Colab restarts.
