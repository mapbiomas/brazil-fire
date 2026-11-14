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

The manifest identity is country, sensor, periodicity, mosaic method, period
key, band and target. Campaign is deliberately excluded. M1's default
`years=None` spans all available years; pass an explicit year range to work in
smaller batches. If the notebook supplies a fixed year list, expand it to cover
the legacy periods that should be included. The manifest is stored in GCS rather
than the cloned notebook, so progress survives Colab restarts.
