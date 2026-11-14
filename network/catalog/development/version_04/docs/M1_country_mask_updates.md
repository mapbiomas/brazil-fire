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
   enabled. GCS checks the actual objects again when **Update** is clicked:
   - If the COG exists, M2 downloads and masks that COG on its original grid,
     rebuilds it, then overwrites the same object. This path creates no Earth
     Engine task and leaves the old COG intact until the replacement upload.
   - If the COG is absent but chunks exist, M2 builds the masked COG from those
     chunks. This path also creates no Earth Engine task.
   - If neither COG nor chunks exist, M1 uses the normal Earth Engine export
     flow for a new product.
4. Refresh M1 to see the completed item return to **OK**. Failed masking or
   assembly leaves the item eligible for retry; an absent product remains
   eligible for the normal export path.

The manifest identity is country, sensor, periodicity, mosaic method, period
key, band and target. Campaign is deliberately excluded. M1's default
`years=None` spans all available years; pass an explicit year range to work in
smaller batches. If the notebook supplies a fixed year list, expand it to cover
the legacy periods that should be included. The manifest is stored in GCS rather
than the cloned notebook, so progress survives Colab restarts.
