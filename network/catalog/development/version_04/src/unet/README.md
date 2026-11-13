# U-Net test (burned area, Landsat annual)

Colab notebook: [`unet_01_brazil_pantanal.ipynb`](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/classification-unet/network/catalog/development/version_04/notebooks/unet_01_brazil_pantanal.ipynb) (branch `classification-unet`).

An experimental alternative to the pixel MLP of version_04 (M4 training, M5
inference). The U-Net learns from image chips instead of single pixels. It
uses the **same inputs** as the MLP flow:

- mosaics: the M2 COGs in `LIBRARY_IMAGES/{SENSOR}/ANNUALLY/MINNBR/{year}/COG/`
  (one band per file, Landsat reflectance x 100 as uint8, valid where
  `dayOfYear > 0`);
- samples: the M3 CSVs in `{campaign}/LIBRARY_SAMPLES/` (`fire` = 1/0,
  `region`, `period`, `.geo` polygon).

It produces a probability raster in the **M5 format** (int16 probability x 1000
plus `dayOfYear`, nodata -9999), so the M6/M7 post-processing can read it.
Nothing in `src/core` is changed. The first test is Brazil, COLLECTION_06,
Pantanal, 2024.

## Folder structure

```
src/unet/
  unet_config.py    config defaults (one dict), GCS paths
  unet_io.py        mosaic reader (by window), M3 samples, region geometry, GCS writes
  unet_geo.py       spatial blocks, chip centers, windows, 1/0/255 labels
  unet_features.py  spectral indices, normalization, pixel metrics (NumPy)
  unet_chips.py     chips -> TFRecord shards, preview figure, train statistics
  unet_model.py     tf.data pipeline, U-Net, masked BCE+Dice, masked metrics, training
  unet_predict.py   sliding-window inference over the region, COG upload
  unet_report.py    test metrics, MLP baseline, comparison table
notebooks/unet_01_brazil_pantanal.ipynb   Colab entry point
tests/test_unet_*.py                      unit tests
docs/unet/                                method notes for presentations
```

## How to run

1. Export the mosaics (M1/M2) and the samples (M3) as usual.
2. Open the notebook in Colab with a **GPU** runtime. Use a fresh runtime:
   the M4 trainer switches TensorFlow to v1 mode, which breaks Keras 2.
3. Edit the config cell (`REGION`, `YEARS`, `SAMPLE_GLOB` or `SAMPLE_FILES`)
   and run the cells in order. Each step saves its result in GCS, so a step
   can be re-run alone.

Outputs, all inside `{gcs_campaigns_prefix}/{campaign}/{EXPERIMENT}/`
(`EXPERIMENT` = `UNET_01` by default). The code refuses to write anywhere else.

```
chips/{year}_{split}_{shard}.tfrecord.gz   raw features (reflectance + indices) and labels
splits/blocks_{year}.gpkg                  blocks with train/val/test
splits/region_{region}.geojson             cached region geometry
stats/norm_stats.json                      clip limits, mean and std (train chips only)
models/{MODEL_NAME}.keras                  trained U-Net (+ _metadata.json, _history.csv)
predict/{region}_{year}_prob.tif           COG, M5 format
reports/chips_{year}.json                  inventory, sample counts, dropped chips, overlaps
reports/chips_preview_{year}.png           16 train chips with labels
reports/metrics_{year}.json                test metrics
reports/comparison_{years}.json            MLP A x MLP B x U-Net table
```

## How to add a region or year

Change `REGION` and `YEARS` in the config cell. `REGION` must match
`region_nam` in the campaign's regions asset and the `region` column of the
samples (case-insensitive). Set `FILTER_SAMPLES_BY_REGION=False` if the
samples do not carry the region. Use a new `MODEL_NAME` (or `EXPERIMENT`)
for each new model so earlier results are kept.

## Method in short

- Spatial split first: blocks of `BLOCK_KM` (25 km) with samples are shuffled
  (seed) into train/val/test (70/15/15). A chip that crosses blocks of
  different splits is dropped; an assert checks that no val/test chip touches
  a train block.
- Chip centers: up to `POINTS_PER_POLYGON` random points inside each polygon,
  then the same number of fire and not-fire centers per split.
- Labels: 1 inside fire polygons, 0 inside not-fire polygons, 255 elsewhere
  and on nodata. Fire wins where both overlap (logged).
- Features: 5 bands (blue, red, nir, swir1, swir2) + NBR, NBR2, MIRBI, NDVI,
  BAI, EVI, from one function used for chips and inference. Values are
  clipped to the train 1-99 percentiles, then standardized.
- Model: U-Net, filters 32-64-128-256, bottom 512, batch norm, sigmoid output,
  masked BCE + Dice, early stopping on validation IoU of the fire class.
- Threshold: best F1 on validation; the test is reported at that threshold
  and at 0.5. The product keeps the M5 rule (stored probability `> 0`), so a
  cut can still be chosen later in GEE.
- Baseline: the M4 network (layers 7-14-7-14-7, Adam 0.001, 7000 batches of
  1000, no dropout) re-implemented in TF2 and trained on the labeled pixels
  of the same train chips; evaluated on the same test pixels.

## Tests

From the repository root (TensorFlow, rasterio and geopandas are needed for
some tests; they are skipped when missing):

```
python -m unittest discover -s network/catalog/development/version_04/tests -p "test_unet_*.py"
```

## Known limits

- Polygons are labeled only inside; pixels between polygons are unknown (255).
  If the U-Net misses edges, the planned fix is to fully label some chips.
- Dark water in the Pantanal can look like a burn scar; check the preview
  chips over lagoons and wet areas.
- The MLP baseline is scored on pixels, not on the M5 tile product.
