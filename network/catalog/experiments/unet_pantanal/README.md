# U-Net test - Fire Monitor, Brazil, Pantanal (Sentinel-2, monthly)

Colab notebook: [`unet_monitor_pantanal.ipynb`](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/classification-unet/network/catalog/experiments/unet_pantanal/notebooks/unet_monitor_pantanal.ipynb) (branch `classification-unet`).

An independent experiment: a **U-Net** that learns burned area from image
chips, as an alternative to the pixel MLP of the catalog (version_04 M4/M5).
The default setup is the **Fire Monitor** (`MONITOR_01`): Sentinel-2 monthly
mosaics and monthly samples. This folder does not import any catalog version;
it only **reads** the data the catalog already produced in Google Cloud Storage:

- mosaics: the M1/M2 COGs in
  `{GCS_PREFIX}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/{period}/COG/`
  (one band per file, reflectance x 100 as uint8, 10 m, valid where
  `dayOfYear > 0`);
- samples: the M3 CSVs in `{GCS_PREFIX}/MONITOR_01/LIBRARY_SAMPLES/`
  (`fire` = 1/0, `region`, `period`, `.geo` polygon);
- regions: `users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2`, where each
  feature gets `region_name = bioma + "_r" + id` (computed in the code, as in
  the GEE script `f.set('region_name', f.getString('bioma').cat('_r').cat(f.get('id')))`).
  `REGION` is one of those names (for example `Pantanal_r1`; the error message
  lists the valid ones). `SAMPLE_REGION` is the value of the `region` column in
  the sample CSVs (for example `Pantanal 60`); the two names can differ.
  The older `MONITOR_01/AUXILIARY/regiones_fuego_brazil_v1` asset stays as a
  comment in `src/unet_config.py`.

**First test setup:** only `green` and `swir2` are exported for `2026_10`, so
`BANDS=["green", "swir2"]`, `INDICES=[]` (no index can be computed from these
two) and `VALID_BAND=None` (no `dayOfYear` yet: a pixel is valid where the
bands are not nodata and not all 0, and output band 2 stays -9999). The full
band and index lists are kept commented in `src/unet_config.py` and in the
notebook. With no indices, the MLP comparison has only A (bands) and C (U-Net).

`PERIODS` are the months used for training (mosaic **and** samples needed);
`PREDICT_PERIODS` are the months to classify (only the mosaic is needed;
default = `PERIODS`).

A period is a catalog period key: `2026_10` (monthly) or `2024` (annual). The
same code runs the annual Landsat collection with `CAMPAIGN="COLLECTION_06"`,
`SENSOR="landsat"`, `PERIODS=["2024"]` and `CHIP=128`.

It writes a probability raster per period in the catalog **M5 format** (int16
probability x 1000 plus `dayOfYear`, nodata -9999), so the catalog
post-processing (M6/M7) can read it.

The notebook and the printed summaries are in Portuguese for now (a language
option comes later); the texts of the step summaries live in `src/unet_resumo.py`.

## Folder structure

```
unet_pantanal/
  README.md
  requirements.txt
  notebooks/unet_monitor_pantanal.ipynb   Colab entry point
  src/
    unet_env.py       authentication (GEE + GCS service account) and GCS access
    unet_config.py    all settings in one dict (project, data, model), periods, GCS paths
    unet_io.py        mosaic reader (by window), M3 samples, region geometry, GCS writes
    unet_geo.py       spatial blocks, chip centers, windows, 1/0/255 labels
    unet_features.py  spectral indices, normalization, pixel metrics (NumPy)
    unet_chips.py     chips -> TFRecord shards, preview figure, train statistics
    unet_model.py     tf.data pipeline, U-Net, masked BCE+Dice, masked metrics, training
    unet_predict.py   sliding-window inference over the region, COG upload
    unet_report.py    test metrics, MLP baseline, comparison table
    unet_resumo.py    didactic printed summaries of the notebook steps (Portuguese)
  tests/test_unet_*.py                        unit tests
  docs/proposta_metodologia_unet_pantanal.md  slides outline: method proposal (Portuguese)
  docs/apresentacao_unet_pantanal.md          slides outline with results to fill (Portuguese)
```

## How to run

1. The monthly mosaics (catalog M1/M2) and samples (catalog M3) must be in GCS.
2. Open the notebook in Colab with a **GPU** runtime. Use a fresh runtime:
   the catalog M4 trainer switches TensorFlow to v1 mode, which breaks Keras 2.
3. Edit the config cell (`PERIODS`, `REGION`, `SAMPLE_GLOB` or `SAMPLE_FILES`)
   and run the cells in order. Each step saves its result in GCS, so a step
   can be re-run alone.

Chips of every period in `PERIODS` train **one** model; validation and test
use the same spatial blocks in every period. Each period gets its own map.

Samples follow the catalog M4 rules for the period: the `period` column
(`2026_10` or `202610`); when it is empty, the date at the end of the file
name (`..._2026_10.csv`); when it has a later year than the file name, the
file date wins. `SAMPLE_GLOB` (default `*{period}*.csv`) picks the files.

Project settings (`BUCKET`, `GCS_PREFIX`, `CAMPAIGN`, `GEE_PROJECT`,
`REGIONS_ASSET`) are in `src/unet_config.py` and can be overridden in the
notebook.

Outputs, all inside `{GCS_PREFIX}/{CAMPAIGN}/{EXPERIMENT}/`
(`EXPERIMENT` = `UNET_01` by default). The code refuses to write or delete
anywhere else.

```
chips/{period}_{split}_{shard}.tfrecord.gz  raw features (reflectance + indices) and labels
splits/blocks_{period}.gpkg                 blocks with train/val/test
splits/region_{region}.geojson              cached region geometry
stats/norm_stats.json                       clip limits, mean and std (train chips only)
models/{MODEL_NAME}.keras                   trained U-Net (+ _metadata.json, _history.csv)
predict/{region}_{period}_prob.tif          COG, M5 format
reports/chips_{period}.json                 inventory, sample counts, dropped chips, overlaps
reports/chips_preview_{period}.png          16 train chips with labels
reports/metrics_{period}.json               test metrics
reports/comparison_{periods}.json           MLP A x MLP B x U-Net table
```

## How to add a region or period

Change `REGION` and `PERIODS` in the config cell. `REGION` must match
`region_nam` in the regions asset and the `region` column of the samples
(case-insensitive). Set `FILTER_SAMPLES_BY_REGION=False` if the samples do not
carry the region. Use a new `MODEL_NAME` (or `EXPERIMENT`) for each new model
so earlier results are kept.

## Method in short

- Spatial split first: blocks of `BLOCK_KM` (25 km) with samples are shuffled
  (seed) into train/val/test (70/15/15). A chip that crosses blocks of
  different splits is dropped; an assert checks that no val/test chip touches
  a train block.
- Chip centers: up to `POINTS_PER_POLYGON` random points inside each polygon,
  then the same number of fire and not-fire centers per split.
- Chips: 256 x 256 px (2.56 km at 10 m).
- Labels: 1 inside fire polygons, 0 inside not-fire polygons, 255 elsewhere
  and on nodata. Fire wins where both overlap (logged).
- Features: first test green + swir2; full set 5 bands (blue, red, nir,
  swir1, swir2) + NBR, NBR2, MIRBI, NDVI, BAI, EVI, from one function used
  for chips and inference. Values are
  clipped to the train 1-99 percentiles, then standardized.
- Model: U-Net, filters 32-64-128-256, bottom 512, batch norm, sigmoid output,
  masked BCE + Dice, early stopping on validation IoU of the fire class.
- Threshold: best F1 on validation; the test is reported at that threshold
  and at 0.5. The product keeps the M5 rule (stored probability `> 0`), so a
  cut can still be chosen later in GEE.
- Baseline: the catalog M4 network (layers 7-14-7-14-7, Adam 0.001, 7000
  batches of 1000, no dropout) re-implemented in TF2 and trained on the
  labeled pixels of the same train chips; evaluated on the same test pixels.

## Tests

From the repository root (TensorFlow, rasterio and geopandas are needed for
some tests; they are skipped when missing):

```
python -m unittest discover -s network/catalog/experiments/unet_pantanal/tests -p "test_unet_*.py"
```

## Known limits

- Polygons are labeled only inside; pixels between polygons are unknown (255).
  If the U-Net misses edges, the planned fix is to fully label some chips.
- Dark water in the Pantanal can look like a burn scar; check the preview
  chips over lagoons and wet areas. Monthly mosaics can also keep clouds or
  cloud shadows.
- One month may have few samples; add more months to `PERIODS` to train with
  more chips.
- The Pantanal at 10 m is large; classification of one month reads and
  writes billions of pixels (expect a long run on Colab).
- The MLP baseline is scored on pixels, not on the M5 tile product.
