# MapBiomas Fire Catalog - Production

> **Production release.** Mirrors `development/version_03` (which mirrors
> `development/version_04` at release time). Do not edit; see `PRODUCTION.md`.
> See [`../README.md`](../README.md) for the catalog.
>
> Colab links: [`COLAB_LINKS.csv`](./COLAB_LINKS.csv) /
> [`COLAB_LINKS.txt`](./COLAB_LINKS.txt).

## Colab links

| Country | MONITOR_01 (S2 monthly) | Annual (Landsat, Collection 02) |
|---|---|---|
| Brazil | [monitor_01_brazil.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_brazil.ipynb) | [collection_02_brazil.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_brazil.ipynb) |
| Bolivia | [monitor_01_bolivia.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_bolivia.ipynb) | [collection_02_bolivia.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_bolivia.ipynb) |
| Colombia | [monitor_01_colombia.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_colombia.ipynb) | [collection_02_colombia.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_colombia.ipynb) |
| Chile | [monitor_01_chile.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_chile.ipynb) | [collection_02_chile.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_chile.ipynb) |
| Ecuador | [monitor_01_ecuador.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_ecuador.ipynb) | [collection_02_ecuador.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_ecuador.ipynb) |
| Paraguay | [monitor_01_paraguay.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_paraguay.ipynb) | [collection_02_paraguay.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_paraguay.ipynb) |
| Venezuela | [monitor_01_venezuela.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/monitor_01_venezuela.ipynb) | [collection_02_venezuela.ipynb](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/catalog/production/notebooks/collection_02_venezuela.ipynb) |

Full URLs are also available in [`COLAB_LINKS.csv`](./COLAB_LINKS.csv).

## Overview

This folder contains the classification engine of the MapBiomas Fire Catalog.
It started as the Peru monitor (Sentinel-2 monthly) and now runs any campaign
declared in `campaigns/`, with the prediction target declared in `targets/`.

The monitor flow (`MONITOR_01`) is validated. The annual collection flow
(`COLLECTION_02`, Landsat) is in final validation. Post-processing, statistics
and promotion are handled by the GEE scripts (M7-M9).

## Pipeline (M0-M9)

| Module | Responsibility | Code |
|---|---|---|
| **M0** | Campaign/target resolution, authentication, paths | `M0_auth_config.py`, `M_campaigns.py`, `M_targets.py` |
| **M1** | Export quality mosaics from GEE to GCS chunks | `M1_export_logic.py`, `M1_export_ui.py` |
| **M2** | Assemble chunks into COGs and sync the cache | `M2_mosaic_logic.py`, `M2_mosaic_ui.py` |
| **M3** | Collect training samples in GEE | `M3_toolkit.js`, `M3_sample_ui.py` |
| **M4** | Train the DNN, evaluate and save weights/metrics | `M4_algorithms_dnn.py`, `M4_data_extractor.py`, `M4_ui.py` |
| **M5** | Tile inference (probability + auxiliary band) | `M5_inference.py`, `M5_classifier.py`, `M5_classifier_ui.py` |
| **M6** | Regional mosaic, consolidated stats, GEE upload | `M6_publisher.py`, `M6_ui.py` |
| **M7** | GEE curation: ft00 selection/merge, ft01 LULC + isolated pixels, ft02 temporal memory | `src/gee/M7_*.js` |
| **M8** | GEE statistics per stage and LULC | `src/gee/M8_statistics.js` |
| **M9** | GEE promotion UI (pre-official) | `src/gee/M9_00_promote_ui.js` |

> **M6 vs M7:** M6 publishes the regional product (mosaic + statistics + GEE
> upload). LULC masking, isolated-pixel removal and the temporal filter live in
> M7 (`ft01`/`ft02`). The old "curator" role is now the M9 promotion UI.

### GEE scripts (canonical source)

The `.js` files in this folder are mirrors. The canonical scripts live in the
`mapbiomas-fire` GEE repository (`users/mapbiomasworkspace1/mapbiomas-fire`):

| Local file | GEE script |
|---|---|
| `M7_00_selection_merge.js` | [`5-Monitor-Fuego/M7/monthly/ft00_selection_merge`](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/M7/monthly/ft00_selection_merge) |
| `M7_01_lulc_filters.js` | [`5-Monitor-Fuego/M7/monthly/propuesta_a/ft01_lulc_filters`](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/M7/monthly/propuesta_a/ft01_lulc_filters) |
| `M7_02_temporal_filter.js` | [`5-Monitor-Fuego/M7/monthly/propuesta_a/ft02_temporal_filter`](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/M7/monthly/propuesta_a/ft02_temporal_filter) |
| `M8_statistics.js` | [`5-Monitor-Fuego/M8/M8_statistics`](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/M8/M8_statistics) |
| `M9_00_promote_ui.js` | [`5-Monitor-Fuego/M9/M9_00_promote_ui`](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/M9/M9_00_promote_ui) |

The GEE Code Editor links require access to the `mapbiomas-fire` repository.
When a script changes in GEE, update the mirror in `src/gee/`.

## Classification output

- Each tile stores **probability (int16, 0-1000)** and an auxiliary band
  (`dayOfYear` for `burned_area`).
- Product convention: any positive probability counts as burned
  (`probability > 0`), matching the GEE filters. Model evaluation in M4 still
  uses 0.5, because that is a model-quality metric, not the product threshold.
- Only `burned_area` is enabled today. Other targets require a profile in
  `targets/` and are rejected until implemented.

## Storage layout

```
GCS images:      {gcs_root}/LIBRARY_IMAGES/{SENSOR}/{PERIODICITY}/{MOSAIC}/{period}/COG/
GCS campaign:    {gcs_root}/{CAMPAIGN}/LIBRARY_MODELS
                 {gcs_root}/{CAMPAIGN}/LIBRARY_SAMPLES
                 {gcs_root}/{CAMPAIGN}/LIBRARY_CLASSIFICATIONS
GEE:             {gee_root}/{CAMPAIGN}/...
```

Annual assets use `ANNUALLY` in GCS and the legacy `YEARLY` folder in GEE.
The old `.../FIRE/MONITOR/VERSION_01/...` paths are no longer used.

## Campaigns and targets

```python
from M_campaigns import set_campaign
set_campaign("brazil.monitor_01", target_id="burned_area")
```

Campaigns are declared in `campaigns/*.json` and targets in `targets/*.json`.
The catalog covers seven countries; every annual campaign uses **Collection 02**.

## Environment

```bash
conda env create -f environment.yml
conda activate fire_monitor
```

- M4 uses TensorFlow `compat.v1`; M5 inference runs in NumPy (no TensorFlow).
- M2 needs GDAL/rasterio; M1/M3 need `earthengine-api`; all stages use `gcsfs`.

## Status and next steps

- `MONITOR_01` (Sentinel-2 monthly): validated.
- `COLLECTION_02` (Landsat annual): final end-to-end validation.
- M7-M9: current GEE curation/statistics/promotion flow.
- Next: finish the collection validation, add future target profiles, and
  deprecate `version_00`/`version_01` after a public notice.
