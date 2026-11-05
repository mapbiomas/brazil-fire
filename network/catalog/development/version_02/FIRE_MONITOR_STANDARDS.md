# MapBiomas Fire Monitor — Development Standards

> Pipeline M0-M8 | Sentinel-2 | Peru Fire Monitor
> Updated: 2026-06-05 | Última revisão: notebooks parametrizados, M5 diagnostic, asset por campanha

---

## 1. Architecture Overview

### 1.1 Code Splitting

All modules split into two files:

- **`M#_logic.py`** — Pure processing: Earth Engine, GDAL, file I/O. **No ipywidgets.**
- **`M#_ui.py`** — UI class inheriting from `PipelineStepUI`. Layout, buttons, reactivity.

### 1.2 Module Inventory

| Module | Type | Status | Key files |
|--------|------|--------|-----------|
| M0 | Config + Auth | ✅ Stable | `M0_auth_config.py`, `M_regions.py`, `M_lang.py` |
| M1 | Export (GEE → GCS) | ✅ Stable | `M1_export_logic.py`, `M1_export_ui.py` |
| M2 | Mosaic Assembly (COG) | ✅ Stable | `M2_mosaic_logic.py`, `M2_mosaic_ui.py` |
| M3 | Sample Collection | ✅ Stable | `M3_sample_ui.py`, `M3_toolkit.js` |
| M4 | DNN Training | ✅ Stable | `M4_algorithms_dnn.py`, `M4_analytics.py`, `M4_data_extractor.py`, `M4_ui.py` |
| M5 | Classification | ✅ Stable | `M5_classifier.py`, `M5_classifier_ui.py`, `M5_inference.py`, `M5_workplan.py` |
| M6 | Post-Processing | ✅ Stable | `M6_publisher.py`, `M6_ui.py` |
| M7 | Curation | 🔨 Stub | `M7_curator.py`, `M7_filter.py` |
| M8 | Final Release | 🔨 Stub | `M8_curator.py` |

Infrastructure: `M_cache.py`, `M_gcs.py`, `M_lang.py`, `M_mosaics.py`, `M_regions.py`, `M_ui_components.py`

---

## 2. Campaign Concept

Outputs are organized by **campaign** — a named directory grouping `LIBRARY_CLASSIFICATIONS`, `LIBRARY_MODELS`, and `LIBRARY_SAMPLES`. Images and cache are shared across campaigns.

```
CATALOG_01/
├── LIBRARY_IMAGES/           ← shared (all campaigns)
├── .CACHE/                   ← shared
├── CHUNKS/                   ← shared
├── MONITOR_01/               ← campaign (continuous monitoring)
│   ├── LIBRARY_CLASSIFICATIONS/
│   ├── LIBRARY_MODELS/
│   └── LIBRARY_SAMPLES/
│       └── AUXILIARY/        ← region FeatureCollection
└── COLLECTION_01/            ← future (annual collection)
```

**Rules:**
- Campaign name is `CAMPAIGN` in `set_global_opts()` — never hidden in code.
- No redundant subfolders: the campaign directory IS the namespace.
- Each campaign has isolated outputs; shared resources (images, cache) stay at root.
- **Each campaign needs its own copy of the region FeatureCollection** (`AUXILIARY/regiones_fuego_{country}_v1`). Copy from the canonical source before first use:
  ```python
  import ee
  ee.data.createAsset({'type': 'FOLDER'}, 'projects/mapbiomas-{country}/assets/FIRE/CATALOG_01/{campaign}')
  ee.data.createAsset({'type': 'FOLDER'}, 'projects/mapbiomas-{country}/assets/FIRE/CATALOG_01/{campaign}/AUXILIARY')
  ee.data.copyAsset(
    'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/regiones_fuego_{country}_v1',
    'projects/mapbiomas-{country}/assets/FIRE/CATALOG_01/{campaign}/AUXILIARY/regiones_fuego_{country}_v1'
  )
  ```
- See ADR: `adr/005-campaign-concept.md`

---

## 3. Module Details

### M0 — Setup & Authentication

Agnostic config: **9 required parameters, zero defaults.** No MapBiomas-specific assumptions.

**Notebook branch strategy**

Two parallel notebooks: `mapbiomas_fire_sentinel_peru.ipynb` (main/production) + `_dev.ipynb` (development). Both parameterized with `CONTRY_SELECTED` for easy multi-country reuse:

```python
CONTRY_SELECTED = 'peru'  # change to 'bolivia', 'chile', etc.

import sys, os, subprocess, shutil
if os.path.exists("/content/fire_monitor"):
    shutil.rmtree("/content/fire_monitor")
subprocess.run(["git", "clone", "-b", "BRANCH",    # BRANCH = 'dev' or 'main'
    "https://github.com/mapbiomas/peru-fire.git", "fire_monitor"])
from M0_auth_config import set_global_opts, authenticate, print_config
authenticate(project=F'mapbiomas-{CONTRY_SELECTED}')

set_global_opts(
    country=F'{CONTRY_SELECTED}',
    campaign='CAMPAIGN',                              # MONITOR_01 (main) / MONITOR_DEV (dev)
    sensor='sentinel2',                               # string for one; list only for >1
    periodicity='monthly',                            # 'monthly' or 'yearly'
    mosaic_methods=['minnbr', 'minnbr_buffer'],
    personal_task_flag='TASK_FLAG',                   # MONITOR (main) / MONITOR_DEV (dev)
    language='en',
    gcs_bucket='mapbiomas-fire',
    gcs_library_images_prefix=F'sudamerica/{CONTRY_SELECTED}/CATALOG_01',
    gcs_campaigns_prefix=F'sudamerica/{CONTRY_SELECTED}/CATALOG_01',
    gee_project=F'mapbiomas-{CONTRY_SELECTED}',
    gee_library_images_prefix=F'projects/mapbiomas-{CONTRY_SELECTED}/assets/FIRE/CATALOG_01',
    gee_campaigns_prefix=F'projects/mapbiomas-{CONTRY_SELECTED}/assets/FIRE/CATALOG_01',
    asset_regions=F'projects/mapbiomas-{CONTRY_SELECTED}/assets/FIRE/CATALOG_01/{campaign}/AUXILIARY/regiones_fuego_{CONTRY_SELECTED}_v1',
)
print_config()
```

**Key differences between notebooks:**

| | `peru.ipynb` (main) | `peru_dev.ipynb` (dev) |
|--|---------------------|----------------------|
| Clone branch | `main` (no `-b`) | `dev` (`-b dev`) |
| Campaign | `MONITOR_01` | `MONITOR_DEV` |
| Task flag | `MONITOR` | `MONITOR_DEV` |
| Asset regions | `.../MONITOR_01/AUXILIARY/...` | `.../MONITOR_DEV/AUXILIARY/...` |

**Dev notebook** exists only in the `dev` branch — never merged into `main`. See `adr/009-dev-branch-strategy.md`.

Dynamic path helpers replace fixed CONFIG keys:
- `gcs_samples_path(campaign)` → `{prefix}/{campaign}/LIBRARY_SAMPLES`
- `gcs_models_path(campaign)` → `{prefix}/{campaign}/LIBRARY_MODELS`
- `gcs_classifications_path(campaign)` → `{prefix}/{campaign}/LIBRARY_CLASSIFICATIONS`
- `gee_samples_path(campaign)`, `gee_classifications_path(campaign)` — GEE equivalents

See ADR: `adr/006-agnostic-zero-defaults.md`

### M1 — Export (GEE → GCS)

Multi-sensor export: Sentinel-2, Landsat 5/7/8/9, MODIS, HLS. Sensor-specific radiometric corrections and cloud masking. Outputs GeoTIFF chunks to `{gcs_library_images_prefix}/LIBRARY_IMAGES/{SENSOR}/{PERIOD}/{MOSAIC}/{temporal_id}/CHUNKS/`.

Supports monthly and annual periods with configurable mosaic types (minnbr, minnbr_buffer, median, minndvi).

### M2 — Mosaic Assembly (COG)

Assembles M1 chunks into Cloud-Optimized GeoTIFFs via GDAL VRT → compressed COG (DEFLATE). Outputs to `.../{temporal_id}/COG/`. Results consumed by M4 (training) and M5 (classification). After a successful COG upload, M2 removes the source chunks for that band; rebuilding a COG requires re-exporting the chunks from M1.

### M3 — Sample Collection (GEE JS Toolkit)

Interactive polygon drawing on satellite imagery in GEE Code Editor. Fire (burned) and notFire (unburned) polygons exported as FeatureCollections (GEE) + CSV (GCS) with metadata.

**Access:**
- Source code: `src/core/M3_toolkit.js`
- GEE Editor: `users/mapbiomasworkspace1/mapbiomas-fire:5-Monitor-Fuego/Toolkit_Monitor_Fuego`
- Tutorial slides: Google Slides link in `M3_sample_ui.py`

**5 UI locales:** `en`, `es`, `pt`, `fr`, `id` (configurable via `APP_LANG` in toolkit).

**Import rule (critical):** When loading samples back into drawing tools, use `Map.drawingTools().addLayer({geometries: geomList})` — **never** `drawingLayer.geometries().reset()` with data from `.evaluate()`. The `reset()` method fails with `Internal error: object not in API` on GeoJSON-converted geometries.

See ADR: `adr/008-m3-import-addlayer-vs-reset.md`

### M4 — DNN Training

Trains a Deep Neural Network classifier from M3 polygon samples + M2 COG mosaics.

**Features:**
- Flexible band extraction matrix (any sensor × mosaic × band combination)
- PCA 3D latent space projection (t-SNE removed)
- 2×3 KPI grid: Accuracy, Precision, Recall / F1-Score, Auto Rating, Your Rating
- Single sync button with spinner (3 removed)
- Auto-suggest shortname: `{region}_{bandCount}b` (e.g. `r1_4b`, `multiregion_4b`)
- Path always built via `gcs_models_path() + training_id` — never trusts cache
- Cache deleted at startup (`m4_ranking_cache.json` + `state.json`)

See ADR: `adr/007-m4-redesign.md`

### M5 — Tile-Based Classification

Loads DNN model from GCS, classifies each tile (cim-world-1-250000 grid) independently. Produces int16 2-band rasters: band 0 = probability (0-1000), band 1 = dayOfYear.

**Region selection — auto-population logic:**

In `_populate_dropdowns()` (`M5_classifier_ui.py:201`), the region checkbox grid is populated by querying GEE:

```python
asset = CONFIG['asset_regions']  # set in M0 notebook
fc = ee.FeatureCollection(asset)
names = fc.aggregate_array('region_nam').distinct().getInfo()
```

If the query returns names, regions are populated automatically. Otherwise a diagnostic banner is shown in the UI with the exact cause:

| Diagnóstico | Causa provável |
|-------------|----------------|
| `Asset nao encontrado no GEE: {path}` | FeatureCollection path doesn't exist in GEE. Check `CONFIG[asset_regions]`. |
| `Asset existe mas esta VAZIO (0 features)` | Campaign copy exists but is empty. Re-copy from `AUXILIARY_DATA/`. |
| `Coluna {property} nao encontrada` | Asset has features but `region_nam` property is missing. Check `M_regions.py`. |

**Property `region_nam`** is defined in `M_regions.py:9` and must match the target property in the GEE FeatureCollection. Every country's region asset follows the same schema.

### M6 — Post-Processing & Publication

Assembles M5 tiles into regional mosaics, applies LULC mask and sieve filter, computes burned area statistics (CSV: tile + region), uploads to GEE as ImageCollections. Interactive dashboard for batch monitoring and coverage visualization.

### M7 / M8 — Curation & Release (under development)

M7: spatial/temporal masks, variant listing. M8: best-selection evaluation, official publication.

---

## 4. Data Structure

### 4.1 GCS Paths

```
gs://{bucket}/{gcs_library_images_prefix}/
└── LIBRARY_IMAGES/
    └── {SENSOR}/{PERIOD}/{MOSAIC}/{temporal_id}/
        ├── CHUNKS/          ← M1 output
        └── COG/             ← M2 output

gs://{bucket}/{gcs_campaigns_prefix}/
├── .CACHE/
└── {CAMPAIGN}/
    ├── LIBRARY_SAMPLES/     ← M3 export (CSV)
    ├── LIBRARY_MODELS/      ← M4 output (weights.npz + metadata.json)
    │   └── {model_id}/
    │       ├── weights.npz
    │       ├── metadata.json
    │       └── workplan/
    └── LIBRARY_CLASSIFICATIONS/  ← M5 + M6
        └── {model_id}/
            ├── CLASSIFIED_TILES/    ← M5: per-tile rasters
            ├── CLASSIFIED_REGION/   ← M6: regional mosaics
            └── STATS/              ← M6: CSV statistics
```

### 4.2 GEE Assets

```
projects/{gee_project}/assets/FIRE/CATALOG_01/
├── LIBRARY_IMAGES/
│   └── {SENSOR}/{PERIOD}/{MOSAIC}/{band}/
│       └── image_{country}_fire_{sensor}_{mosaic}_{band}_{temporal_id}
└── {CAMPAIGN}/
    ├── AUXILIARY/
    │   └── regiones_fuego_{country}_v1     (FeatureCollection)
    ├── LIBRARY_SAMPLES/                     (FeatureCollections)
    │   └── sample_{id}_{region}_{period}
    └── LIBRARY_CLASSIFICATIONS/
        └── REGIONAL/
            └── {model_id}/                  (ImageCollection)
                └── {region}_{period}        (Image: probability + dayOfYear)
```

### 4.3 Region FeatureCollections

```
projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/regiones_fuego_{country}_v1
```

Centralized in `M_regions.py`. 10 countries registered. Property name: `region_nam`.

---

## 5. Naming Conventions

| Entity | Pattern | Example |
|--------|---------|---------|
| Mosaic image (GEE) | `image_{country}_fire_{sensor}_{mosaic}_{band}_{date}` | `image_peru_fire_sentinel2_minnbr_swir1_2025_08` |
| Chunk (GCS) | `image_{country}_fire_{sensor}_{mosaic}_{band}_{date}_{tileid}.tif` | `image_peru_fire_sentinel2_minnbr_nir_2025_08_0000196608-0000131072.tif` |
| COG (GCS) | `image_{country}_fire_{sensor}_{mosaic}_{band}_{date}_cog.tif` | `image_peru_fire_sentinel2_minnbr_nir_2025_08_cog.tif` |
| Sample (GEE + GCS) | `samples_{id}_{region}_{period}` | `samples_0001_r1_costa_norte_2025_05` |
| Model    | `training_{id}_{shortname}_{sensor}` | `training_001_peru_r1_v1_sentinel2` |
| Tile     | `tile_{region}_{cell}_{period}.tif` | `tile_peru_r1_costa_norte_SA-00000_2025_05.tif` |
| Regional mosaic | `region_{region}_{model}_{period}.tif` | `region_peru_r1_training_001_2025_05.tif` |

---

## 6. Internationalization

All user-facing text must use `L.XXX` from `M_lang.py`. Tech messages (print/log/raise) must be in English.

**5 supported locales:** `en` (default class attributes), `es` (STRINGS_ES), `pt` (STRINGS_PT), `fr` (STRINGS_FR), `id` (STRINGS_ID).

To add a locale: create `STRINGS_XX` dict, register in `SUPPORTED_LOCALES`, call `L.load_locale('xx')`.

For the M3 toolkit (JavaScript): edit `APP_LANG` variable and add the corresponding block in the `L` dictionary.

---

## 7. Performance Rules

- **Async:** Always dispatch to GEE via `ee.batch.Export`. Never block the UI.
- **Cache:** Use `M_cache.py` to avoid repetitive GCS listings. Cache is rebuilt on M0 startup via `clean_cache=True`.
- **Colab:** Notebook auto-clones fresh repo + runs `authenticate()` before `set_global_opts()`. No persistent local state. `CONTRY_SELECTED` variable at the top of M0 cell defines all project paths via f-strings.
- **Task naming:** All GEE tasks prefixed with `PERSONAL_TASK_FLAG` for filtering.
- **GDAL:** Auto-detected via `ensure_gdal_path()` (Windows, Linux, Colab).

---

## 8. Key Architectural Decisions

| ADR | Topic | Rule |
|-----|-------|------|
| 005 | Campaign concept | Outputs under `{campaign}/` in GCS and GEE |
| 006 | Agnostic M0 | 9 required params, zero defaults, dynamic helpers |
| 007 | M4 redesign | PCA only, single sync, 2×3 KPIs, auto-shortname |
| 008 | M3 import | Use `addLayer()`, never `geometries().reset()` with `.evaluate()` |
| 009 | Dev branch strategy | `dev` branch with isolated notebook; merge preserving M0 via `--ours` |
| — | M5 region diagnostic | Auto-populates from `CONFIG[asset_regions]`; shows banner if asset not found / empty / wrong column |
| — | Notebook parametrização | `CONTRY_SELECTED` + f-strings for all GCS/GEE paths; single variable change for new countries |
