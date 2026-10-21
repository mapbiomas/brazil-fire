# MapBiomas Fire Monitor - Pipeline (M0-M6)

## Documentation and User Guide
Expand this section to understand the data architecture, environment requirements, and workflow.

### Introduction and Context

This pipeline is the automated processing core for the **MapBiomas Fire Scar Monitoring**. It is organized into sequential stages (M0-M8), each with a specific responsibility:

- **M0 - Setup and Authentication**: Auto-detects Colab vs local environment, installs dependencies, configures GCS/GEE project paths via `set_global_opts()`, authenticates with Earth Engine, and sets the language locale. Centralizes all configuration (sensor, periodicity, country, language, GEE/GCS project).

- **M1 - Export (GEE → GCS)**: Multi-sensor satellite image export. Handles Sentinel-2 (and in future: Landsat 5/7/8/9, MODIS, HLS). Applies sensor-specific radiometric corrections, cloud masking (QA_PIXEL, Fmask, CS), and exports optimized GeoTIFF chunks to Google Cloud Storage. Supports monthly and annual periods with configurable mosaics (minnbr, minnbr_buffer).

- **M2 - Mosaic Assembly (COG)**: Assembles exported chunks into full-region Cloud-Optimized GeoTIFFs using GDAL VRT for virtual stacking. Produces compressed, tiled COGs with LZW compression.

- **M3 - Sample Collection (GEE Toolkit JavaScript)**: Training data collection via a custom GEE JavaScript Toolkit. Users draw fire (burned) and notFire (unburned) polygons on satellite imagery. Samples are exported with full metadata. Supports multi-country, multi-language (EN/ES/PT/FR/ID), and regional classification layers (classification, probability, dayOfYear).

- **M4 - DNN Training**: Deep Neural Network classification. Uses a flexible band extraction matrix to sample pixel values from M2 mosaics using M3 polygon samples. Trains a DNN classifier with configurable architecture, generates t-SNE audit plots, and saves model weights and metadata to GCS.

- **M5 - Tile Classification**: Produces int16 2-band COG tile rasters (probability 0–1000 + dayOfYear) for each cim-world grid cell. Classifies tiles independently using a trained DNN model from M4. Results are consumed by M6 for mosaicking and publication.

- **M6 - Mosaic, Statistics & Publication**: Assembles classified M5 tiles into regional mosaics, computes burned area statistics (CSV), and uploads results to Google Earth Engine. Includes an interactive dashboard for batch monitoring, analytics filtering, and coverage visualization.

- **M7 - Post-Classification Masks**: 🔨 **Under development.** Will apply spatial and temporal masks to classified outputs before final publication.

- **M8 - Final Version Evaluation & Official Release**: 🔨 **Under development.** Will evaluate candidate versions and select the official release for publication.

**Global Inputs:** Raw image collections from Google Earth Engine (Sentinel-2), vector polygons for training (samples), and auxiliary land cover maps.

**Global Outputs:** Optimized chunks and mosaics (COGs) on Google Cloud Storage (GCS), neural network model weights (DNN), int16 2-band classification rasters (probability + dayOfYear), consolidated statistics (CSV), and versioned ImageCollections on GEE.

> **Note:** Both the local PC disk and Colab temporary storage act as **ephemeral space**. Persistence always occurs in the **Google Cloud Storage (GCS) Bucket**.

### Data Lifecycle and Naming Rules

| Stage | Weight | Inputs → Outputs | Naming Rule | Example |
| :--- | :--- | :--- | :--- | :--- |
| **M1: Export** | **Light** | **IN:** GEE Collections<br>**OUT:** GCS Chunks | `image_{country}_fire_{sensor}_{mosaic}_{band}_{temporal_id}_{suffix}` | `image_peru_fire_sentinel2_minnbr_blue_2025_08_00000-00000` |
| **M2: Mosaic** | **Medium** | **IN:** GCS Chunks<br>**OUT:** COG Mosaics (GCS) | same as M1 seed, stored in `/COG/` dir | `image_peru_fire_sentinel2_minnbr_blue_2025_08` |
| **M3: Samples** | **Light** | **IN:** Mosaics (M2)<br>**OUT:** Polygons (GCS/Asset) | `sample_{id}_{temporal_id}`| `sample_0001_2025_07` |
| **M4: Train** | **Medium** | **IN:** M3 Samples + M2 Mosaics<br>**OUT:** DNN (GCS) | `training_{id}_{shortname}_{sensor}` | `training_0001_amazon_sentinel2` |
| **M5: Classify**| **Heavy**| **IN:** M4 Model + M2 Mosaics<br>**OUT:** int16 Tiles (GCS) | `tile_{region}_{period}_{cell}.tif` | `tile_amazonia_2025_08_SA-24-V-C.tif` |
| **M6: Publish**| **Heavy**| **IN:** M5 Tiles<br>**OUT:** Mosaics + Stats CSV + GEE Assets | `{region}_{period}_mosaic.tif` | `peru_2025_08_mosaic.tif` |
| **M7: Mask** | — | Post-Classification Masks | — | 🔨 Under development |
| **M8: Release**| — | Final Version Evaluation & Release | — | 🔨 Under development |

### Data Architecture and Relationships (M1-M6)

The monitor operates a circular synchronization flow between three environments:

| Environment | Components | Role |
| :--- | :--- | :--- |
| **Google Earth Engine** | ImageCollections / Assets | Raw data source and final destination |
| **Google Cloud Storage** | Chunks / COGs / Models / Samples / Classifications | Persistence and central workspace |
| **Local Cache (Temp)** | VRT Stack / Tiles / NumPy Arrays | High-speed I/O processing |

#### Persistence Map (Where to find the data)

| Stage | Extension | Main Cloud Storage Path (GCS) |
| :--- | :--- | :--- |
| **M1: Export** | `.tif` | `{gcs_catalog_prefix}/LIBRARY_IMAGES/{SENSOR}/MONTHLY/{MOSAIC}/{date}/CHUNKS/` |
| **M2: Mosaic** | `.tif` | `{gcs_catalog_prefix}/LIBRARY_IMAGES/{SENSOR}/MONTHLY/{MOSAIC}/{date}/COG/` |
| **M3: Samples** | `.csv` | `{gcs_catalog_prefix}/LIBRARY_SAMPLES/{campaign}/` |
| **M4: Train** | `.pb / .json` | `{gcs_catalog_prefix}/LIBRARY_MODELS/training_{id}_{shortname}_{sensor}/` |
| **M5: Classify** | `.tif` | `{gcs_catalog_prefix}/LIBRARY_CLASSIFICATIONS/{model_id}/CLASSIFIED_TILES/` (tiles) |
| | `.tif` | `{gcs_catalog_prefix}/LIBRARY_CLASSIFICATIONS/{model_id}/CLASSIFIED_REGION/` (mosaics) |
| | `.csv` | `{gcs_catalog_prefix}/LIBRARY_CLASSIFICATIONS/{model_id}/STATS/` (statistics) |

#### Standard Naming Rules
- **Images (M1/M2):** `image_{country}_fire_{sensor}_{mosaic}_{band}_{temporal_id}`
- **Samples (M3):** `sample_{id}_{country}_{region}_{temporal_id}`
- **Classification (M5):** `tile_{region}_{period}_{cell}.tif` (int16 2-band: probability + dayOfYear)
- **Mosaic (M6):** `{region}_{period}_mosaic.tif`

---

### Feedback and Fault Tolerance

- **Remaining:** The "Select Remaining" button will only mark items **pending** in GCS.
- **Overwrite:** Manually re-marking a success-flagged item (green) indicates the intention to replace the file.
- **Edit Mode (`EDIT_MODE`):** If activated in M0, the UI exposes "Delete" buttons in GCS.

---

## [M0] — Environment Setup (Choose a Route)

### Option A: Google Colab Initialization

```python
# M0.1a — Colab environment setup (Clone repo)
import os
if not os.path.exists("fire_monitor"):
    !git clone https://github.com/mapbiomas/peru-fire.git fire_monitor
%cd fire_monitor/mapbiomas_fire_monitor/version_01/src/core
```

```python
# Install GDAL binaries and Python dependencies
!apt-get update -qq && apt-get install -y -qq gdal-bin python3-gdal
!pip install -q earthengine-api gcsfs rasterio scipy tqdm
```

### Option B: Local Initialization

**Local Requirements (GDAL / Conda)**
If you receive a `Missing vital dependencies` error, the monitor will automatically search for GDAL in your Conda installation.

```python
# M0.1b — Local path configuration (Optional, M0.2 does this automatically)
import sys, os
REPO_ROOT = os.path.abspath("..")
SRC_PATH  = os.path.join(REPO_ROOT, "src", "core")
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
```

### Common Initialization (M0.2)

**"Invincible" Cell:** This cell auto-detects if you are in Local or Colab and configures the paths.

```python
# M0.2 — Monitor Initialization
import sys, os

def auto_path_setup():
    """Locate the src/core folder in different environments"""
    possible_paths = [
        os.path.abspath("."),
        os.path.abspath("../src/core"),
        "/content/fire_monitor/mapbiomas_fire_monitor/version_01/src/core",
    ]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "M0_auth_config.py")):
            if p not in sys.path: sys.path.insert(0, p)
            return p
    return None

found_path = auto_path_setup()
COUNTRY = "peru"

from M0_auth_config import set_country, authenticate, set_global_opts

set_country('peru')
set_global_opts(
    sensor='sentinel2',             # ['sentinel2', 'landsat', 'hls', 'modis']; list only for >1
    periodicity='monthly',          # 'monthly' or 'yearly'; list only for >1
    personal_task_flag='MONITOR',   # prefix for GEE task names (e.g. MONITOR, TEST)
    sampling_campaign='monitor_01', # campaign folder in LIBRARY_SAMPLES (GCS)
    clean_cache=False,              # True = reset local + GCS cache at startup
    language='en',                  # ['en', 'es', 'pt', 'fr', 'id']
    mosaic_methods=['minnbr', 'minnbr_buffer'],  # ['minnbr', 'minnbr_buffer', 'median', 'minndvi']
)
authenticate()
```

---

## Processing Flow (Sequential)

### [M1] — Export (GEE → GCS)
```python
from M1_export_ui import run_ui, start_export
ui_exporter = run_ui(years=[2025, 2026])
start_export(ui_exporter)
```

Exports satellite image collections (Sentinel-2, Landsat, MODIS, HLS) from GEE to GCS chunks with radiometric corrections and cloud masking.

### [M2] — Mosaic Assembly (COG)
```python
from M2_mosaic_ui import run_ui, start_mosaic_assembly
ui_assembler = run_ui(years=[2025, 2026])
start_mosaic_assembly(ui_assembler)
```

Assembles exported chunks into full-region Cloud-Optimized GeoTIFFs using GDAL VRT virtual stacking.

### [M3] — Sample Collection (GEE Toolkit)
```python
from M3_sample_ui import show_toolkit_links
show_toolkit_links()
```

Training data collection via the GEE JavaScript Toolkit. Draw fire/notFire polygons on satellite imagery.

### [M4] — DNN Training
```python
from M4_model_trainer import run_ui, start_training
ui_trainer = run_ui()
start_training(ui_trainer)
```

Trains a Deep Neural Network classifier using a flexible band extraction matrix with configurable architecture.

### [M5] — Regional Classification
```python
# 1. M5 Scheduling Interface
from M5_classifier_ui import run_m5_ui
ui_m5 = run_m5_ui(years=[2025, 2026], periodicity_active=["monthly"])

# 2. M5 Processing Engine
from M5_classifier import run_m5_workplan
# n_workers: auto (cpu_count). Force with: run_m5_workplan(n_workers=4, progress_callback=ui_m5._on_tile_progress)
run_m5_workplan(progress_callback=ui_m5._on_tile_progress)
```

Tile-based burned area classification producing int16 2-band COG rasters (probability 0-1000 + dayOfYear).

### [M6] — Mosaic, Stats & Publication
```python
# 1. M6 Dashboard Interface
from M6_ui import run_m6_ui
ui_m6 = run_m6_ui()

# 2. M6 Publish (mosaic + stats + GEE upload)
from M6_publisher import run_m6_publish
# upload_gee=True sends regional mosaics to Google Earth Engine
run_m6_publish(upload_gee=True)
```

Assembles classified M5 tiles into regional mosaics, computes burned area statistics, and publishes to GEE.

### [M7] — Post-Classification Masks 🔨 (Under development)
```python
# 🔨 M7 — Post-Classification Masks (Under development)
# Apply spatial and temporal masks to classified outputs.
# from M7_masks import run_m7_masks
```

### [M8] — Final Version Evaluation & Official Release 🔨 (Under development)
```python
# 🔨 M8 — Final Version Evaluation & Official Release (Under development)
# Evaluate candidate versions and select the official release.
# from M8_release import run_m8_release
```
