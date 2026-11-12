# MapBiomas Fire - Pipeline (M0-M9)

## Documentation and User Guide
Short guide to the pipeline. Run the sections in order: each module reads what the previous one saved.

### How it works (5 steps)

1. **Images (M1-M2):** GEE builds one mosaic per period (`minnbr`: the lowest-NBR observation, where a burn scar shows best) and exports it to GCS. M2 joins the pieces into Cloud-Optimized GeoTIFFs (COGs).
2. **Samples (M3):** in the GEE toolkit, draw `fire` and `notFire` polygons for each region and period.
3. **Training (M4):** a neural network (DNN) learns burned vs. unburned from the samples. Default: 5 hidden layers `7, 14, 7, 14, 7`, no dropout, 7000 iterations (same setup as the legacy collection).
4. **Classification (M5):** the model scans the region tile by tile and saves the burn probability and the day of the burn for every pixel.
5. **Publication and post-processing (M6 -> M7-M9):** M6 builds regional mosaics and statistics and uploads them to GEE. M7-M9 run in GEE: national merge, filters, statistics and promotion.

### Modules

| Module | Runs in | What it does | Output |
|---|---|---|---|
| **M0** | Colab | Setup, paths (`set_global_opts()`), authentication | config printed as `[CONFIG]`, `[GEE]`, `[GCS]` |
| **M1** | Colab -> GEE | Exports period mosaics | GCS `CHUNKS/` |
| **M2** | Colab | Assembles chunks into COGs | GCS `COG/` |
| **M3** | GEE toolkit | Training polygons | GEE + GCS `LIBRARY_SAMPLES/` |
| **M4** | Colab | Trains the DNN | GCS `LIBRARY_MODELS/{model}/` |
| **M5** | Colab | Classifies tiles | GCS `CLASSIFIED_TILES/` |
| **M6** | Colab | Regional mosaics, stats, GEE upload | GCS `CLASSIFIED_REGION/`, `STATS/`; GEE `REGIONAL/` |
| **M7** | GEE | ft00 national merge, ft01 land-use + small patches, ft02 temporal filter | GEE `FILTERED/` |
| **M8** | GEE | Burned-area statistics per filter stage | tables |
| **M9** | GEE | Promotion of the chosen version | pre-official collection |

### Classification output (M5)

Each tile is an int16 GeoTIFF with 2 bands (nodata `-9999`):

| Band | Name | Value |
|---|---|---|
| b1 | `probability` | burn probability x 1000 (0-1000; `50` = 5%), every valid pixel |
| b2 | `dayOfYear` | day of the year of the burn, where `probability > 0` |

Because the probability is kept, a stricter burned cut can be tested later in GEE without classifying again.

### Where things are saved

```
GCS {gcs_library_images_prefix}/LIBRARY_IMAGES/{SENSOR}/{PERIODICITY}/{MOSAIC}/{period}/
    CHUNKS/  (M1)    COG/  (M2)

GCS {gcs_campaigns_prefix}/{campaign}/
    LIBRARY_SAMPLES/                    M3  sample_{id}_{region}_{period}.csv
    LIBRARY_MODELS/{model}/             M4  weights, metadata, metrics
    LIBRARY_CLASSIFICATIONS/{model}/
        CLASSIFIED_TILES/               M5  tile_{region}_{cell}_{period}.tif
        CLASSIFIED_REGION/              M6  region_{region}_{model}_{period}.tif
        STATS/                          M6  stats_tile.csv, stats_region.csv

GEE {gee_campaigns_prefix}/{campaign}/
    AUXILIARY/                          M0  fire regions
    LIBRARY_SAMPLES/                    M3  sample polygons
    LIBRARY_CLASSIFICATIONS/REGIONAL/   M6  probability + dayOfYear per region
    LIBRARY_CLASSIFICATIONS/FILTERED/   M7  ft00, ft01, ft02
```

Model name: `training_{id}_{shortname}_{sensor}` (e.g. `training_0001_amazon_sentinel2`).

### Good to know

- Colab and local disk are temporary. Everything that matters is saved in GCS and GEE.
- Re-running is safe: modules skip what already exists in GCS. To classify again with a new model, train it under a new name (or delete the old tiles).
- In the selection lists, **Select remaining** marks only pending items; re-marking a green (done) item overwrites it.
- Access: GEE uses your own account; GCS uses the service account. If M0 prints `[GCS][WARN]`, ask to join the service-account group.

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

### [M7-M9] — Post-processing in Google Earth Engine
These steps are ready and run in the **GEE Code Editor**, not in this notebook. They read the regional images that M6 uploaded to GEE.

| Step | What it does | GEE script |
|---|---|---|
| **M7 ft00** | Picks one model per region and merges them into a national image | [ft00_selection_merge](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-MapBiomas-Fuego/M7/ft00_selection_merge) |
| **M7 ft01** | Land-use mask + removal of small isolated patches | `5-MapBiomas-Fuego/M7/{country}/{CAMPAIGN}/{monthly|yearly}/` (folder) |
| **M7 ft02** | Temporal filter (consistency with previous periods) | `5-MapBiomas-Fuego/M7/{country}/{CAMPAIGN}/{monthly|yearly}/` (folder) |
| **M8** | Burned-area statistics per filter stage and land use | [M8_statistics](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-MapBiomas-Fuego/M8/M8_statistics) |
| **M9** | Promotes the chosen version to pre-official | [M9_00_promote_ui](https://code.earthengine.google.com/?scriptPath=users/mapbiomasworkspace1/mapbiomas-fire:5-MapBiomas-Fuego/M9/M9_00_promote_ui) |

The links need access to the `mapbiomas-fire` GEE repository. Copies of the scripts are kept in `src/gee/` of this version.
