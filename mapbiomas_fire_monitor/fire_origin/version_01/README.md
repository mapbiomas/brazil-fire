# MapBiomas Fire Monitor – Fire Trajectory and Origin (version_01)

## 🔗 Open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/mapbiomas_fire_monitor/fire_origin/version_01/mapbiomas_fire_origin.ipynb)

Monthly product that groups active-fire detections (hotspots) into fire events with a spatio-temporal DBSCAN, estimates how each event spread over time, and combines this with the monthly burned-area scars of the MapBiomas Fire Monitor. It outputs a 30 m trajectory map and the likely ignition (origin) areas of each fire.

> This folder holds the notebook **as it was used to produce the `_v1` assets** (outputs cleared, code unchanged). Improvements will be added in small, separate commits and listed under [Version history](#-version-history).

## 🗂️ Folder structure

```
mapbiomas_fire_monitor/fire_origin/version_01/
├── README.md
└── mapbiomas_fire_origin.ipynb   # whole pipeline in a single Colab notebook
```

## 🔥 Pipeline (notebook steps)

| Step | What it does | Runs on | Output |
|---|---|---|---|
| 1 | Exports monthly hotspots from GEE to CSV, standardizing the date/time field | GEE batch tasks | `gs://mapbiomas-fire/monitor/product_fire_origin/01_FOCOS_MENSAIS_ORIGINAL/focos_YYYY_MM.csv` |
| 2 | ST-DBSCAN per month, carrying the previous month to inherit `CLUSTER_ID`; computes `Lapsetime` and `R_lapsetime`; retroactively updates the 6 previous months | Colab (pandas, scikit-learn) | `.../02_st-dbscan-output_v1/DBSCAN_YYYY_MM.csv` |
| 2.1 | Ingests the clustered CSVs as GEE tables | GEE | `.../FOGO/MONITORAMENTO/focos-st-dbscan-output/DBSCAN_YYYY_MM` |
| 3 | Rasterizes clustered hotspots (0.001°), nearest-neighbour fill (15 px) and per-cluster linear interpolation of `R_lapsetime` | Colab (rasterio, SciPy) | `.../03_STFireOrigins-output_v1/STFIreOrigins_YYYY_MM.tif` |
| 3.1 | Ingests the GeoTIFFs into an ImageCollection | GEE | `st_fire_origins_raster30m_v1` |
| 4 | Combines trajectories with the monthly burned area, searching M, M-1, M+1 … ±3 months; exports 5 bands at 30 m | GEE batch tasks | `fire_monitor_trajetory_v1` |

### Final bands (step 4)

| Band | Meaning |
|---|---|
| `fire_relative_propagation_lapestime` | Relative time within the fire event, 0 (start) to 1 (end); `2` = burned area without an estimated trajectory |
| `fire_cluster_id` | Fire event ID (`0` = no cluster) |
| `fire_time_start` | Timestamp of the nearest hotspot, in ms (`0` = no information) |
| `fire_origin_zone` | `1` where 0 ≤ relative time ≤ 0.1 |
| `fire_origin_point` | `1` where relative time = 0 |

### Main parameters (cell "Definição de Parâmetros Globais")

| Parameter | Value in v1 |
|---|---|
| `pa_eps_spatial` | 0.02° (~2.2 km) |
| `pa_eps_temporal` | 48 h |
| `pa_min_samples` | 2 hotspots |
| Processing window | `export_start_year`–`export_end_year`, `export_start_month`–`export_end_month` |

## ▶️ How to run

1. Open the notebook with the Colab badge above.
2. Run the import and authentication cells (Earth Engine project `mapbiomas-fire-485203`; Google Cloud access to the `mapbiomas-fire` bucket is required).
3. Set the year/month window in the global parameters cell.
4. Run the steps in order. Steps 1, 2.1, 3.1 and 4 only **submit** GEE tasks: wait for them to finish in the [Task Manager](https://code.earthengine.google.com/tasks) before running the next step.

> ⚠️ The **last cell deletes every image of `fire_monitor_trajetory_v1`**. Do not use "Run all".

To save changes made in Colab back to this repository: **File → Save a copy in GitHub**.

## 🕒 Version history

- **v1.0.0 (2026-11-06)** – notebook added as used to produce the `_v1` assets.
- **v1.0.1 (2026-11-06)** – rewritten overview; parameters grouped (period first) with comments explaining the ST-DBSCAN limits and what counts as a fire event; unused imports removed and `rasterio` installed. No parameter values changed.

## 📚 Data sources

- Monthly hotspots (South America): `projects/mapbiomas-fire-485203/assets/DATABASE/monthly-focus-sul-america`
- Monthly burned area (MapBiomas Fire Monitor): `projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1`
- Biomes: IBGE (`projects/ee-ipam/assets/IBGE/IBGE_Biomas`)
- Access to the private assets and to the bucket requires MapBiomas Fire credentials.
