# MapBiomas Fire – Fuel Material

## 🔗 Open the Toolkit: **[explore_and_export (Google Earth Engine)](https://code.earthengine.google.com/?scriptPath=users%2Fworkspaceipam%2Fcombustible-material-accumulation%3Atoolkit%2Fexplore_and_export)**

Monthly maps of surface fuel condition for the Brazilian biomes at 30 m. Each pixel holds the sub-pixel
fractions of non-photosynthetic vegetation (NPV, dry fine fuel), green vegetation (GV) and soil, obtained by
linear spectral mixture analysis of a minimum-NBR monthly mosaic of NASA HLS imagery. The product supports
integrated fire management: planning prescribed burns, firebreaks and pre-positioning of resources.

Developed by the MapBiomas Fire / IPAM team.

## 🗂️ Folder structure

```
mapbiomas_fire_monitor/fuel_material/
└── version_01/   # assets *_v1 (property version = '1'): GEE code, ATBD, evaluation scripts, references
```

| Folder | What it is |
|---|---|
| [`version_01`](./version_01/) | Current product: export and Toolkit code, Algorithm Theoretical Basis Document (ATBD v3) and the scripts behind its evaluation |

Product folders follow `<product>/version_NN`, where `NN` matches the `version` property of the published assets.
