# MapBiomas Fire Monitor

Scripts for the continuous monitoring of fire in Brazil: products derived from the monthly MapBiomas Fire Monitor and tools to export and distribute its data.

## 🗂️ Folder structure

```
mapbiomas_fire_monitor/
├── fire_origin/
│   └── version_01/   # fire trajectory and origin product (ST-DBSCAN), single Colab notebook
├── version_01/       # export & distribution tool (export_and_vectorization)
└── version_02/       # export & distribution engine (mapbiomas_cog package)
```

| Folder | What it is |
|---|---|
| [`fire_origin/version_01`](./fire_origin/version_01/) | Monthly fire trajectory and ignition-origin product (`_v1` assets) |
| [`version_01/export_and_vectorization`](./version_01/export_and_vectorization/) | Export, mosaic, publication and optional vectorization of MapBiomas catalog data |
| [`version_02`](./version_02/) | Tiled, resumable COG export engine that replaces the version_01 export |

Product folders follow `<product>/version_NN`, where `NN` matches the version suffix of the published assets (`version_01` → `_v1`).
