# Fuel Material – version 01

## 🔗 Open the Toolkit: **[explore_and_export (Google Earth Engine)](https://code.earthengine.google.com/?scriptPath=users%2Fworkspaceipam%2Fcombustible-material-accumulation%3Atoolkit%2Fexplore_and_export)**

Code, documentation and evaluation of version 1 of the MapBiomas Fire Fuel Material product: monthly NPV, GV
and soil fractions (0–100 %) at 30 m, from a minimum-NBR mosaic of NASA HLS v002 and an unconstrained linear
spectral unmixing with four endmembers (GV, NPV, soil, cloud). The method is described in the ATBD.

Developed by the MapBiomas Fire / IPAM team.

## 🗂️ Folder structure

```
version_01/
├── README.md
├── gee/
│   ├── fuel_to_asset.js          # exports the monthly assets (Earth Engine Code Editor)
│   └── explore_and_export.js     # the Toolkit: on-demand mosaics, maps, thumbnails, GeoTIFF export
├── atbd/
│   ├── ATBD_fuel_material_v3_pt.md   # ATBD v3 (Portuguese), with annexes A and B
│   └── figures/
├── evaluation/                   # Python scripts that produce the figures of ATBD section 3 and annex A
└── references/
    ├── README.md                 # full reference list with DOIs
    └── *.pdf                     # open-access papers only
```

| Item | Purpose |
|---|---|
| `gee/fuel_to_asset.js` | Monthly (and annual) export of the `npv`, `gv`, `soil` bands as Byte to `projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material/monthly_minnbr` (and `monthly_median`) |
| `gee/explore_and_export.js` | Public Toolkit: territorial and temporal filters, monthly assets and real-time mosaics (HLS, Sentinel-2 SR, Landsat 5–9, MODIS, VIIRS), export to Google Drive |
| `atbd/` | Algorithm Theoretical Basis Document, version 3 |
| `evaluation/` | Fuel response to fire: PARNA das Emas series, ICMBio prescribed burns, MapBiomas Fire detection, comparison with Landsat |
| `references/` | References cited in the ATBD |

The Earth Engine scripts in `gee/` are mirrors. The source of truth is the Earth Engine repository
`users/workspaceipam/combustible-material-accumulation` (`export_mosaic/v01/fuel_toAsset` and
`toolkit/explore_and_export`); update this folder when that code changes.

## ▶️ Run locally

**Earth Engine code.** Paste the files in `gee/` into the [Earth Engine Code Editor](https://code.earthengine.google.com/), or open the Toolkit link above.

**Evaluation scripts** (Python 3.10+):

1. `pip install earthengine-api pandas numpy matplotlib pillow requests` and authenticate with `earthengine authenticate`.
2. Set the environment variables:
   - `EE_PROJECT`: your Google Earth Engine project.
   - `ICMBIO_BURNS_PREFIX`: prefix of the yearly ICMBio burn-polygon assets (e.g. `.../AAF_ICMBIO_`). **Private data**: request access from ICMBio; without it, only the PARNA das Emas series (`serie_uc*.py`) runs.
   - `BIOMES_ASSET`: IBGE biome boundaries as a FeatureCollection with a `Bioma` property.
3. Run from `evaluation/`, in this order:
   - `python serie_uc.py --uc emas` then `python serie_uc_figuras.py` (PARNA das Emas series, Landsat comparison; ATBD figures 3, 4 and A2–A4)
   - `python biblioteca_exemplos.py` (31 example burns; figures 5–7)
   - `python historico_fogo.py` then `python historico_fogo_figuras.py` (MapBiomas Fire detection of the burns; figure A1)

Tables and figures are written to `version_01/tabelas/` and `version_01/figuras/` (not versioned). Comments in the
scripts are in Portuguese.

## 📚 Official data sources

- [MapBiomas Fire – Fuel Material monthly assets](https://code.earthengine.google.com/?asset=projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material/monthly_minnbr) (`projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material/monthly_minnbr`)
- [NASA HLS v002](https://lpdaac.usgs.gov/products/hlss30v002/) (`NASA/HLS/HLSS30/v002`, `NASA/HLS/HLSL30/v002` in Earth Engine)
- [MapBiomas Fire Collection 5.1 and Fire Monitor](https://brasil.mapbiomas.org/) (`projects/mapbiomas-public/assets/brazil/fire/...`)
- [MapBiomas Land Cover Collection 11](https://brasil.mapbiomas.org/) (`projects/mapbiomas-public/assets/brazil/lulc/collection11/...`)
- ICMBio burned-area polygons (AAF), 2019–2025 — not public

## ⚠️ Known gaps

- The ATBD is in Portuguese only; the English version is pending.
- Figures 1 (processing flow) and 2 (Pantanal map) of the ATBD are only in the published document.
