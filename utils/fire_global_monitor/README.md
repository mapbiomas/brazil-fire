# MapBiomas Fire · Global fire monitor

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/global_analises/](https://mapbiomas.github.io/brazil-fire/global_analises/)**

> Exploratory, unofficial page. Languages: Portuguese, English, Spanish, French, Bahasa Indonesia and Chinese (selector at the top, or `?lang=en`).

Monthly ranking of burned area, active fires (hotspots) and fire emissions for the whole world, by continent, country and state/province, with maps, monthly and annual charts, seasonality and **fire frequency** (area that burned at least once in 2001–Y, by number of years burned). Built for quick global comparisons, not as an official product.

Highlights:

- Levels: continents, countries and states/provinces; filters by continent and country; the map zooms to the selection.
- Indicators: MCD64A1 burned area (also as % of territory), VIIRS burned area, active fires, CO₂, CH₄ and PM2.5 emissions; burned area split by land cover.
- Every card has **CSV** and **PNG** downloads (PNG with title and source).
- Fire frequency card: ranking by % of territory or area burned at least once, stacked by frequency class (1, 2, 3–5, 6–10, 11+ years), and a curve of accumulated area for each period 2001–Y.

## 🗂️ Folder structure

```
utils/fire_global_monitor/
├── README.md
├── CLAUDE.md                 # technical notes
├── app/                      # published site (GitHub Pages: /global_analises/)
│   ├── index.html            # single-page app, no build, no external libraries
│   └── data/                 # CSV/JSON read by the page
├── scripts/                  # data pipeline (GWIS files + Google Earth Engine)
├── i18n/                     # translation tables (pt → en, es, fr, ind, zh)
└── raw_gwis/                 # GWIS downloads (git-ignored)
```

| Item | Purpose |
|---|---|
| `app/data/adm0_monthly.csv` | Monthly series per country (GWIS 2002–2025 + GEE after the end of GWIS) |
| `app/data/adm1/<year>.csv` | States/provinces (GADM level 1, GWIS), one file per year, loaded on demand |
| `app/data/freq_adm0.csv`, `freq_adm1.csv` | Fire frequency per period 2001–Y (GEE) |
| `app/data/world_110m.json`, `adm1_geo.json` | Simplified boundaries for the maps (Natural Earth, geoBoundaries) |
| `app/data/continents.json`, `i18n.json`, `meta.json` | Country → continent, translations, data periods |
| `scripts/build_from_gwis.py` | GWIS ZIPs (+ GEE CSV) → `app/data/adm0_monthly.csv`, `adm1/*.csv`, `meta.json` |
| `scripts/gee_run_exports.py`, `fetch_assets_to_csv.py` | GEE: monthly hotspots and burned area per country, one task per month |
| `scripts/gee_run_frequency.py`, `fetch_frequency_to_csv.py` | GEE: fire frequency per period, one task per period |
| `scripts/build_adm1_geo.py` | State boundaries for the map, matched by name to the GWIS (GADM) units |
| `scripts/build_i18n.py` | `i18n/*.csv` → `app/data/i18n.json` |
| `scripts/ee_profiles.py` | Several Earth Engine logins on one machine (`--account`) |

## ▶️ Run locally

1. `cd utils/fire_global_monitor/app && python -m http.server 8000`, then open http://localhost:8000 (opening the file directly blocks the data requests).
2. To rebuild the data:
   1. Download from the [GWIS country profile](https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads) the MCD64A1, MVP64 and GFED ZIPs into `raw_gwis/`.
   2. `python scripts/gee_run_exports.py --account <profile> --level 0 --start 2024-01 --end 2026-09`, wait for the tasks, then `python scripts/fetch_assets_to_csv.py --account <profile>`.
   3. `python scripts/build_from_gwis.py`.
   4. Frequency: `python scripts/gee_run_frequency.py --account <profile> --level 0` (and `--level 1`), then `python scripts/fetch_frequency_to_csv.py --account <profile> --level 0` (and `--level 1`).
   5. Translations: `python scripts/build_i18n.py`.

Earth Engine tasks are named `WS_IPAM_GLOBAL_ANALISES-<file>`; the default Cloud project is `mapbiomas-brazil` (change with `--project`).

## ➕ Add a new item (language, indicator...)

1. **Language:** add a column to `i18n/traducoes.csv`, `extra.csv` and `paises.csv` (instructions in `i18n/INSTRUCOES_TRADUCAO.md`), add it to `LANGS` in `scripts/build_i18n.py` and to `LANG_UI`/`LOCALE` in `app/index.html`, then run `build_i18n.py`.
2. **Indicator from GWIS:** read it in `scripts/build_from_gwis.py`, add the column to `OUT_COLS` and an entry to `METRICS` in `app/index.html`.
3. **New GWIS year:** download the new ZIPs into `raw_gwis/` and run `build_from_gwis.py`.

## 📚 Official data sources

- [MODIS MCD64A1 v6.1 burned area](https://lpdaac.usgs.gov/products/mcd64a1v061/) (NASA/USGS), aggregated by the [GWIS country profile](https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads) (JRC / Copernicus) and in [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD64A1)
- [VIIRS VNP64](https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads) burned area and [GFED](https://www.globalfiredata.org/) emissions, via GWIS
- [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) active fires ([GEE `FIRMS`](https://developers.google.com/earth-engine/datasets/catalog/FIRMS))
- Boundaries: [Natural Earth](https://www.naturalearthdata.com/), [geoBoundaries](https://www.geoboundaries.org/) ([GEE](https://developers.google.com/earth-engine/datasets/catalog/WM_geoLab_geoBoundaries_600_ADM1)), GADM (GWIS units)
