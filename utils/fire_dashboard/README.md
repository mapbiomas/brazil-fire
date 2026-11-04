# MapBiomas Fire · Dashboard

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/dashboard/](https://mapbiomas.github.io/brazil-fire/dashboard/)**

Interactive dashboard of MapBiomas Fire **Collection 5.1** (Brazil, 1985–2025). It reads the statistics straight from the public MapBiomas Platform API, so there is nothing to build or install. Filter by biome, state, municipality and period, and explore the products in tabs: annual, accumulated, monthly, frequency, last fire year, scar size, severity, return interval, land tenure, territories rankings and maps.

Highlights:

- Charts can be switched between bars and lines, or bars and doughnut (small toggle at the top right of each card).
- Annual and accumulated burned area can be stacked **by biome**, with two earthy palettes.
- Accumulated tab: share of each biome (or state/municipality) that burned at least once.
- Municipality × biome: the map and the statistics use only the part of the municipality inside the biome.

## 🗂️ Folder structure

```
utils/fire_dashboard/
├── README.md
├── CLAUDE.md          # technical documentation
└── app/
    ├── index.html
    ├── style.css
    └── app.js
```

| Item | Purpose |
|---|---|
| `app/index.html` | Page shell; loads Chart.js, Leaflet and Turf.js from cdnjs |
| `app/style.css` | Styles (light/dark through CSS tokens) |
| `app/app.js` | API client, one loader per tab, charts, maps |

## ▶️ Run locally

1. Open `app/index.html` in a browser (internet access is required: API, map tiles and libraries).
2. The filters and the active tab are kept in the URL hash, so a view can be shared as a link.

## ➕ Adapt it to another country or collection

The dashboard is **Brazil only and Portuguese only today** (no Spanish or English interface yet, and no other country is configured). To adapt it:

1. In `app/app.js`, change `CONFIG` (`region`, `colecaoExigida`, default years) and the `SUBTHEMES` / `LEGENDS` lookups if the country uses different subtheme names.
2. Biome colors (`BIOMA_CORES`, `PALETAS_BIOMA`) are the Brazilian ones; define the country's own.
3. Translate the interface texts (pt, en, es, as in `utils/_base/i18n/`) before publishing it for another country.

## 📚 Official data sources

- [MapBiomas Platform API](https://plataforma.mapbiomas.org/api/v1) (statistics, territories and geometries)
- [MapBiomas Fire](https://fogo.mapbiomas.org/)
