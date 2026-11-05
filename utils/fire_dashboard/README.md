# MapBiomas Fire · Dashboard (Brazil)

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/dashboard/](https://mapbiomas.github.io/brazil-fire/dashboard/)**

> 🇧🇷 **Brazil only**, except the **GIF factory** tab. The statistics tabs cover Brazilian data (Collection 5.1, 1985–2025) and the interface is in Portuguese. The GIF factory tab also works for Argentina, Bolivia, Chile, Colombia, Paraguay, Peru and Indonesia. See [Adapt it to another country](#️-adapt-it-to-another-country-or-collection).

Interactive dashboard of MapBiomas Fire **Collection 5.1** for Brazil (1985–2025), with the MapBiomas Fogo logo. It reads the statistics straight from the public MapBiomas Platform API, so there is nothing to build or install. Filter by biome, state, municipality and period, and explore the products in tabs: annual, accumulated, monthly, frequency, last fire year, scar size, severity, return interval, land tenure, territories rankings and maps.

Highlights:

- Charts can be switched between bars and lines, or bars and doughnut (small toggle at the top right of each card).
- Annual and accumulated burned area can be stacked **by biome**, with two earthy palettes.
- Accumulated tab: share of each biome (or state/municipality) that burned at least once.
- Municipality × biome: the map and the statistics use only the part of the municipality inside the biome.
- Interactive maps: burned area, share of the territory or change between two periods, with a year timeline (play), bubbles, click-to-detail and CSV export; raw tiles with an overlay layer, a two-year swipe and a highlight of the selected area.
- **GIF factory** tab: one frame per year from the official Platform map tiles, with title, territory, year, legend, source and logo (as in [`mapbiomas_gif_factory`](../mapbiomas_gif_factory/)), built in the browser and downloadable as GIF (or the last frame as PNG). Pick the country at the top of the tab; frame texts follow the country language (pt, es or en). Use the published site: opened as a local file, the browser may block reading the map images.

## 🗂️ Folder structure

```
utils/fire_dashboard/
├── README.md
├── CLAUDE.md          # technical documentation
└── app/
    ├── index.html
    ├── style.css
    ├── app.js
    └── logos/                        # copies of utils/_base/logos (PT, ES, EN; the page must open standalone)
```

| Item | Purpose |
|---|---|
| `app/index.html` | Page shell; loads Chart.js, Leaflet and Turf.js from cdnjs (the GIF encoder, gifenc, is loaded from jsDelivr on demand) |
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

- [MapBiomas Platform API](https://plataforma.mapbiomas.org/api/v1) (statistics, territories, geometries and map tiles; other countries through the region in the path, e.g. `/peru/...`)
- Basemap: Esri World Light Gray Canvas
- [MapBiomas Fire](https://fogo.mapbiomas.org/)
