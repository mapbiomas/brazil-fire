# MapBiomas Fire · Dashboard

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/dashboard/](https://mapbiomas.github.io/brazil-fire/dashboard/)**

> 🌎 **Eight countries:** Brazil, Argentina, Bolivia, Chile, Colombia, Paraguay, Peru and Indonesia. Pick the country and the language at the top of the page (or open a link such as `.../dashboard/#pais=peru&lang=en`). **Languages:** Portuguese, Spanish and English for every country, plus Bahasa Indonesia for Indonesia. Default: Portuguese for Brazil, Spanish for the South American countries, English for Indonesia. Territory and legend names come from the API in the chosen language when available.

Interactive dashboard of MapBiomas Fire, with the official logo of each country (Fogo, Fuego or Fire). It reads the statistics straight from the public MapBiomas Platform API, so there is nothing to build or install. Filter by biome (or ecoregion), first and second administrative level (state/municipality, department/district, region/commune…) and period, and explore the products in tabs: annual, accumulated, monthly, frequency, last fire year, scar size, severity, return interval, land tenure, territories rankings, maps and a GIF factory.

Brazil uses Collection 5.1 (1985–2025) and falls back to staging only if production fails. Other countries use the collection the Platform serves (read from the informative note). **Only the tabs a country has data for are shown:**

| Product | Countries |
|---|---|
| Annual, accumulated, monthly, frequency, territories, maps, GIF factory | All eight |
| Last fire year, scar size | All except Paraguay and Indonesia |
| Severity, return interval, land tenure categories | Brazil only |

Highlights:

- Charts can be switched between bars and lines, or bars and doughnut (small toggle at the top right of each card).
- Every card has **⬇ PNG** (chart with title, legend and source) and **▦ Table** (values on demand, downloadable as CSV).
- Monthly tab: choose the fire season (start and end month, e.g. July–June, labelled 2020/21) and the month palette (official legend or fire tones); burned area by year and month, stacked.
- Annual and accumulated burned area can be stacked **by biome** (Brazil: two earthy palettes; other countries: categorical colors per biome or ecoregion).
- Accumulated tab: share of each biome (or state/municipality) that burned at least once.
- Municipality × biome: the map and the statistics use only the part of the municipality inside the biome.
- Interactive maps: burned area, share of the territory, or one year compared with another year or with the historical mean, with a year timeline (play), click-to-outline and detail, CSV export and **PNG download (title, legend, scale bar and source)**; raw subproducts grouped as annual or by period, with an overlay, a two-year swipe for annual subproducts (both years selectable), clickable outlines and a highlight of the selected area.
- **GIF factory** tab: one frame per year from the official Platform map tiles, with title, territory, year, legend, source and logo (as in [`mapbiomas_gif_factory`](../mapbiomas_gif_factory/)), built in the browser and downloadable as GIF (or the last frame as PNG); **Generate and download GIF** saves it in one click. Optional boundaries on the frames (e.g. biomes or states over Brazil); speeds of 700, 350 or 180 ms per frame. Frame texts and logo follow the chosen language. Use the published site: opened as a local file, the browser may block reading the map images.

## 🗂️ Folder structure

```
utils/fire_dashboard/
├── README.md
├── CLAUDE.md          # technical documentation
└── app/
    ├── index.html
    ├── style.css
    ├── i18n.js        # translations (pt → es, en, id)
    ├── app.js
    └── logos/                        # copies of utils/_base/logos (PT, ES, EN; the page must open standalone)
```

| Item | Purpose |
|---|---|
| `app/index.html` | Page shell; loads Chart.js, Leaflet and Turf.js from cdnjs (the GIF encoder, gifenc, is loaded from jsDelivr on demand) |
| `app/style.css` | Styles (light/dark through CSS tokens) |
| `app/i18n.js` | Interface translations: one line per Portuguese text with its Spanish, English and Bahasa Indonesia versions |
| `app/app.js` | Country list (`PAISES`), API client, one loader per tab, charts, maps |

## ▶️ Run locally

1. Open `app/index.html` in a browser (internet access is required: API, map tiles and libraries).
2. The country, filters and active tab are kept in the URL hash, so a view can be shared as a link.

## ➕ Add a country

1. Check that the country has the fire theme on the Platform API (`/{region}/themes`).
2. Add an entry to `PAISES` in `app/app.js`: API region, name, language, logo, the territory categories used by the filters (regex on the category key: country, biome or ecoregion, first and second administrative levels) and their Portuguese labels.
3. Open `#pais=<region>` and go through the tabs. Subthemes are matched by name (pt, es, en) or analysis type; tabs without data stay hidden.
4. Missing today: official biome colors outside Brazil and land tenure categories outside Brazil. Translations were written without native-speaker review; please review `app/i18n.js` before official use.

## 🌐 Translations

The code is written in Portuguese and `app/i18n.js` maps each Portuguese text to Spanish, English and Bahasa Indonesia. The page translates whatever it shows (cards, charts, map tooltips, GIF frames). To fix a translation, edit its column; to translate a new text, add a line. Numbers follow the language format (`8,7 millones de ha`, `8.7 million ha`).

## 📚 Official data sources

- [MapBiomas Platform API](https://plataforma.mapbiomas.org/api/v1) (statistics, rankings, territories, geometries and map tiles; each country through the region in the path, e.g. `/peru/...`)
- Basemap: Esri World Light Gray Canvas
- [MapBiomas Fire](https://fogo.mapbiomas.org/)
