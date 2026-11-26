# MapBiomas Fire · Legends

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/legends/](https://mapbiomas.github.io/brazil-fire/legends/)**

Consult the legends and palettes of every MapBiomas Fire collection published so far (per country, collection and product) and export them as GEE code, CSV, JSON, SVG/PNG or HEX lists (with pixel values, or comma-separated).

| Country | Published collections |
|---|---|
| 🇧🇷 Brazil | **5.1** · Monitor |
| 🇦🇷 Argentina | 1 |
| 🇧🇴 Bolivia | 1 |
| 🇨🇱 Chile | 1 |
| 🇨🇴 Colombia | 1 |
| 🇵🇾 Paraguay | 1 |
| 🇵🇪 Peru | 2 · Monitor |
| 🇮🇩 Indonesia | 1 · 2 · Monitor |

Colors marked **validada** in the app are the ones registered in this folder (`registry.csv` with status `active`). For the other collections the app shows a generated suggestion until their colors are registered.

## Folder structure

```
utils/fire_legends/
├── README.md
├── registry.csv          # one row per legend: status, years, GEE asset
├── countries.csv         # countries on the map tab and their status
├── legends/              # one CSV per legend: pixel_value, name_pt/en/es, hex
├── LEGENDS.md            # all registered legends as tables (generated)
├── preview.html          # the same with color swatches (generated)
├── build_legends.py      # validates the CSVs and regenerates the files above
├── app/
│   ├── index.html        # the web app
│   ├── palettes.json     # registered palettes for the app (generated)
│   └── countries.json    # countries for the map tab (generated)
└── tests/
```

## Status of a legend (`registry.csv`)

| status | meaning |
|---|---|
| `active` | official legend, in use |
| `review` | published, but something needs confirming before it is registered |
| `pending_colors` | pixel values are documented, colors are not |
| `deprecated` | replaced by a newer collection |
| `not_found` | the product exists, no legend document found |

## Add or change a legend

1. Create `legends/<country>_col<N>_<product>.csv` (copy an existing one). `hex` is 6 digits, lowercase (`#a900ff`); `pixel_value` is a whole number.
2. Add a row in `registry.csv`: `file` = the CSV name and `app_country`, `app_collection`, `app_product` = the codes the app uses (for example `brasil`, `5`, `fire_frequency`).
3. Run `python build_legends.py`. It checks the CSVs, lists any problem and rewrites `LEGENDS.md`, `preview.html` and `app/palettes.json`.
4. Commit the CSVs with the generated files. `python -m unittest discover -s utils/fire_legends/tests` checks they are in sync. The app is republished automatically.

Only the Python standard library is needed.

## Run locally

- Validate: `python build_legends.py --check` (from this folder).
- App: open `app/index.html` through a local server, for example `python -m http.server` inside `app/` (it loads `palettes.json`).

## Countries map (tab "Mapa")

The map tab colors each country by its status in `countries.csv`:

| status | on the map |
|---|---|
| `published` | Collection published (dark red) |
| `upcoming` | To be published (red) |
| `pilot` | Pilot / training (light red) |

To add a country, add a row with `country` (the code the app uses, e.g. `venezuela`), `iso_n3` (3-digit ISO numeric code, e.g. `862`), `iso_a2` (2-letter ISO code, e.g. `VE`, used for the short labels of the downloaded map) and the names in PT/EN/ES, and `monitor` (`published`, `upcoming` or empty) for the Fire Monitor view of the map, then run `python build_legends.py` (it writes `app/countries.json`). Clicking a country shows its collections, taken from `PUBLISHED` in `app/index.html`.

## Other countries

Legends are stored per country, collection and product. To add one, follow the steps above; leave `status` as `pending_colors` or `not_found` when colors or documents are missing, instead of copying Brazil's values. Shared logos and texts: [`utils/_base/`](../_base/). Technical notes: [CLAUDE.md](CLAUDE.md).
