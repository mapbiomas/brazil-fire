# MapBiomas Fire · Legends

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/legends/](https://mapbiomas.github.io/brazil-fire/legends/)**

Consult the legends and palettes of every MapBiomas Fire collection published so far (per country, collection and product) and export them as GEE code, QGIS colormap, SLD, SVG/PNG or plain HEX lists.

| Country | Published collections |
|---|---|
| 🇧🇷 Brazil | 2 · 3 · 4 · 4.1 · **5** · Monitor |
| 🇦🇷 Argentina | 1 |
| 🇧🇴 Bolivia | 1 |
| 🇨🇱 Chile | 1 |
| 🇨🇴 Colombia | 1 |
| 🇵🇾 Paraguay | 1 |
| 🇵🇪 Peru | 1 · 2 · Monitor |
| 🇮🇩 Indonesia | 1 · 2 · Monitor |

_Inventory of 30/10/2026 (official site and `gs://mapbiomas-public`)._ Colors marked **validada** in the app come from the official legend documents registered in this folder (today: **Brazil, Collection 5**). For the other collections the app shows a generated suggestion until their colors are registered.

## Folder structure

```
utils/fire_legends/
├── README.md
├── registry.csv          # one row per legend: status, years, GEE asset, source
├── legends/              # one CSV per legend: pixel_value, name_pt/en/es, hex
├── LEGENDS.md            # all registered legends as tables (generated)
├── preview.html          # the same with color swatches (generated)
├── build_legends.py      # validates the CSVs and regenerates the files above
├── app/
│   ├── index.html        # the web app
│   └── palettes.json     # registered palettes for the app (generated)
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

## Other countries

Legends are stored per country, collection and product. To add one, follow the steps above; leave `status` as `pending_colors` or `not_found` when colors or documents are missing, instead of copying Brazil's values. Shared logos and texts: [`utils/_base/`](../_base/). Technical notes: [CLAUDE.md](CLAUDE.md).

## Sources

- Brazil Collection 5: [legend code PDF](https://brasil.mapbiomas.org/wp-content/uploads/sites/3/2026/09/Codigo-Legenda-MapBiomas-Fogo-Brazil-Col5.pdf)
- Peru Collection 1: [legend code PDF](https://peru.mapbiomas.org/wp-content/uploads/sites/14/2025/10/CODIGO-DE-LEGENDA-FUEGO-COL-1-Peru.pdf) (pixel values only, no colors published)