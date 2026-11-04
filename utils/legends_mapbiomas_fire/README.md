# MapBiomas Fire · Legends

Register of the MapBiomas Fire legends (pixel values, names in PT/EN/ES and HEX colors), with which ones are **active**, which are still **pending** and where each one comes from.

- **[LEGENDS.md](LEGENDS.md)**: all registered legends as tables (generated).
- **[preview.html](preview.html)**: the same legends with color swatches; download and open in a browser (generated).
- **[registry.csv](registry.csv)**: one row per legend: country, collection, product, status, years, GEE asset and source.
- **[legends/](legends/)**: one CSV per legend with the classes: `pixel_value, name_pt, name_en, name_es, hex`.

## Status

| status | meaning |
|---|---|
| `active` | official legend, in use |
| `review` | published, but something needs confirming before it is registered |
| `pending_colors` | pixel values are documented, colors are not |
| `deprecated` | replaced by a newer collection |
| `not_found` | the product exists, no legend document found |

## Add or change a legend

1. Create `legends/<country>_col<N>_<product>.csv` (copy an existing one). `hex` is 6 digits, lowercase (`#a900ff`); `pixel_value` is a whole number.
2. Add a row for it in `registry.csv` (`file` = the CSV name). When a new collection replaces an old one, set the old row to `deprecated`.
3. Run `python build_legends.py`. It checks the CSVs, lists any problem and rewrites `LEGENDS.md` and `preview.html`.
4. Commit the CSVs together with the generated files. `python -m unittest discover -s utils/legends_mapbiomas_fire/tests` checks that they are in sync.

Only the Python standard library is needed.

## What is registered today

| Country | Collection | Source | State |
|---|---|---|---|
| Brazil | 5 (5.1) | [Legend code PDF](https://brasil.mapbiomas.org/wp-content/uploads/sites/3/2026/09/Codigo-Legenda-MapBiomas-Fogo-Brazil-Col5.pdf) | 6 products with colors; annual burned area under `review` |
| Peru | 1 | [Legend code PDF](https://peru.mapbiomas.org/wp-content/uploads/sites/14/2025/10/CODIGO-DE-LEGENDA-FUEGO-COL-1-Peru.pdf) | 8 products, pixel values only (no colors published) |
| Chile | 1 | [news](https://uchile.cl/noticias/244097/mapbiomas-fuego-revela-mas-de-12-millones-de-hectareas-afectadas) | collection announced, no legend document found |

No fire legend was found for Bolivia, Colombia, Ecuador, Paraguay or Venezuela.
