# MapBiomas GIF Factory — pilot

A prototype for producing map GIFs from multiband Earth Engine images in Google Colab. The initial catalog includes experimental fire products for Brazil and Peru; assets, permissions, legends, and results must be checked before production use.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/utils/mapbiomas_gif_factory/notebooks/piloto.ipynb)

## How to use

1. Open the notebook above. In the first cell, `GEE_PROJECT` defaults to `mapbiomas-brazil`; change it to an Earth Engine project you can access if necessary. The cell clones the repository on the first run and, on subsequent runs, fetches and checks out `REPO_REF`, reinstalls the package, and clears cached modules, so updates are available without restarting the runtime.
2. Run the cells in order. Edit `LAYERS` and `TERRITORIES` to add custom items; multiband image bands are discovered automatically.
3. In the interface, choose an initiative and navigate the **Mapas** (Maps), **Territórios** (Territories), **Saídas e destino** (Outputs and destination), and **Resultados** (Results) tabs. Under Mapas, filter by theme and collection and use the buttons to select products. A batch never mixes initiatives: changing initiatives clears the selections.
4. Select at least one product, one territory, and the file types. Whole countries and individual biomes are available for Brazil and Peru. **Custom territories are available in every initiative**, allowing an existing product to run over a boundary not included in the presets. The **Personalizado** (Custom) initiative shows only your layers, with all territory groups available.
5. Choose whether the main library is local or, by enabling automatic upload, on Drive. Use **Verificar disponibilidade** (Check availability) to check the selected combinations, then click **Gerar GIFs** (Generate GIFs). Alternatively, run the `ui.start_batch()` cell.
6. Complete, compatible results already in the main library will not be reprocessed. If there is a conflict, the process warns you and does not overwrite files. To upload local files later, select them in the upload section; the required metadata is copied with them.

The **Mapas** tab brings together presets and reference records by theme and collection. The **Agendados** (Queued) list remains visible across tabs and summarizes the maps, territories, and number of combinations to be submitted when the batch runs. Selection does not replace validation of the asset, bands, processing, and legend in Earth Engine.

Storage under `/content` disappears when the Colab session ends. Temporary data is not automatically uploaded to Drive.

## Folders and availability

The library persists for the duration of the local session, or on Drive if Drive is selected as the destination. There is no separate delivery folder per run:

```text
mapbiomas_gif_factory/
├── Brasil_Fogo_Colecao_5/Area_queimada_anual/Amazonia/
│   ├── GIFs/
│   ├── Colagens/
│   ├── Frames/
│   └── metadados/
└── Peru_Fuego_Coleccion_2/Area_quemada_anual/Peru/
```

Intermediate files remain visible under `/content/temp/<iniciativa_tema_colecao>/<produto>/<territorio>/<período>/<execução>/processamento/`. They remain in the session for inspection, including after failures; the project does not automatically delete old folders.

- **Disponível (Available):** the selected output types have compatible metadata and all registered files check out.
- **Parcial (Partial):** compatible metadata exists, but a file or output type is missing; regeneration in the temporary folder may be necessary. Existing files are not replaced.
- **Não gerado (Not generated):** no result is registered for the selection.
- **Verificar (Check):** metadata is missing or invalid, files have changed, or the configuration differs. There is no automatic overwrite; inspect the destination folder.

With automatic upload enabled, Drive is the reference. Deleting a file on Drive makes the next check report a partial result, even if a local copy exists. With automatic upload disabled, the local library is the reference. A later selective upload carries the metadata and may appear partial until all files of a type have been copied.

A complete series uses `completo` in the metadata and has no extra period in the GIF/collage filename. A continuous partial range adds, for example, `_2020-2023`; an irregular selection receives an identifier, and the exact band names are stored in the metadata. Partial frames are placed in their own subfolder. The check verifies configuration, band list, file size, and SHA-256; it does not detect changes to Earth Engine asset content if the address and band names remain the same.

## Custom layers and territories

The `LAYERS` and `TERRITORIES` cells are explicit Python lists. Edit them and run the interface again. Layers in `LAYERS` appear in the **Personalizado** initiative and are saved under `Personalizado_<tema>_<coleção>/...`; presets remain under Brazil and Peru. Territories in `TERRITORIES` appear in the **Personalizados** (Custom) group and can be combined with products from any initiative.

For a multiband `ee.Image` layer:

```python
LAYERS = [{
    "id": "minha_camada",  # unique identifier: letters, numbers, _ or -
    "name": "Minha camada",
    "theme": "Meu tema",             # optional
    "collection": "Minha coleção",   # optional
    "asset": "projects/.../minha_imagem",
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"},
               {"label": "Fogo", "color": "800000"}],
    # "start_year": 2020,  # optional; omit both for all bands
    # "end_year": 2023,
}]
```

`id`, `name`, `asset`, `visualization`, and `legend` are required. `bands` is optional for explicit selection. Legend colors must match the palette; for nonconsecutive classes, do not assume that a short palette correctly represents the values. The Fire 4 and MATOPIBA examples are experimental.

### Sequence axis: time or depth

By default, the animation runs through the bands' years. Soil products organized by depth layer use a different axis, with `axis` and `depth_band`:

```python
LAYERS = [{
    "id": "argila_col3",
    "name": "Fração de argila",
    "theme": "Solo",
    "collection": "Coleção 3",
    "asset": "projects/.../mapbiomas_brazil_collection3_soil_clay_fraction_v1",
    "axis": "depth",          # "time" (default) or "depth"
    "depth_band": "clay_fraction",
    "visualization": {"min": 0, "max": 100, "palette": ["ffffcc", "800026"]},
    "legend": [{"label": "0", "color": "ffffcc"}, {"label": "100", "color": "800026"}],
}]
```

For this axis, the factory reads the `depth` property of each image in the collection, builds the sequence, and labels frames such as `0–10 cm`. `start_year`/`end_year` are rejected, and collages and special annual GIFs are disabled. Catalog depth products are disabled: each variable's legend and visualization still need validation.

For cumulative series such as `fire_frequency_1985_1999`, the frame label uses the final year when all windows begin in the same year; otherwise, the full range is retained.

A territory is an `ee.FeatureCollection`. The optional filter uses `ee.Filter.inList`:

```python
TERRITORIES = [{
    "id": "meus_biomas",
    "name": "Cerrado e Caatinga",
    "asset": "projects/ee-ipam/assets/IBGE/limite_biomas_IBGE_2026",
    "filter": {"field": "Bioma", "values": ["Cerrado", "Caatinga"]},
}]
```

For a single value, you can use `"value": "Cerrado"` instead of `values`. Omit `filter` to use the entire collection. Confirm access to the asset and the attribute values. The project does not yet automatically check whether the territory intersects the data.

In the **Territórios** tab, type a municipality name to see results in real time; select and add one or more. The search uses FAO GAUL 2025, not the official IBGE boundaries. Conservation unit (UC) and Indigenous territory (TI) options appear in the same search, but their sources are still awaiting connection and validation. The search is case-sensitive and returns up to 30 results. Features from the same source can be combined into one FeatureCollection; internal boundaries are not dissolved. The account must have access to the asset.

The catalog contains presets and a reference inventory, grouped by theme and collection. Unverified non-fire reference records remain disabled rather than inheriting a style by name; presets remain experimental. When a selectable layer is chosen, the **Ajustar paleta e legenda** (Adjust palette and legend) panel allows colors and labels to be reviewed. An available suggestion does not prove correctness or successful execution.

Reference fire legends use a pinned snapshot of `utils/fire_legends`, matched by initiative, collection, product, and asset. Only consecutive, multi-class palettes can currently be represented without inferring pixel values; other reference entries remain disabled. Legacy presets have their own experimental styles and are not proof that a legend is validated. `build_vera_fire_legends.py` regenerates the snapshot from the pinned CSV sources in `fire_legends_source/`. The provisional non-fire inventory in `validation/theme_legends/` records source styles without asserting a product, asset, or processor match.

## Limitations and next steps

- The adapter supports multiband `ee.Image`, simple temporal collections, and known Soil depth structures. Collections with multiple multiband variables allow variable selection; duplicate periods, collections with more than 500 images, mosaics, RGB, and products requiring specific processors still need adaptation. Peru, Fire frequency, and Soil are experimental and should be checked in Colab.
- Bands with an identifiable year or range are ordered chronologically; ambiguous selections are stopped rather than producing an assumed sequence.
- The library does not automatically import results generated by the previous version; old files without compatible metadata appear as `Verificar` if they occupy the new path.
- The time estimate covers only the current download; the batch progress bar counts processed combinations, including failures.
- The availability check consults metadata and files; it does not check whether the territory intersects the data coverage.
- The frame layout uses fixed-size elements: very small output dimensions or unusually narrow territories can produce clipped or overlapping text. Inspect output before use; the notebook uses the default 2000-pixel dimension.
- Planned extensions include verified land use and land cover 10/10.1 legends, products with their own processors, and visual validation of Soil and Atmosphere collections.

## Development

`pip install -e .` and `python -m unittest discover -s tests`. Visual and end-to-end validation requires Earth Engine, asset permissions, Google Drive, and a Colab session.
