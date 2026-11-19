# MapBiomas GIF Factory — pilot

A prototype for producing map GIFs from multiband Earth Engine images in Google Colab.

## Install

```bash
pip install mapbiomas-gif-factory
```

In Colab or Jupyter, after `ee.Initialize(project=...)`:

```python
from gif_factory.app import show_app
show_app("my-earth-engine-project")
```

The package bundles the catalog, palettes and territory sources; nothing else needs to be downloaded. To test an unreleased branch:

```bash
pip install "git+https://github.com/mapbiomas/brazil-fire.git@<branch>#subdirectory=utils/mapbiomas_gif_factory"
```

## Catalog

The map catalog comes from `src/gif_factory/config/`, ported from the local project (`mapbiomas-ia/FABRICA DE GIFS/config`). Each product there is defined once: asset or processor, band slicing, and the visualization key (palette, legend, min/max) in `visualization.yaml`. Catalog products run through the same pipeline as the local project, so maps come out with the same design and colors.

- 138 products: Brasil (Fire 3 and 5, LULC 9 to 11, Degradation 9, 10.1 and 11, Soil 3, Atmosphere 1, Oeste Baiano), Peru (Fire 2), Paraguay (Fire 1) and DR Congo (LULC 1).
- Soil collection 3 includes products animated **by depth** instead of by year (`axis: depth`: clay, sand and silt fractions from 0–10 to 90–100 cm; textural group, subgroup and class from 0–10 to 60–100 cm), with the official MapBiomas Solo palettes copied to `soil_legends_source/` from `mapbiomas/brazil-soil`.
- Atmosphere collection 1 (annual precipitation, air and surface temperature, PM2.5, PM10, rain-free days, persistent rain, vapor pressure deficit, water availability) is **beta**: the data is official, but MapBiomas Atmosfera publishes no palettes, so the colors are proposed by the factory. Products that pick one image of a collection use `collection_filter: {property, value}`.
- `config/catalog_status.yaml` sets each product's status: **validado** (shown by default), **beta** (shown only when "Mostrar mapas beta" is ticked) or **pendente** (hidden). Without an entry, a product is validado.
- To add or fix a product, edit the collection YAML in `config/initiatives/<initiative>/collections/` and, if needed, add a style to `visualization.yaml`.
- `python validation/legend_check.py` compares the catalog's class styles with the official legends: land cover against the MapBiomas Platform API (`prd.plataforma.mapbiomas.org/api/v1`, which publishes pixel value, name, color and level per initiative) and fire products against `fire_legends_source/`. Known, intentional differences are listed in the script with their reason; anything else fails the check.
- `python validation/catalog_review.py` checks every product against Earth Engine (access, bands, processor) and writes a contact sheet with a thumbnail and legend per product to `validation/catalog_review/<UTC>/index.html`.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/utils/mapbiomas_gif_factory/notebooks/piloto.ipynb)

## How to use

The notebook has two form cells; nobody needs to edit code.

1. **1 · Conectar** — set `GEE_PROJECT` (defaults to `mapbiomas-brazil`) and whether to save to Google Drive, then run. It installs the package from PyPI (about 160 KB, instead of cloning this repository), authenticates Earth Engine and mounts Drive. `VERSAO` pins a release; `BRANCH_DE_TESTE` installs a GitHub branch for testing.
2. **2 · Abrir a Fábrica** — opens the interface (`gif_factory.app.show_app`), a 4-step flow with a summary bar that stays visible:
   1. **Mapa** — pick one or more maps from the catalog. Filter by initiative, theme and collection, or type part of the name (accents and case are ignored). The side panel shows the legend. Beta maps appear only with **Mostrar mapas beta**.
   2. **Território** — pick one or more territories, in any of four ways:
      - **Buscar**: official MapBiomas territory tables (`projects/mapbiomas-territories`, public): country, biome, region, state, municipality, indigenous land, protected area, quilombo, basin, Legal Amazon, Matopiba; and the equivalents for Peru, Paraguay and DR Congo (`config/territory_sources.yaml`). Several items can be merged into one area.
      - **Desenhar no mapa**: rectangles or polygons drawn on the map become one area.
      - **Enviar arquivo**: GeoJSON, KML, GeoPackage or a zipped shapefile.
      - **Asset do GEE**: any FeatureCollection the account can read, optionally filtered by a field.
      Every territory is outlined on the map.
   3. **Período e saídas** — all years or a year range (available years are read from the maps), file types (GIF, collage, yearly images; more under advanced options), Drive destination, and a quick preview of the first map over the first territory.
   4. **Gerar** — review the combinations, optionally check what already exists, generate, follow progress, and see the GIFs (with a zip download in Colab).
3. Results already complete in the library are not regenerated, and nothing is overwritten.

`show_app(project, layers=[...])` still accepts extra layers (shown under **Meus mapas**) for analysts who script; `gif_factory.ui.show_ui` remains as a compatibility wrapper.

Storage under `/content` disappears when the Colab session ends; with Drive enabled, results go to `MyDrive/MapBiomas_GIF_Factory`.

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

## Analyst mode: maps and territories outside the catalog

Nothing here requires editing the notebook.

**New map from an asset** (step 1, **Novo mapa a partir de um asset**):

1. Paste an image asset (one band per year) or an image collection (one image per year) and click **Ler asset**. The years are detected from the band names; depth sequences (Soil) are recognised too.
2. Give it a name, theme and collection (default `beta`).
3. Choose the colors:
   - **Estilo do catálogo**: reuse any of the ~60 catalog styles (palette, classes and legend), for example `fire`, `lulc_nivel1` or `fire_col5_frequency`;
   - **Minhas cores**: minimum, maximum, hex colors and optional one label per color. **Detectar valores** suggests the min/max from the last period.
4. **Ver prévia** shows the last period over the first chosen territory (or the asset's own area). **Salvar em Meus mapas** saves and selects it.

Saved maps appear under the **Meus mapas** initiative and run like catalog maps. They are stored in `MyDrive/MapBiomas_GIF_Factory/meus_itens.json` and return in later sessions (without Drive, only for the current session). **Proposta para o catálogo** produces the YAML to paste into `config/` so the map can become an official catalog product (start it as `beta` in `config/catalog_status.yaml`).

**My territories**: areas drawn on the map, uploaded files and pasted assets are saved automatically to the same file. Choose **Meus territórios** as the base in step 2 to reuse or delete them.

For scripting, `show_app(project, layers=[...], territories=[...])` still accepts layer and territory dictionaries (`id`, `name`, `asset`, `visualization`, `legend`; territories with `asset` and optional `filter`, or `geometry` as GeoJSON).

### Sequence axis: time or depth

By default, the animation runs through the bands' years. Soil collections organised by depth (`depth` property such as `000_010`) are detected and labelled `0–10 cm`; `start_year`/`end_year` do not apply, and collages and special annual GIFs are disabled. For cumulative series such as `fire_frequency_1985_1999`, the frame label uses the final year when all windows begin in the same year.

## Limitations and next steps

- The adapter supports multiband `ee.Image`, simple temporal collections, and known Soil depth structures. Collections with multiple multiband variables allow variable selection; duplicate periods, collections with more than 500 images, mosaics, RGB, and products requiring specific processors still need adaptation. Peru, Fire frequency, and Soil are experimental and should be checked in Colab.
- Bands with an identifiable year or range are ordered chronologically; ambiguous selections are stopped rather than producing an assumed sequence.
- The library does not automatically import results generated by the previous version; old files without compatible metadata appear as `Verificar` if they occupy the new path.
- The time estimate covers only the current download; the batch progress bar counts processed combinations, including failures.
- The availability check consults metadata and files; it does not check whether the territory intersects the data coverage.
- The frame layout uses fixed-size elements: very small output dimensions or unusually narrow territories can produce clipped or overlapping text. Inspect output before use; the notebook uses the default 2000-pixel dimension.
- Planned extensions include verified land use and land cover 10/10.1 legends, products with their own processors, and visual validation of Soil and Atmosphere collections.

## Releasing

1. Update `version` in `pyproject.toml` (for example `0.2.1`) and merge to `main`.
2. Push the tag `gif-factory-v0.2.1`. The workflow `.github/workflows/gif-factory-publish.yml` (repository root) runs the tests, builds the package, checks that the tag matches the version and publishes it to PyPI.
3. Running the workflow manually publishes to TestPyPI instead, as a rehearsal.

Publishing uses PyPI trusted publishing, so no token is stored in the repository. One-time setup on pypi.org (and test.pypi.org): *Publishing → Add a pending publisher* with owner `mapbiomas`, repository `brazil-fire`, workflow `gif-factory-publish.yml` and environment `pypi` (`testpypi`).

## Development

`pip install -e .` and `python -m unittest discover -s tests`. Visual and end-to-end validation requires Earth Engine, asset permissions, Google Drive, and a Colab session.
