# Repository Guidelines

## Project Structure & Module Organization

- `access_tools/` — scripts and interfaces that help users access and interact with data (e.g. `qgis/`).
- `network/` — generalized algorithms and structures to share the MapBiomas Fire methodology with other countries (`network/catalog/`: campaigns, Colab notebooks, shared `src/core`).
- `mapbiomas_fire_collections/` — legacy and active processing for Brazilian fire collections (`collection_04`, `collection_05`).
- `mapbiomas_fire_monitor/` — continuous fire monitoring (`version_01`, `version_02`).
- `utils/` — supporting tools (`fire_monitor_highlights`, `fire_legends`, `mapbiomas_gif_factory`); see `utils/README.md`. Shared standards (logos, i18n strings, templates, docs) live in `utils/_base/`; see `utils/_base/README.md`.
- Root files: `README.md`, `GUIDE_CREDENTIALS_SETUP.md`, `PLANNING_COLLECTION_04.md`, `requirements.txt`.

## Build, Test, and Development Commands

There is no single build system. Each folder runs on its own (Google Earth Engine JavaScript, Python, Colab notebooks, static HTML). Run commands from the repository root.

- Catalog tests: `python -I -m unittest discover -s network/catalog/tests`
- Legends registry: `python utils/fire_legends/build_legends.py` (validate and regenerate), tests with `python -m unittest discover -s utils/fire_legends/tests`
- `utils/fire_monitor_highlights`: static HTML, open `src/<country>/*.html` in a browser (no build).
- `utils/mapbiomas_gif_factory`: see its `README.md` (Colab notebook, `pyproject.toml`).

## Coding Style & Naming Conventions

- Tool, folder and file names: English, `snake_case`. Tools in `utils/` are named `fire_<what>` (e.g. `fire_legends`).
- **READMEs are always written in English.** Technical notes (`CLAUDE.md`) and comments may be Portuguese or English, as described in `utils/_base/docs/padroes_de_codigo.md`.
- Prefer the Python standard library for small tools; keep notebooks thin and put logic in `src/`.
- Do not redefine shared assets (logos, UI strings, fire legends): take them from `utils/_base/` and `utils/fire_legends/`.

## Working for Other Countries (MapBiomas Fire network)

This repository is the hub other countries copy and adapt, so:

- **No Brazil-only hardcoding.** Country, language, API URL, GEE project and asset paths are parameters or files per country, never buried in shared code.
- **One folder or file per country** when APIs, data or language differ (see `utils/fire_monitor_highlights/src/<country>/`, `network/catalog/**/campaigns/<country>.*.json`). To add a country, copy the closest one and adapt.
- **Three UI languages: Portuguese, English, Spanish** (`pt`/`pt-BR`, `en`, `es`), aligned with `M_lang.py`. New user-facing text needs all three.
- **Mirror trees:** `network/catalog/development/version_03` and `production` are kept in sync; change both.
- **Keep it reproducible without private access:** document data sources and public URLs; mark anything that needs credentials or private assets.
- **Say what is missing for a country** (no data, no colors, no legend) in the docs instead of silently using Brazil's values.
- Review text written by non-native speakers before publishing it for another country.

## Documentation Guidelines

- `README.md` of each tool (English): public link first, short description, folder structure, how to run, how to add an item, data sources. Template: `utils/_base/templates/README_template.md`.
- `CLAUDE.md` (when present): technical decisions for developers and agents. Do not duplicate content between the two.

## Commit & Pull Request Guidelines

- Do not commit or push without explicit authorization from the repository owner for that change. Policy: `utils/_base/docs/politica_commit_push.md`.
- Keep commits small and faithful to the original files; describe what changed and why.

## Security & Configuration Tips

- Never commit credentials, tokens or private asset IDs.
- `GUIDE_CREDENTIALS_SETUP.md` contains real personal data and must be reviewed separately before it is kept public.
