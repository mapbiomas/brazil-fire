# Repository Guidelines

## Project Structure & Module Organization

- `access_tools/` — scripts and interfaces that help users access and interact with data (e.g. `qgis/`).
- `network/` — generalized algorithms and structures to share the MapBiomas Fire methodology with other countries (`network/catalog/`: campaigns, Colab notebooks, shared `src/core`).
- `mapbiomas_fire_collections/` — legacy and active processing for Brazilian fire collections (`collection_04`, `collection_05`).
- `mapbiomas_fire_monitor/` — continuous fire monitoring (`version_01`, `version_02`).
- `utils/` — supporting tools (`fire_monitor_highlights`, `legends_mapbiomas_fire`, `mapbiomas_gif_factory`). Shared standards (logos, color palettes, i18n strings, templates) live in `utils/_base/`; see `utils/_base/README.md`.
- Root files: `README.md`, `GUIDE_CREDENTIALS_SETUP.md`, `PLANNING_COLLECTION_04.md`, `requirements.txt`.

## Build, Test, and Development Commands

There is no single build system. Each folder runs on its own (Google Earth Engine JavaScript, Python, Colab notebooks, static HTML). Run commands from the repository root.

- Catalog tests: `python -I -m unittest discover -s network/catalog/tests`
- Legends registry: `python utils/legends_mapbiomas_fire/build_legends.py` (validate and regenerate), tests with `python -m unittest discover -s utils/legends_mapbiomas_fire/tests`
- `utils/fire_monitor_highlights`: static HTML, open `src/<country>/*.html` in a browser (no build).
- `utils/mapbiomas_gif_factory`: see its `README.md` (Colab notebook, `pyproject.toml`).

## Coding Style & Naming Conventions

- Tool, folder and file names: English, `snake_case`.
- Comments and user documentation may be Portuguese or English, as described in `utils/_base/docs/padroes_de_codigo.md`.
- Prefer the Python standard library for small tools; keep notebooks thin and put logic in `src/`.
- Do not redefine shared assets (logos, LULC colors, fire legends, UI strings): import them from `utils/_base/`.

## Documentation Guidelines

- `README.md` of each tool: how to use it (for users). `CLAUDE.md` (when present): technical decisions (for developers and agents). Do not duplicate content between them.
- Use the templates in `utils/_base/templates/` for new tools.

## Commit & Pull Request Guidelines

- Do not commit or push without explicit authorization from the repository owner for that change. Policy: `utils/_base/docs/politica_commit_push.md`.
- Keep commits small and faithful to the original files; describe what changed and why.

## Security & Configuration Tips

- Never commit credentials, tokens or private asset IDs.
- `GUIDE_CREDENTIALS_SETUP.md` contains real personal data and must be reviewed separately before it is kept public.
