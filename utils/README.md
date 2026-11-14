# MapBiomas Fire · Utils

Supporting tools for the MapBiomas Fire network. Each folder is independent and has its own README.

| Tool | What it does | Online |
|---|---|---|
| [`fire_monitor_highlights/`](fire_monitor_highlights/) | Monthly Fire Monitor bulletin (Brazil, Peru), printable as PDF | [Brazil](https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/brazil/) · [Peru](https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/peru/) |
| [`mapbiomas_fire_export_and_vectorization/`](mapbiomas_fire_export_and_vectorization/) | MapBiomas Fogo: export, mosaic, publish and vectorization pipeline (Colab) | [Colab](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/utils/mapbiomas_fire_export_and_vectorization/mapbiomas_fire_export_and_vectorization.ipynb) |
| [`fire_legends/`](fire_legends/) | Registry and web app of MapBiomas Fire legends and palettes | [App](https://mapbiomas.github.io/brazil-fire/legends/) |
| [`fire_dashboard/`](fire_dashboard/) | Interactive dashboard of MapBiomas Fire for eight countries (Brazil, Argentina, Bolivia, Chile, Colombia, Paraguay, Peru, Indonesia; interface in Portuguese, Spanish, English and Bahasa Indonesia), filters by biome and administrative levels, maps and GIF factory | [App](https://mapbiomas.github.io/brazil-fire/dashboard/) |
| [`mapbiomas_gif_factory/`](mapbiomas_gif_factory/) | Cartographic GIFs from Earth Engine images (Colab pilot) | — |
| [`_base/`](_base/) | Shared logos, UI strings, templates and standards | — |

## Conventions

- Folder and file names: English, `snake_case`, tools named `fire_<what>`.
- READMEs are always in English.
- Shared assets come from `_base/`; start new tools from `_base/templates/`.
- No commit or push without explicit authorization (`_base/docs/politica_commit_push.md`).
