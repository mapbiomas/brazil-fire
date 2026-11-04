# utils/_base — shared standards

Single source of logos, UI strings (i18n), documentation templates and coding rules for the tools in `utils/`. For the Brazil team and the countries of the MapBiomas Fire network.

**Rule:** do not redefine logos or texts inside a tool. Point to this folder.

## Structure

```
utils/_base/
├── README.md
├── logos/          # MapBiomas Fire and Monitor logos (PT/EN/ES), variacoes/ (empty)
├── palette/        # empty for now (palettes to be defined)
├── i18n/           # pt-BR.json, en.json, es.json (same keys)
├── templates/      # README, CLAUDE and CHANGELOG templates for new tools
└── docs/           # code standards, commit policy, output naming (Portuguese)
```

## How to use

| I need… | Use |
|---|---|
| a logo | `logos/` (rules in `logos/README.md`) |
| a fire class color | CSVs in `utils/fire_legends/legends/` |
| an interface text | `i18n/<language>.json`, key under `strings` |
| a new tool | copy the files in `templates/` |

## Status

Phase 1 done. `fire_legends` and `fire_monitor_highlights` follow these standards (README format, folder naming, logos from here). `mapbiomas_gif_factory` has not been migrated yet. `palette/` is empty on purpose.
