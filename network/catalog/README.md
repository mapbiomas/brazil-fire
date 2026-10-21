# MapBiomas Fire Catalog

Classification catalog for the MapBiomas Fire network. It evolves from the
Peru monitor pipeline (`peru-fire/mapbiomas_fire_monitor/version_01`) and
makes **campaign** and **target** first-class concepts.

- **Campaign** = where the data comes from (country + source + sensor + periodicity).
- **Target** = what is predicted (`burned_area` today; other profiles later).

The catalog never redefines the export layout. Each campaign declares its own
`gcs_root`/`gee_root`, and every path is derived from it, so the addresses used
by the export/vectorization pipeline stay exactly the same.

## Versions

| Version | Contents | Status |
|---|---|---|
| `development/version_00` | Collection 1 legacy (moved from `network/version_01`) | Kept for compatibility; removal planned |
| `development/version_01` | Collection 2 active (moved from `network/version_02`) | Kept for compatibility; removal planned |
| `development/version_02` | reserved | — |
| `development/version_03` | stable reference (mirrors `version_04`) | Source for production |
| `development/version_04` | campaign/target catalog | Active development |
| `production/` | copy of `development/version_03` | Shipped release |

## Layout

```
network/catalog/
├── .agents/          # internal worktree, gitignored
├── development/
│   ├── version_00/   # Collection 1 (legacy)
│   ├── version_01/   # Collection 2 (active)
│   ├── version_03/   # stable reference (mirrors version_04)
│   └── version_04/   # development
│       ├── campaigns/  # one JSON per campaign + index.json
│       ├── targets/    # one JSON per target (only burned_area enabled)
│       ├── src/core/   # M_campaigns.py, M_targets.py, M_catalog.py + pipeline
│       ├── notebooks/  # one thin notebook per country/campaign
│       └── COLAB_LINKS.csv|txt
└── production/       # copy of version_03 (shipped release)
    ├── campaigns/
    ├── targets/
    ├── src/core/
    ├── notebooks/
    ├── COLAB_LINKS.csv|txt
    └── PRODUCTION.md
```

`.agents/` is our development workspace and is not shipped. `version_03` is the
**stable reference**; `version_04` is the **active development copy**;
`production/` is the shipped copy of `version_03`.

## Campaigns (7 countries)

| Country | Monitor (S2 monthly) | Annual (Landsat) |
|---|---|---|
| Brazil | `brazil.monitor_01` | `brazil.collection_06` |
| Bolivia | `bolivia.monitor_01` | `bolivia.collection_02` |
| Colombia | `colombia.monitor_01` | `colombia.collection_02` |
| Chile | `chile.monitor_01` | `chile.collection_02` |
| Ecuador | `ecuador.monitor_01` | `ecuador.collection_02` |
| Paraguay | `paraguay.monitor_01` | `paraguay.collection_02` |
| Venezuela | `venezuela.monitor_01` | `venezuela.collection_02` |

Annual campaigns use **Collection 02**, except Brazil (**Collection 06**).

## Usage

```python
from M_campaigns import set_campaign
set_campaign("brazil.monitor_01", target_id="burned_area")
```

Development notebooks are under `development/version_04/notebooks/`. The public
entry points are under `production/notebooks/`.

## Colab links

- Production (shared with users):
  [`production/COLAB_LINKS.csv`](./production/COLAB_LINKS.csv) and
  [`production/COLAB_LINKS.txt`](./production/COLAB_LINKS.txt).
- Development (internal tests):
  `development/version_04/COLAB_LINKS.csv`.

## Adding a campaign or target

- New campaign: add `development/version_04/campaigns/{country}.{campaign}.json`
  and, if desired, a notebook. No pipeline code changes.
- New target: add `development/version_04/targets/{id}.json`. Only
  `task: "binary"` is implemented today; other tasks fail loudly until the
  profile is built.

## Release flow

1. Edit `development/version_04`.
2. When ready, refresh `development/version_03` with the `version_04` contents.
3. Copy the `version_03` contents to `production/`.
4. Regenerate the production `COLAB_LINKS.csv`/`.txt`.

## Deprecation plan

`version_00` and `version_01` are kept only so existing scripts, notebooks and
links continue to work. They will be removed after a public notice. New work
must go to `version_04`. `version_02` is reserved.

## Provenance

`development/version_03/REFERENCE.md` records the source version/commit. The
original Peru baseline is `peru-fire/mapbiomas_fire_monitor/version_01` at
`58897ca`, preserved in the peru-fire repository and in git history.
