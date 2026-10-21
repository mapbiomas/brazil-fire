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
| `version_00` | Collection 1 legacy (moved from `network/version_01`) | Kept for compatibility; removal planned |
| `version_01` | Collection 2 active (moved from `network/version_02`) | Kept for compatibility; removal planned |
| `version_02` | reserved | — |
| `version_03` | frozen snapshot of the Peru monitor (`peru-fire@58897ca`) | Reference only |
| `version_04` | campaign/target catalog | Active development |

## Layout

```
network/catalog/
├── .agents/        # internal worktree (branch catalog/v04), gitignored
├── version_00/     # Collection 1 (legacy, compatibility)
├── version_01/     # Collection 2 (active, compatibility)
├── version_02/     # reserved
├── version_03/     # Peru monitor baseline (frozen)
└── version_04/     # product shipped in this repository
    ├── campaigns/  # one JSON per campaign + index.json
    ├── targets/    # one JSON per target (only burned_area enabled)
    ├── src/core/   # M_campaigns.py, M_targets.py, M_catalog.py + pipeline
    └── notebooks/  # one thin notebook per country/campaign
```

`.agents/` is our development workspace and is not shipped. `version_03` is a
frozen baseline; `version_04` is the editable product.

## Campaigns (7 countries)

| Country | Monitor (S2 monthly) | Annual (Landsat) |
|---|---|---|
| Brazil | `brazil.monitor_01` | `brazil.collection_02` |
| Bolivia | `bolivia.monitor_01` | `bolivia.collection_02` |
| Colombia | `colombia.monitor_01` | `colombia.collection_02` |
| Chile | `chile.monitor_01` | `chile.collection_01` |
| Ecuador | `ecuador.monitor_01` | `ecuador.collection_02` |
| Paraguay | `paraguay.monitor_01` | `paraguay.collection_01` |
| Venezuela | `venezuela.monitor_01` | `venezuela.collection_02` |

Annual collection differs per country (Collection 01 vs 02). It is a one-line
change in the campaign JSON; no code changes.

## Usage

```python
from M_campaigns import set_campaign
set_campaign("brazil.monitor_01", target_id="burned_area")
```

The notebooks under `version_04/notebooks/` are thin entry points: they clone
this repository, call `set_campaign(...)`, authenticate, and run the existing
M1-M6 interfaces.

## Adding a campaign or target

- New campaign: add `version_04/campaigns/{country}.{campaign}.json` and, if
  desired, a notebook. No pipeline code changes.
- New target: add `version_04/targets/{id}.json`. Only `task: "binary"` is
  implemented today; other tasks fail loudly until the profile is built.

## Deprecation plan

`version_00` and `version_01` are kept only so existing scripts, notebooks and
links continue to work. They will be removed after a public notice. New work
must go to `version_04`. `version_02` is reserved.

## Provenance

`version_03` is a copy of `mapbiomas/peru-fire`,
`mapbiomas_fire_monitor/version_01`, commit `58897ca` (2026-10-21). See
`version_03/SOURCE.md`.
