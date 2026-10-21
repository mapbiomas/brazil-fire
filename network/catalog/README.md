# MapBiomas Fire Catalog

Classification catalog for the MapBiomas Fire network. It evolves from the
Peru monitor pipeline (`peru-fire/mapbiomas_fire_monitor/version_01`) and
makes **campaign** and **target** first-class concepts.

- **Campaign** = where the data comes from (country + source + sensor + periodicity).
- **Target** = what is predicted (`burned_area` today; other profiles later).

The catalog never redefines the export layout. Each campaign declares its own
`gcs_root`/`gee_root`, and every path is derived from it, so the addresses used
by the export/vectorization pipeline stay exactly the same.

## Layout

```
network/catalog/
├── .agents/        # internal worktree (branch catalog/v01), gitignored
├── version_00/     # frozen snapshot of peru-fire monitor v01
└── version_01/     # product shipped in this repository
    ├── campaigns/  # one JSON per campaign + index.json
    ├── targets/    # one JSON per target (only burned_area enabled)
    ├── src/core/   # M_campaigns.py, M_targets.py, M_catalog.py + pipeline
    └── notebooks/  # one thin notebook per country/campaign
```

`.agents/` is our development workspace and is not shipped. `version_00` is a
frozen baseline; `version_01` is the editable product.

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

The notebooks under `version_01/notebooks/` are thin entry points: they clone
this repository, call `set_campaign(...)`, authenticate, and run the existing
M1-M6 interfaces.

## Adding a campaign or target

- New campaign: add `version_01/campaigns/{country}.{campaign}.json` and, if
  desired, a notebook. No pipeline code changes.
- New target: add `version_01/targets/{id}.json`. Only `task: "binary"` is
  implemented today; other tasks fail loudly until the profile is built.

## Provenance

`version_00` is a copy of `mapbiomas/peru-fire`,
`mapbiomas_fire_monitor/version_01`, commit `58897ca` (2026-10-21). See
`version_00/SOURCE.md`.
