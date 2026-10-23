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
| `development/version_03` | campaign/target catalog | Active development + source for production |
| `production/` | copy of `development/version_03` | Shipped release |

> `development/version_04` was removed; `version_03` is now both the active
> development copy and the stable reference.

## Layout

```
network/catalog/
├── .agents/          # internal worktree, gitignored
├── development/
│   ├── version_00/   # Collection 1 (legacy)
│   ├── version_01/   # Collection 2 (active)
│   └── version_03/   # campaign/target catalog (active development)
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
**active development copy and stable reference**; `production/` is the shipped
copy of `version_03`.

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

The notebooks configure the pipeline explicitly:

```python
from M0_auth_config import set_global_opts, authenticate

COUNTRY_SELECTED = "brazil"
CAMPAIGN_SELECTED = "MONITOR_01"

set_global_opts(
    country=COUNTRY_SELECTED,
    campaign=CAMPAIGN_SELECTED,
    sensor="sentinel2",
    periodicity="monthly",
    mosaic_methods=["minnbr", "minnbr_buffer"],
    personal_task_flag="MONITOR",
    language="en",
    gcs_bucket="mapbiomas-fire",
    gcs_library_images_prefix=f"sudamerica/{COUNTRY_SELECTED}/CATALOG_01",
    gcs_campaigns_prefix=f"sudamerica/{COUNTRY_SELECTED}/CATALOG_01",
    gee_project=f"mapbiomas-{COUNTRY_SELECTED}",
    gee_library_images_prefix=f"projects/mapbiomas-{COUNTRY_SELECTED}/assets/FIRE/CATALOG_01",
    gee_campaigns_prefix=f"projects/mapbiomas-{COUNTRY_SELECTED}/assets/FIRE/CATALOG_01",
    asset_regions=f"projects/mapbiomas-{COUNTRY_SELECTED}/assets/FIRE/CATALOG_01/{CAMPAIGN_SELECTED}/AUXILIARY/regiones_fuego_{COUNTRY_SELECTED}_v1",
)
authenticate()
```

`set_campaign("brazil.monitor_01", target_id="burned_area")` remains available as
an optional helper for scripts and catalog tooling.

Campaigns are declared in `campaigns/*.json` and targets in `targets/*.json`.
The catalog covers seven countries; annual campaigns use **Collection 02**,
except Brazil (**Collection 06**).


Development notebooks are under `development/version_03/notebooks/`. The public
entry points are under `production/notebooks/`.

## Colab links

- Production (shared with users):
  [`production/COLAB_LINKS.csv`](./production/COLAB_LINKS.csv) and
  [`production/COLAB_LINKS.txt`](./production/COLAB_LINKS.txt).
- Development (internal tests):
  `development/version_03/COLAB_LINKS.csv`.

## Adding a campaign or target

- New campaign: add `development/version_03/campaigns/{country}.{campaign}.json`
  and, if desired, a notebook. No pipeline code changes.
- New target: add `development/version_03/targets/{id}.json`. Only
  `task: "binary"` is implemented today; other tasks fail loudly until the
  profile is built.

## Release flow

1. Edit `development/version_03`.
2. Copy the `version_03` contents to `production/`.
3. Regenerate the production `COLAB_LINKS.csv`/`.txt`.
4. M3 toolkit: the catalog ships the v08 snapshot
   (`5-Monitor-Fuego/M3/dev/M3_Toolkit_Monitor_Fuego_v08`). Copy it to
   `development/version_03/src/core/M3_toolkit.js` and promote it with the rest;
   `production` keeps an identical copy. The GEE link points to
   `5-Monitor-Fuego/M3/M3_Toolkit_Monitor_Fuego` (current production) as the
   live reference.

## Deprecation plan

`version_00` and `version_01` are kept only so existing scripts, notebooks and
links continue to work. They will be removed after a public notice. New work
must go to `version_03`. `version_02` is reserved.

## Provenance

`development/version_03/REFERENCE.md` records the source version/commit. The
original Peru baseline is `peru-fire/mapbiomas_fire_monitor/version_01` at
`58897ca`, preserved in the peru-fire repository and in git history.

## Validation

See [`VALIDATION.md`](./VALIDATION.md) for the validated entry points and the
offline check command:

```bash
python -m unittest discover -s network/catalog/tests -v
```
