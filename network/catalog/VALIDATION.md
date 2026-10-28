# Catalog version validation

This document records the validated entry points for every version. The
runtime code tests run per version with `python -m unittest discover -s tests`.

| Version | Entry point | Validation |
|---|---|---|
| `network/version_01/` (legacy root mirror) | `classification_algorithms/mapbiomas_fire_classification_for_collab.ipynb` | Notebook points to `network/version_01/...`; `A_0_1` keeps its historical `network/classification_algorithms` path. **Reference version:** `catalog/development/version_00`. **Deprecated:** remove after public notice |
| `network/version_02/` (legacy root mirror) | `classification_algorithms/mapbiomas_fire_classification_for_collab.ipynb` | `A_0_1` and notebook point to `network/version_02/...`; unchanged from the pre-move commit `45b7a17`. **Reference version:** `catalog/development/version_01`. **Deprecated:** remove after public notice |
| `catalog/development/version_00/` | `classification_algorithms/A_0_1...` and notebook | Both point to `network/catalog/development/version_00/classification_algorithms` (Collection 1) |
| `catalog/development/version_01/` | `classification_algorithms/mapbiomas_fire_classification_for_collab.ipynb` | `A_0_1` and notebook point to `network/catalog/development/version_01/classification_algorithms` (Collection 2) |
| `catalog/development/version_03/` | `notebooks/monitor_01_*.ipynb` and `collection_*.ipynb` | 14 notebooks; `repo_path = network/catalog/development/version_03/src/core`; 105 tests OK |
| `catalog/production/` | `notebooks/monitor_01_*.ipynb` and `collection_*.ipynb` | 14 notebooks; `repo_path = network/catalog/production/src/core`; 105 tests OK |

Catalog `version_02` is reserved and has no folder. `version_04` was removed;
`version_03` is now the active development copy and the stable reference.

## Commands

```bash
# Layout and path validation (offline)
python -m unittest discover -s network/catalog/tests -v

# Runtime code per catalog version
python -m unittest discover -s network/catalog/development/version_03/tests
python -m unittest discover -s network/catalog/production/tests
```

## Notes

- The root `network/version_02` is a **temporary mirror** for compatibility.
  It is not the canonical copy; `catalog/development/version_01` is.
- The legacy root mirror will be removed after a public notice.
- M3 toolkit: the catalog ships the v08 snapshot
  (`5-Monitor-Fuego/M3/dev/M3_Toolkit_Monitor_Fuego_v08`); `version_03` and
  `production` have identical v08 copies. The notebooks link to the copy of the
  version being run; the M3 gateway links to the current GEE production
  reference (`5-Monitor-Fuego/M3/M3_Toolkit_Monitor_Fuego`).
- M3 v08 scope: nine countries, annual axis from 1985, monthly from 2019,
  `MBFIRE Collection`/`MBFIRE Monitor` reference layers, Bolivia regions pinned
  to `CATALOG_01/COLLECTION_02`, FILTERED matrix without horizontal scrollbar.
- The catalog versions are path-validated; GEE/GCS operations still require
  authentication and are not part of the offline test suite.
- Offline layout suite: 10 tests OK (`network/catalog/tests`).
