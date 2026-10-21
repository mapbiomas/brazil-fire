# Source snapshot

This directory is a frozen copy of the Peru Fire Monitor pipeline.

- **Repository:** https://github.com/mapbiomas/peru-fire
- **Path:** `mapbiomas_fire_monitor/version_01`
- **Commit:** `58897ca` — `fix: align burned-area threshold and auto-refresh caches`
- **Date:** 2026-10-21
- **Copied by:** MapBiomas Fire hub (agent session)

## Policy

`version_03` is a **baseline**: it is not edited. The working copy is
`../version_04/`, which evolves toward the campaign/target catalog. Fixes made
in `peru-fire` after this commit are **not** synced automatically; backport
them manually when they matter.

## Excluded from the copy

Generated or local-only artifacts were intentionally left out:

- `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`
- `TEMPORARY/`
- `state.json`, `m4_ranking_cache.json` (local caches)
- `gcs_audit_report.json` (generated audit)
