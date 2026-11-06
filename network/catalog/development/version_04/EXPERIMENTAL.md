# Experimental version 04

This is an experimental copy of version 03, not a production release. M1
uses a local parallel computePixels GCS exporter with per-band manifests.
M2 validates manifests before assembly and requests COGs without overviews.
Regional asset submission and the M0 configuration are integrated but have
not been validated by launching live asset tasks. Do not use this copy for
production. Do not run concurrent exports of the same band and period.

`src/core/M_export_grid.py` provides only projected, origin-anchored grid
arithmetic. The CRS, Brazilian tile size, clipping, Earth Engine request limits,
cloud authentication, retries, output manifests and downstream integrations
still require implementation and validation.

`M_compute_pixels.py` supplies the High Volume transport. A small live Landsat
request and impersonated GCS upload have been tested; national-scale exports,
interrupted Colab sessions and end-to-end M2 assembly still need testing.
New exports filter tiles conservatively against the country geometry and pin
the Earth Engine asset snapshot per band. Existing manifests retain their
original grid and snapshot behavior. The local runner is not Dataflow; it
cannot bypass Earth Engine quotas or restricted mode.

`M1_region_export.py` is connected to M0/M1. M3 has an experimental
period-filtered reader; it does not check regional completeness and still
requires live validation. Do not use it for production.
