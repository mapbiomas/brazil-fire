# Catalog experiments

Independent method tests that read the data produced by the catalog
(`../development/`, `../production/`) but do not change it. Each experiment
has its own folder with code, notebook, tests and README, and imports
nothing from the catalog versions. An experiment that works may later become
part of a new catalog version.

| Experiment | What it tests | Status |
|---|---|---|
| [`unet_pantanal`](unet_pantanal/) | U-Net on image chips instead of the pixel MLP (Brazil, Pantanal, 2024) | in test |
