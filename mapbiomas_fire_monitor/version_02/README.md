# mapbiomas_cog (version_02)

Engine de export/resume de COGs multi-iniciativa do MapBiomas Fire.

Substitui a "fabrica" de download do `version_01/export_and_vectorization` por
um engine tiled/resumivel com estado manifest-first, mantendo a mesma
experiencia no Colab (mesma autenticacao) e o notebook `all_initiatives` como
alvo unico desta v02.

## Abrir no Google Colab

[![Open In Colab — All Initiatives](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/mapbiomas_fire_monitor/version_02/notebooks/mapbiomas_directlink_all_initiatives.ipynb)

Rode as células em ordem: **1** clone+install → **2** auth → **3** config → **4** UI.
A célula 1 é idempotente e usa `sys.path` como fallback, então `import mapbiomas_cog`
funciona mesmo se o `pip install -e` falhar.

## Status: M1–M7

| # | Entregue |
|---|----------|
| M1 | Dominio: grid determinístico (`EPSG:6933`), `Product`/`Unit`, adapter do `OBJ` |
| M2 | Engine: runner `ee_batch_tiled` (submit/sync/retry) + manifest por tile |
| M3 | Assembly: COG em Python (rasterio), sem `gdalbuildvrt`/`gdal_translate` |
| M4 | UI: `mapbiomas_cog.ui.run_ui` + notebook fino `all_initiatives` |
| M5 | CLI, `catalog` (inventário/status) e `publish.gcs` (espelho no `mapbiomas-public`) |
| M6 | Runner `hv_local` (High Volume API em Python puro) + fallback + benchmark |
| M7 | Testes offline com EE/GCS fake, CI (GitHub Actions), docs |

Pendente: validação no Colab (submissão EE real e assembly rasterio) e
`publish.gee` (vetores), adiado para fase própria.

## Runners

| Runner | Como funciona | Quando usar |
|--------|---------------|-------------|
| `ee_batch_tiled` (default) | `Export.image.toCloudStorage` por tile | Custo/infra do version_01; escala |
| `hv_local` | `ee.data.computePixels` por tile em threads + backoff | Latência baixa, unidades menores; mais EECU |
| `dataflow` | plugin futuro (datensee) | paralelismo massivo |

Ambos gravam tiles no GCS e usam o mesmo `assemble` (COG) e manifest. O
`hv_local` é opt-in; o `Engine` aceita `fallback=` (ex.: `hv_local` →
`ee_batch_tiled`).

## CLI

```bash
mapbiomas-cog status
mapbiomas-cog status --country ecuador --theme lulc --collection collection_03 --product coverage
mapbiomas-cog export   --country ecuador --theme lulc --collection collection_03 --product coverage [--runner hv_local]
mapbiomas-cog sync     --country ecuador --theme lulc --collection collection_03 --product coverage
mapbiomas-cog assemble --country ecuador --theme lulc --collection collection_03 --product coverage
mapbiomas-cog publish  --country ecuador --theme lulc --collection collection_03 --product coverage
```

## Benchmark de runners

```bash
python -m mapbiomas_cog.benchmarks.benchmark_runner \
  --country ecuador --theme lulc --collection collection_03 --product coverage \
  --unit 1985 --sample 4        # ou --plan-only
```

## Estrutura

```
version_02/
├── pyproject.toml
├── src/mapbiomas_cog/
│   ├── domain/      modelos, TileGrid, paths, config (adapter do OBJ)
│   ├── state/       manifest (_manifest.json / _failures.json)
│   ├── engine/      facade, planner, images, runners/{ee_batch,hv_local}
│   ├── assemble/    COG em Python (rasterio)
│   ├── catalog/     inventario + status (manifest-first)
│   ├── publish/     GCS publico (+ GEE, futuro)
│   ├── ui/          UI ipywidgets
│   ├── benchmarks/  benchmark_runner
│   └── cli.py
├── notebooks/mapbiomas_directlink_all_initiatives.ipynb
└── tests/           (offline; EE/GCS fake)
```

## Desenvolvimento

```bash
pip install -e ".[dev]"
mapbiomas-cog --version
pytest -q          # 33 testes offline
```

CI: `.github/workflows/mapbiomas-cog-ci.yml` (roda `pytest` com `PYTHONPATH=src`).

## Extracao futura

O pacote foi desenhado para virar `github.com/mapbiomas/mapbiomas_cog`:
- catalogo acessado por caminho, configurável via `MAPBIOMAS_COG_CONFIG`;
- sem dependência de código do `version_01` além do `config.py` (OBJ);
- layout `src/` pronto para `git filter-repo`.

Decisao de arquitetura: `adr/013-mapbiomas-cog-export-engine.md`.
Especificacao: `specs/mapbiomas-cog/spec.md`.
