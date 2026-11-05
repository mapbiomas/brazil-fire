# MapBiomas Fire Monitor — M7 / M8 / M9 Scripts

JavaScript (GEE Code Editor) scripts for MONITOR_01 post-classification pipeline.
Sincronizado com `mapbiomas-fire/5-Monitor-Fuego` (M7 v4.9, M8 v3.1, M9 v5.5).

## Scripts

| Script | Stage | Description |
|---|---|---|
| `M7_00_selection_merge.js` | M7 — UI | Form (campaign + collection + period) + model-per-region selection + national image export (SplitPanel) |
| `M7_01_lulc_filters.js` | M7 — ft01 | Water LULC mask by region + solitary pixel cleanup (connectedPixelCount <= 2) |
| `M7_02_temporal_filter.js` | M7 — ft02 | Monthly memory filter (removes burn if scar in previous months) |
| `M8_statistics.js` | M8 — Stats | Universal statistics generator (area + LULC) → CSV GCS/Drive |
| `M9_00_promote_ui.js` | M9 — UI | Redesign v5.5: paises checkboxes multi + campanhas multi (root por pais+campanha) + grid de datas; etapas ftXX auto-descobertas; grid por etapa (propuestas x data global); **cor unica por ft/proposta** (paleta 40 cores) + quadrado colorido antes do checkbox e no header da coluna; ✅ por celula com origem persistida via setAssetProperties; aviso de sincronizacao; botao dinamico: data promovida => Despromover; mosaico MINNBR por pais; mapa |

## Naming

- Folders: `FILTERED/`, `PRE_PUBLIC/` (UPPER)
- Collections: `{collection}/ft{NN}` (lower), ex: `propuesta_a/ft01`
- Images: `{ano}` or `{ano}_{mes}` (lower)
- Bands: `probability` + `dayOfYear`

## Pipeline M7

```
M6 REGIONAL/{model_id}/{region}_{period}
   → M7_00  → FILTERED/{col}/ft00/{period}   (national merge)
   → M7_01  → FILTERED/{col}/ft01/{period}   (water LULC + solitary pixels)
   → M7_02  → FILTERED/{col}/ft02/{period}   (temporal memory filter)
```

## Multiple collections

`propuesta_a` e apenas o default. Para criar outra colecao:

- `M7_00`: digite o nome em "Crear nueva colección" (ou selecione uma existente)
- `M7_01` / `M7_02`: edite `COLLECTION_BASE` no topo do script (copie/duplique por colecao)
- `M8_statistics`: universal — `COLLECTIONS = []` varre todas automaticamente; `STAGES` define os estagios
- `M9_00`: paises (checkboxes multi) + campanhas (checkbox multi, root por pais+campanha) + grid de datas; etapas `ftXX` descobertas automaticamente

## Asset paths

```
CATALOG_01/MONITOR_01/LIBRARY_CLASSIFICATIONS/
├── FILTERED/{collection}/ft{00..02}/{period}
├── PRE_PUBLIC/{campanha}/{period}
└── LIBRARY_STATISTICS/m8_stats_{collection}.csv   (GCS/Drive)
```

## Time properties

Todos os assets exportados por M7/M8 carregam `system:time_start` e
`system:time_end` (periodo mensal) — usado pelo Looker Studio, ordenacao temporal
e pelo filtro temporal do ft02.

## Land cover

- M7_01: `projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1` (classes agua [31,33,34])
- M8: mesma Collection 4 (niveis 0/1/2)