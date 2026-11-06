# MapBiomas Fire Scar Monitoring Pipeline — M0 a M7

## Visão Geral

Pipeline de 7 estágios (M0–M7) que processa imagens de satélite do Google Earth Engine,
exporta para Cloud Storage, monta mosaicos nacionais, treina uma rede neural profunda
(DNN) em amostras fire/notFire coletadas manualmente, classifica tiles por região,
pós-processa e publica produtos versionados de cicatriz de fogo de volta ao GEE.

```
GEE ──M1──→ GCS (shards) ──M2──→ COGs ──M4──→ model.npz
                                          ↑
                                     M3 (samples CSV)
                                          ↓
GEE ←──M6/M7── COGs + estatísticas ←──M5── model.npz + COGs
```

---

## Módulos

### M0 — Setup & Authentication (`M0_auth_config.py`)
- Config globals: país, bucket, versão, bands, sensores, periodicidades, mosaic_methods
- Helpers de path GCS e GEE
- Lazy init EE/GCSFS
- `authenticate()`, `set_global_opts()`, `print_config()`

### M1 — Export GEE → GCS (`M1_export_logic.py`, `M1_export_ui.py`)
- Quality mosaic (minnbr) de coleções Sentinel-2 / Landsat 8/9
- Exporta 1 GeoTIFF por banda/chunk/região para GCS
- UI com abas: anos, checkboxes de sensor/banda, dispatch/delete, tracking de tasks

### M2 — Mosaic Assembly (`M2_mosaic_logic.py`, `M2_mosaic_ui.py`)
- Download shards do GCS → GDAL VRT → COG (DEFLATE)
- Upload do COG para `LIBRARY_IMAGES/{SENSOR}/MONTHLY/{MOSAIC}/{date}/COG/`
- UI: seletores ano/região/banda, progresso, retry, resume

### M3 — Sample Collection (`M3_sample_ui.py`, `M3_toolkit.js`)
- **Python**: thin wrapper que mostra links para o GEE Code Editor
- **JavaScript**: toolkit completo (~500 linhas) no GEE:
  - Seletor de país/região, visualizador de mosaicos monthly/annual
  - Ferramentas de desenho fire (vermelho) / notFire (azul)
  - Formulário de metadados (satélite, data, campanha, coletor)
  - Export automático para CSV no GCS

### M4 — DNN Training (`M4_algorithms_dnn.py`, `M4_analytics.py`, `M4_ui.py`, `M4_data_extractor.py`, `M4_hub_manager.py`)
- **Extração**: carrega CSVs de amostras, clipa COGs com `rasterio.mask` → arrays X, y
- **Balanceamento**: none / undersample / oversample
- **Treinamento**: feedforward DNN (TF1 compat), configurável (layers, lr, epochs, batch)
- **Metadados**: salva architecture, normalizer stats, métricas em `metadata.json`
- **Checkpoints**: `model.npz` (pesos + embeddings) no GCS
- **Dashboard**: 2×3 grid (matriz confusão, distribuição de probabilidades, PCA, UMAP, PR-curve)
- **Model Hub**: cache de modelos treinados (`M4_ranking_cache.json`)

### M5 — Classification (`M5_queue.py`, `M5_inference.py`, `M5_classifier.py`, `M5_publisher.py`, `M5_classifier_ui.py`)
- **Queue**: arquivo `m5_queue.json` com file-locking, ciclo pending → running → done/fail
- **Inference**: baixa COGs + modelo do GCS, classifica cada tile (7 bandas → Byte GeoTIFF)
- **Merging**: `rasterio.merge` dos tiles por região
- **Estatísticas**: CSV com área queimada em hectares
- **Upload GEE**: exporta COG merged como `ee.Image`
- **UI**: seletores modelo/região/período, fila, progresso inline, thumbnails, histórico

### M6 — Post-Processing (`M6_publisher.py`)
- **LULC mask**: exclui classes água/urbano/solo nu (26, 22, 33, 24)
- **Sieve**: `rasterio.features.sieve(connectivity=4)` remove ruído sal-e-pimenta
- **Mosaico nacional**: merge VRT → COG de todas as regiões
- **Upload versionado**: ImageCollection no GEE com metadados de proveniência

### M7 — Curation (`M7_curator.py`)
- **(Stub)** — esqueleto arquitetural, I/O real com placeholders
- Planejado: listar variantes por região → selecionar melhor → mosaico nacional → publicar "Pré-Official"

---

## Supporting Modules

| Módulo | Propósito |
|---|---|
| `M_cache.py` | `CacheManager` — cache JSON local com sync opcional GCS |
| `M_mosaics.py` | Registry `MOSAIC_METHODS`: `minnbr` (ativo), `minnbr_buffer`, `median`, `minndvi` |
| `M_regions.py` | `REGION_ASSETS` — 9 países, property `region_nam` padronizada |
| `M_lang.py` | `L` class — ~190 strings UI em EN/ES/PT |
| `M_ui_components.py` | Componentes reutilizáveis: spinner, empty_state, sync_btn, thumbnails, cards |

---

## Fluxo do Usuário (Ordem no Notebook)

1. **M0** → Configurar país, bucket, sensor, periodicidade, mosaic_method, campaign
2. **M1** → Selecionar anos/sensores/bandas → clicar "Dispatch" → aguardar tasks GEE
3. **M2** → Selecionar ano/região/banda → clicar "Assemble" → aguardar COGs
4. **M3** → Abrir link do GEE JS Toolkit → desenhar amostras → exportar CSV
5. **M4** → Selecionar campanha → configurar hiperparâmetros → "Train" → ver dashboard
6. **M5** → Selecionar modelo → adicionar jobs (região/período) → engine classifica → publica
7. **M6** → (Opcional) aplicar máscara LULC + sieve → upload versionado
8. **M7** → (Stub) curar variantes → publicar pré-oficial

---

## Prompt para Diagrama Mermaid

```
Gere um diagrama Mermaid flowchart da pipeline completa do MapBiomas Fire Scar Monitoring (M0–M7).
O diagrama deve ter duas swimlanes/níveis: "User Actions" (o que o usuário faz no notebook)
e "Pipeline Processing" (processamento automático).

Fluxo geral:

M0 — Setup: usuário configura país, bucket, sensor, mosaic_method, campaign.
      Sistema autentica GEE + GCS e inicializa CONFIG/GLOBAL_OPTS.

M1 — Export: usuário seleciona anos, sensores, bandas e clica "Dispatch".
      Sistema exporta tiles do GEE para GCS (um GeoTIFF por banda/chunk/região).
      Aguarda tasks do GEE.

M2 — Mosaic: usuário seleciona ano, região, banda e clica "Assemble".
      Sistema baixa shards do GCS, cria VRT com GDAL, traduz para COG comprimido,
      salva em GCS.

M3 — Samples: usuário abre link do GEE JavaScript Toolkit, desenha polígonos
      fire/notFire no mapa, preenche metadados, exporta CSV para GCS.

M4 — Training: usuário seleciona campanha de amostras, configura hiperparâmetros
      (layers, lr, epochs) e clica "Train". Sistema:
      (1) extrai pixels dos COGs via rasterio.mask
      (2) balanceia classes (none/undersample/oversample)
      (3) treina DNN (TF1/Keras)
      (4) gera metadados + normalizer stats
      (5) salva model.npz + metadata.json em GCS
      (6) mostra dashboard analytics (matriz confusão, PCA, PR-curve) e model card

M5 — Classification: usuário seleciona modelo treinado, regiões, períodos e
      adiciona jobs à fila. Engine:
      (1) para cada job pending, baixa COGs + modelo do GCS
      (2) classifica cada tile (7 bandas → fire probability → Byte GeoTIFF)
      (3) faz merge dos tiles por região (rasterio.merge)
      (4) gera estatísticas (ha queimado)
      (5) faz upload do merged COG para GEE como ee.Image
      UI mostra progresso inline por job com thumbnails

M6 — Post-processing: sistema opcionalmente:
      (1) aplica máscara LULC (exclui água/urbano/solo nu)
      (2) aplica sieve (remove pixels isolados < limiar)
      (3) monta mosaico nacional de todas as regiões
      (4) faz upload versionado para GEE como ImageCollection

M7 — Curation: (stub) sistema lista variantes por região, seleciona melhor,
      merge nacional, publica como "Pré-Official" no GEE.

Requisitos do diagrama:
- Use subgraph para cada módulo M0–M7
- Mostre as setas de dependência entre módulos
- Inclua decisões (diamantes) para: monthly vs yearly, balance_method, sieve on/off
- Mostre os artefatos intermediários: GEE Asset, GCS bucket, COG files, CSV samples,
  model.npz, metadata.json, m5_queue.json, final GEE ImageCollection
- Use cores para diferenciar User Actions (azul) vs Processing (verde) vs Data Stores (cinza)
- Estilo flowchart TB (top-to-bottom)
```

---

## Histórico de Correções (Sessão Atual)

| Commit | Descrição |
|---|---|
| `06b815d` | M5 Phase 1 — 6 crashes |
| `4a504db` | M5 Phase 2 — 6 logic errors |
| `6eff431` | M5 Phase 3 — 4 robustness fixes |
| `8e3257a` | M4 Phase 1 — 6 crashes |
| `fbcdacd` | M4 Phase 2 — 6 logic issues |
| `2a40672` | M4 Phase 3 — 8 robustness issues |
| `864ff94` | M4 remanescentes #18 e #3 |
| `1cbd765` | M5 alignment: `_campaign()`, queue lock, bare except |
| `e3ac6ad` | 4 bugs residuais M4/cache/M1 |
| `87593b9` | M1/M2 audit: gsutil→gcsfs, sensor propagation, `_get_fs()` cache |
| `9e2ff0f` | Indentação 3 arquivos (SyntaxError) |
| `9a5b83d` | Fix indent json.dump M_cache.py |
| `d235a30` | Fix typo peridiocity_active → periodicity_active |

**Total: ~40+ bugs corrigidos, 0 bare `except:` restantes.**

### Decisões-Chave
- **gcsfs sobre gsutil** — consistência, sem subprocess frágil
- **`_get_fs()` cached globalmente** — instância única em M0_auth_config
- **Sensor propagado como parâmetro** — evita cross-wiring entre abas UI e global default
- **File-based advisory lock** para `m5_queue.json` — cross-platform, previne race UI vs engine
- **`hp_override`** em `view_analytics()` — pula leitura GCS após treino
- **Batch tolera falhas** — wrap individual com try/except por item

### Bugs Conhecidos (Deferidos)
- `M6_publisher.py:34` — importa `list_regions` que não existe (usuário pediu para pular)
- `M7_curator.py` — stub completo, I/O real com placeholders
