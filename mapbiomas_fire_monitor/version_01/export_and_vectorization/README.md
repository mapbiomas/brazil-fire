# Export & Vectorization — Monitor do Fogo

Pipeline de 5 etapas para processar os mapas mensais de area queimada do Monitor
do Fogo (multipais): exportacao do GEE, mosaico, vetorizacao, publicacao no GEE
e sincronizacao com o bucket publico.

## Estrutura

```
export_and_vectorization/
├── README.md
├── mapbiomas_fire_monitor_brazil.ipynb   ← notebook (Colab)
├── config.py                             ← configuracao multipais (paths derivados)
├── state.py                              ← cache e scan GCS/GEE
├── export.py                             ← GEE → GCS tiles (Byte 0/1)
├── mosaic.py                             ← gdalbuildvrt + gdal_translate
├── vectorize.py                          ← gdal_polygonize + zip + upload GEE
├── publish.py                            ← sync bucket publico + limpeza temp
└── ui.py                                 ← UI interativa (grid + pipeline)
```

## Como usar

1. Abra o notebook `mapbiomas_fire_monitor_brazil.ipynb` no Google Colab.
2. Execute a celula 1 para instalar dependencias.
3. Execute a celula 2 para autenticar no GCP e Google Earth Engine.
4. Na celula de config, defina `COUNTRIES` (lista de abas) e demais configs.
5. Opcional: células "Zerar estado local" e "Diagnostico" antes de comecar.
6. Execute a celula da UI para abrir a interface.
7. Troque de pais pela **aba** (cada aba tem sua propria grid).
8. Use o **dropdown de Ano** para filtrar e trabalhar um ano por vez.
9. Clique em **Sincronizar** e processe as etapas pendentes.

## Trocar de pais (abas)

A UI abre uma **aba por pais** (com bandeira). Trocar de aba reconfigura o pais
ativo e re-sincroniza a grid daquele pais — sem editar codigo e sem reiniciar o
kernel. As celulas de processamento (Export/Mosaico/Vetorizacao/Upload/Publicar)
sempre atuam sobre o pais da aba ativa.

```python
COUNTRIES = ["brazil", "indonesia"]   # abas disponiveis na UI
```

Toda a config (coleção, GCS, assets GEE) é derivada do pais ativo em tempo de
chamada, entao trocar de aba sempre propaga para os modulos.

## Colunas da grid (etapas)

| Coluna | Quando vira **OK** | Celula |
|--------|--------------------|--------|
| Export | tiles no GCS (`temp/`) | Export |
| Mosaico | COG montado | Mosaico |
| Vetor GCS | ZIP vetorial no GCS | Vetorizacao |
| Vetor GEE | FeatureCollection no GEE | Upload GEE |
| Publico | COG espelhado no `mapbiomas-public` | Publicar |
| Clean temp | tiles de `temp/` removidos apos consolidacao | Publicar |

A legenda da grid traz a dica **MISS → OK** com a celula que resolve cada coluna.

## Selecionar um ano ou meses especificos

- **Filtro por ano**: o dropdown de ano na UI restringe a grid aos meses do ano
  escolhido. Os botoes **Selecionar Pendentes** e **Selecionar Todos** valem
  apenas para os meses visiveis no filtro.
- **Mes por mes**: marque/desmarque os checkboxes da grid normalmente.
- **Recomecar um periodo ("do zero")**: use a celula de LIMPEZA (descomentando os
  blocos) para apagar tiles / mosaicos / vetores ZIP de um intervalo de anos e
  depois Sincronize — os meses voltam a aparecer como pendentes.

## Fluxo de processamento

```
GEE ImageCollection
       │
       ▼  [1] Export (export.py)  → tiles 0/1 .tif no GCS  .../{product}/temp/
       │
       ▼  [2] Mosaic (mosaic.py)  → COG 0/1 no GCS         .../{product}/
       │
       ▼  [3] Vectorize (vectorize.py) → shapefile+unique_id → .zip
       │                                    .../{product_vectors}/
       │
       ▼  [4] Upload GEE (vectorize.py)
       │         projects/mapbiomas-{country}/assets/FIRE/MONITOR/{product_vectors}
       │
       ▼  [5] Publish (publish.py) → espelha COGs+ZIPs no bucket publico
              e apaga tiles temp/ dos meses consolidados
```

## Padrao de caminhos

**GCS (bucket de processamento):** `gs://{bucket}/initiatives/{country}/fire/monitor/{product}`

| Recurso | Path |
|---------|------|
| ImageCollection (origem) | `projects/mapbiomas-public/assets/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1` |
| Tiles GCS | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1/temp/` |
| Mosaicos GCS | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1/` |
| Vetores GCS (ZIP) | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_vectors_v1/` |
| Vetores GEE | `projects/mapbiomas-{country}/assets/FIRE/MONITOR/mapbiomas_fire_monthly_burned_vectors_v1/` |
| Publico (espelho) | `gs://mapbiomas-public/initiatives/{country}/fire/monitor/...` |

Paises suportados: `brazil`, `indonesia`. Novos paises entram no dict `COUNTRIES`
em `config.py`.

## Convencoes de nomes

- Tiles: `fire_monitor_v1_monthly_burned_{country}_{YYYY}_{MM}XXXXXXXXXX-XXXXXXXXXX.tif`
- Mosaico: `monthly_burned-{country}_{YYYY}_{MM}.tif`
- Vetor (zip): `monthly_burned-{country}_{YYYY}_{MM}.zip`

## Export leve (Byte 0/1)

A exportacao grava tiles **Byte 0/1 puro** (sem `selfMask`, sem banda de mascara,
sem nodata): 0 = sem fogo, 1 = fogo. Uma unica banda maximiza a compressao LZW e
mantem os arquivos na casa dos ~10-20 MB para o pais inteiro (mesmo comportamento
da v1 original). O mosaico mantem o mesmo formato; a vetorizacao usa
`gdal_polygonize -mask` para ignorar os pixels 0.

## Publicacao (celula 5)

`publish_all` faz um sync incremental para o bucket publico (espelho do
`mapbiomas-public`):

1. Copia COGs (`.../{product}/*.tif`) e vetores ZIP (`.../{product_vectors}/*.zip`)
   que ainda nao estao no publico (valida tamanho apos a copia).
2. Para meses com **COG + ZIP validados no publico**, apaga os tiles de `temp/`
   (libera espaco; tiles eram o maior volume intermediario).

Idempotente — pode rodar uma vez por mes para pegar os meses que faltaram.

## Dependencias

- **GDAL**: `gdalbuildvrt`, `gdal_translate`, `gdal_polygonize.py`
- **Python**: `gcsfs`, `earthengine-api`, `geopandas`, `rasterio`, `psutil`, `ipywidgets`
- **Google Cloud**: autenticacao via `google.colab.auth`
- **Google Earth Engine**: autenticacao via `ee.Authenticate()`

## Dados ja processados

Meses ja completos (export + mosaico + vetor GCS + vetor GEE) aparecem como **OK**
na interface e sao ignorados durante o processamento. Apenas meses novos ou
incompletos sao processados. Como o COG so existe apos export+mosaico, meses com
`temp/` ja limpos continuam marcados como OK.
