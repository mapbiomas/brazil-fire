# Export & Vectorization — Monitor do Fogo (Brasil)

Pipeline de 4 etapas para processar os mapas mensais de area queimada do Monitor
do Fogo: exportacao do GEE, mosaico, vetorizacao e publicacao no GEE.

## Estrutura

```
export_and_vectorization/
├── README.md
├── mapbiomas_fire_monitor_brazil.ipynb   ← notebook (Colab)
├── state.py                              ← cache e scan GCS/GEE
├── export.py                             ← GEE → GCS tiles
├── mosaic.py                             ← gdalbuildvrt + gdal_translate
├── vectorize.py                          ← gdal_polygonize + upload GEE
└── ui.py                                 ← UI interativa (grid + pipeline)
```

## Como usar

1. Abra o notebook `mapbiomas_fire_monitor_brazil.ipynb` no Google Colab.
2. Execute a celula 1 para instalar dependencias.
3. Execute a celula 2 para autenticar no GCP e Google Earth Engine.
4. Execute a celula 3 para abrir a interface.
5. Clique em **Sincronizar** para carregar o estado atual.
6. Clique em **Processar Selecionados** para executar as etapas pendentes.

## Fluxo de processamento

```
GEE ImageCollection
       │
       ▼  [1] Export (export.py)
tiles .tif no GCS  →  monitor/monthly_images/temp/
       │
       ▼  [2] Mosaic (mosaic.py)
mosaico COG no GCS →  monitor/monthly_images/monthly_burned/
       │
       ▼  [3] Vectorize (vectorize.py)
shapefile no GCS   →  monitor/monthly_vectors/monthly_burned/
       │
       ▼  [4] Upload GEE (vectorize.py)
FeatureCollection   →  fire_monitor_v1_monthly_burned_brazil_vectors/
```

## Caminhos

| Recurso | Path |
|---------|------|
| ImageCollection (origem) | `projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1` |
| Tiles GCS | `gs://mapbiomas-fire/monitor/monthly_images/temp/` |
| Mosaicos GCS | `gs://mapbiomas-fire/monitor/monthly_images/monthly_burned/` |
| Vetores GCS | `gs://mapbiomas-fire/monitor/monthly_vectors/monthly_burned/` |
| Vetores GEE | `projects/mapbiomas-workspace/FOGO/MONITORAMENTO/fire_monitor_v1_monthly_burned_brazil_vectors/` |

## Convencoes de nomes

- Tiles: `fire_monitor_v1_monthly_burned_brazil_{YYYY}_{MM}XXXXXXXXXX-XXXXXXXXXX.tif`
- Mosaico: `monthly_burned-brazil_{YYYY}_{MM}.tif`
- Vetor: `monthly_burned-brazil_{YYYY}_{MM}.shp`

## Dependencias

- **GDAL**: `gdalbuildvrt`, `gdal_translate`, `gdal_polygonize.py`
- **Python**: `gcsfs`, `earthengine-api`, `geopandas`, `rasterio`, `psutil`, `ipywidgets`
- **Google Cloud**: autenticacao via `google.colab.auth`
- **Google Earth Engine**: autenticacao via `ee.Authenticate()`

## Dados ja processados

Meses ja completos (export + mosaico + vetor GCS + vetor GEE) aparecem como **OK**
na interface e sao ignorados durante o processamento. Apenas meses novos ou
incompletos sao processados.
