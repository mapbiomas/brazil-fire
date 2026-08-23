# Export & Vectorization — Monitor do Fogo

Pipeline de 7 etapas para processar os mapas mensais de area queimada do Monitor
do Fogo (multipais): exportacao do GEE, mosaico, vetorizacao, publicacao no GEE,
publicacao no bucket publico e limpeza do temp.

> ⚠️ **Fluxo beta/experimental**: o catalogo (`config.OBJ`) NAO inclui todos os
> dados do MapBiomas — e um subconjunto validado. Valide os resultados em escopo
> reduzido antes de escalar o processamento.

## Abrir no Google Colab

Os notebooks rodam direto no Colab a partir desta `main` (sem clone manual):

[![Open In Colab — All Initiatives](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/mapbiomas_fire_monitor/version_01/export_and_vectorization/mapbiomas_directlink_all_initiatives.ipynb)
**`mapbiomas_directlink_all_initiatives.ipynb`** — todas as iniciativas/temas (`THEMES = []`)

[![Open In Colab — Fire Monitor Brazil](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/mapbiomas_fire_monitor/version_01/export_and_vectorization/mapbiomas_fire_monitor_brazil.ipynb)
**`mapbiomas_fire_monitor_brazil.ipynb`** — somente fogo (`THEMES = ["fire"]`)

> O link segue o padrao
> `https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/<caminho>/notebook.ipynb`.

## Estrutura

```
export_and_vectorization/
├── README.md
├── mapbiomas_directlink_all_initiatives.ipynb   ← notebook multi-iniciativa (Colab)
├── mapbiomas_fire_monitor_brazil.ipynb          ← notebook focado em fogo (Colab)
├── config.py                             ← configuracao multipais (paths derivados)
├── state.py                              ← cache e scan GCS/GEE
├── export.py                             ← GEE → GCS tiles (Byte 0/1)
├── mosaic.py                             ← gdalbuildvrt + gdal_translate
├── vectorize.py                          ← gdal_polygonize + zip + upload GEE
├── publish.py                            ← sync bucket publico + limpeza temp
└── ui.py                                 ← UI interativa (grid + pipeline)
```

## Como usar

1. Abra um dos notebooks no Google Colab pelos badges da secao acima.
2. Execute a celula 1 para instalar dependencias.
3. Execute a celula 2 para autenticar no GCP e Google Earth Engine.
4. Na celula de config, defina `COUNTRIES` (códigos do OBJ, ex.: `["brasil", "indonesia"]`).
5. Opcional: células "Adicionar coleção", "Zerar estado local" e "Diagnostico".
6. Execute a celula da UI para abrir a navegação.
7. **Navegue**: país → tema (ex.: `fire`) → coleção (ex.: `monitor`, `collection4`) → **produto**.
8. No produto, a grid mostra as **unidades** (bandas p/ imagem multibanda; imagens p/ ImageCollection). Marque as desejadas.
9. Clique em **Sincronizar** e processe as etapas pendentes.

> **Células recolhíveis:** todas as células de código comecam com `#@title`
> (ex.: `#@title Etapa 1: Export`). No Colab, o título vira um cabeçalho
> clicável que recolhe/expande o codigo — o texto de introducao permanece visivel.

## Navegação (pais → tema → colecao → produto)

A Interface abre a árvore `pais → tema → colecao → produto` (guias até coleção +
dropdown de produto). Cada produto tem sua **grid de unidades**: para imagem
**multibanda**, uma linha por banda (ex.: `fire_frequency_1985_2025`); para
**ImageCollection**, uma linha por imagem (ex.: `2024_07`).

As celulas de processamento (Export/Mosaico/Vetorizacao/Upload/Publicar) sempre
atuam no **produto ativo + unidades selecionadas**.

```python
COUNTRIES = ["brasil", "indonesia"]   # codigos do OBJ (abas de pais na UI)
```

## Configuracao (OBJ)

A fonte de verdade e o `config.OBJ`: `OBJ[pais][tema][colecao] = [produtos]`, onde
cada produto tem `product` (nome curto), `assetid` (asset GEE de origem), `type`
(`byte`/`int16`/`float32` — dtype do mosaico), `vectorize` (vetorizacao/upload GEE,
apenas `annual_burned` das colecoes + `monthly_burned` do monitor) e `visible`.
Adicionar/ocultar/remover colecoes e feito pela celula **"Adicionar coleção"**
(form) ou pelos helpers `config.add_collection` / `set_product_visible` /
`remove_collection`.

## Adicionando novos dados (edite e versione o `config.py`)

Como o catalogo e um subconjunto validado (fluxo **beta/experimental**), novos
dados devem entrar com cuidado:

1. **Prefira editar o `config.py` localmente e versionar a mudanca** (commit/PR
   neste repo) — assim a nova colecao/produto fica permanente para toda a equipe.
2. O form **"Adicionar coleção"** da UI altera apenas o `OBJ` **em memoria**
   (sessao atual); serve para experimentar, mas nao persiste.
3. Siga o schema de produto: `product`, `assetid`, `type` (`byte`/`int16`/`float32`),
   `vectorize`, `visible`.
4. **Valide antes de escalar**: rode primeiro 1 pais/1 mes, confira COG, vetor e
   link publico — so depois amplie o escopo.

## Colunas da grid (etapas)

| Etapa | Coluna | Quando vira **OK** | Celula |
|-------|--------|--------------------|--------|
| 1 | Export | tiles no GCS (`temp/`) | Export |
| 2 | Mosaico | COG montado | Mosaico |
| 3 | Vetor GCS | ZIP vetorial no GCS | Vetorizacao |
| 4 | Vetor GEE | FeatureCollection no GEE | Upload GEE |
| 5 | Publico mosaico | COG espelhado no `mapbiomas-public` | Publicar mosaico |
| 6 | Publico vetor | ZIP espelhado no `mapbiomas-public` | Publicar vetor |
| 7 | Clean temp | tiles de `temp/` removidos apos consolidacao | Limpar temp |

A legenda da grid traz a dica **MISS → OK** com a celula que resolve cada coluna.
O cabeçalho de cada coluna mostra o numero da **Etapa**.

## Links de download nos badges OK

Badges **OK** de algumas colunas viram links (apenas quando a etapa esta OK):

- **Mosaico** → baixa o COG: `https://storage.googleapis.com/mapbiomas-fire/{...}/monthly_burned-{country}_{Y}_{M}.tif`
- **Vetor GCS** → baixa o ZIP: `https://storage.googleapis.com/mapbiomas-fire/{...}/monthly_burned-{country}_{Y}_{M}.zip`
- **Vetor GEE** → clique copia o asset ID (`projects/mapbiomas-{country}/assets/FIRE/MONITOR/...`)
- **Publico mosaico** → baixa o COG do bucket publico: `https://storage.googleapis.com/mapbiomas-public/{...}`
- **Publico vetor** → baixa o ZIP do bucket publico: `https://storage.googleapis.com/mapbiomas-public/{...}`
- **Export** (tiles) e **Clean temp** ficam sem link.

Os buckets `mapbiomas-fire` e `mapbiomas-public` sao leitura publica (links diretos).

## Forcar reprocessamento (FORCE por etapa)

Por padrao, cada etapa **pula** (SKIP) as unidades que ja estao OK. Para refazer
uma etapa, ative a variavel `FORCE_<ETAPA>` na propria celula (default `False`):

| Etapa | Variavel | Com `True` |
|-------|----------|-----------|
| 1. Export | `FORCE_EXPORT` | exclui os tiles de `temp/` da unidade **antes** de exportar |
| 2. Mosaico | `FORCE_MOSAIC` | exclui o COG existente e remonta |
| 3. Vetorizacao | `FORCE_VECTOR` | exclui o ZIP existente e revetoriza |
| 4. Upload GEE | `FORCE_UPLOAD` | exclui o asset GEE existente e re-uploada |
| 5. Publicar mosaico | `FORCE_PUBLISH_MOSAIC` | sobrescreve o COG no publico |
| 6. Publicar vetor | `FORCE_PUBLISH_VECTOR` | sobrescreve o ZIP no publico |
| 7. Limpar temp | — | idempotente |

Para reprocessar uma unidade ja completa, selecione-a na grid e rode a etapa com a
`FORCE_<ETAPA> = True`.

## Visual

O UI é autossuficiente: todos os componentes tem **fundo e cores explicitos** de
alto contraste, entao fica legivel independente do tema do Colab. Badges **OK** de
download aparecem como **`🔗 OK`** (sublinhado com outline sutil).

## Selecionar um ano ou unidades especificas

- **Filtro por ano**: o dropdown de ano na UI restringe a grid as unidades do ano
  escolhido. Os botoes **Selecionar Pendentes** e **Selecionar Todos** valem
  apenas para as unidades visiveis no filtro.
- **Unidade por unidade**: marque/desmarque os checkboxes da grid normalmente.
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
       │         projects/mapbiomas-public/assets/{country}/fire/monitor/{product_vectors}
       │
       ▼  [5] Publish mosaic (publish.py) → COG no publico .../{product}/
       │
       ▼  [6] Publish vector (publish.py) → ZIP no publico .../{product_vectors}/
       │
       ▼  [7] Clean temp (publish.py) → apaga tiles temp/ dos meses consolidados
```

## Padrao de caminhos

**GCS (bucket de processamento):** `gs://{bucket}/initiatives/{country}/fire/monitor/{product}`

| Recurso | Path |
|---------|------|
| ImageCollection (origem) | `projects/mapbiomas-public/assets/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1` |
| Tiles GCS | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1/temp/` |
| Mosaicos GCS | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_v1/` |
| Vetores GCS (ZIP) | `gs://mapbiomas-fire/initiatives/{country}/fire/monitor/mapbiomas_fire_monthly_burned_vectors_v01/` |
| Vetores GEE | `projects/mapbiomas-public/assets/{country}/fire/monitor/mapbiomas_fire_monthly_burned_vectors_v01/` |
| Publico (espelho) | `gs://mapbiomas-public/initiatives/{country}/fire/monitor/...` |

Os vetores sao publicados numa **pasta irma da ImageCollection de entrada** (Etapa 1),
no mesmo projeto `mapbiomas-public`, sob `{country}/fire/monitor`. A Etapa 4 cria a
pasta se nao existir e deixa **tudo publico** (`all_users_can_read`): a ACL e aplicada
na pasta (assets novos herdam) e, opcionalmente, por asset via `make_vectors_public`
(rodar apos as tasks do GEE concluirem).

Paises suportados: `brazil`, `indonesia`. Novos paises entram no dict `COUNTRIES`
em `config.py`.

## Convencoes de nomes

- Tiles: `fire_monitor_v1_monthly_burned_{country}_{YYYY}_{MM}XXXXXXXXXX-XXXXXXXXXX.tif`
- Mosaico: `monthly_burned-{country}_{YYYY}_{MM}.tif`
- Vetor (zip): `monthly_burned-{country}_{YYYY}_{MM}.zip`

## Export leve (Byte 0/1 + nodata no mosaico)

A exportacao grava tiles **Byte 0/1 puro** (sem `selfMask`, sem banda de mascara):
0 = sem fogo, 1 = fogo. Uma unica banda mantem os tiles leves.

O **mosaico** e gerado como **Byte 0/1 com `0 = nodata`** (`-a_nodata 0`) e
`COMPRESS=DEFLATE (ZLEVEL=9) + PREDICTOR=2`: tiles 100% oceano ficam quase de
graca via mascara interna do COG, e o DEFLATE comprime melhor que o LZW em dados
0/1 — resultado bem menor (ex.: Indonesia ~16-25 MB por mes, antes ~75 MB).

A vetorizacao usa `gdal_polygonize -mask` e ignora o pixel 0 (nodata) normalmente.

Para comparar codecs num mes real, rode o script
`python -m export_and_vectorization.benchmark_compression --country indonesia --year 2024 --month 7`
(LZW vs DEFLATE vs ZSTD; opcional `--blocksize`).

## Publicacao (etapas 3, 4 e 7)

`publish.py` expoe funcoes incrementais e idempotentes; com `ui=` informado,
todas iteram sobre os contextos afetados pela selecao multi-painel:

1. **`publish_mosaic_all(ui=...)`** — copia COGs (`.../{product}/*.tif`) para o bucket
   publico (valida tamanho apos a copia).
2. **`publish_vector_all(ui=...)`** — copia vetores ZIP (`.../{product_vectors}/*.zip`)
   para o bucket publico (so produtos vetorizaveis).
3. **`cleanup_temp_selected(ui)`** — apaga os tiles de `temp/` das unidades
   **selecionadas na grid** (libera espaco; tiles eram o maior volume
   intermediario). Nao ha condicionais com as demais etapas: a ordem de
   execucao e apenas uma sugestao de fluxo. A delecao e restrita ao padrao de
   tiles em `temp/` — nunca alcanca COGs, ZIPs, bucket publico ou GEE.

`publish_all()` encadeia as duas publicacoes. Pode rodar periodicamente para pegar
as unidades que faltaram.

## Dependencias

- **GDAL**: `gdalbuildvrt`, `gdal_translate`, `gdal_polygonize.py`
- **Python**: `gcsfs`, `earthengine-api`, `geopandas`, `rasterio`, `psutil`, `ipywidgets`
- **Google Cloud**: autenticacao via `google.colab.auth`
- **Google Earth Engine**: autenticacao via `ee.Authenticate()`

## Dados ja processados

Unidades ja completas (as 7 etapas: export + mosaico + vetor GCS + vetor GEE +
publico mosaico + publico vetor + clean temp) aparecem como **OK** na interface
e sao ignorados durante o processamento. Apenas meses novos ou incompletos sao
processados. Como o COG so existe apos export+mosaico, meses com `temp/` ja
limpo continuam marcadas como export OK.
