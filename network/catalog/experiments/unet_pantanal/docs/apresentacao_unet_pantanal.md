# U-Net para área queimada: teste no Monitor do Fogo (Pantanal, Sentinel-2 mensal)

Roteiro para montar a apresentação. Cada seção `##` corresponde a um slide ou bloco. Os campos marcados com **[preencher]** saem dos relatórios gerados pelo notebook, na pasta `reports/` do GCS.

---

## 1. Contexto

- **Monitor do Fogo:** mapeamento mensal de cicatrizes de fogo com Sentinel-2 de 10 m.
- **Fluxo atual (catálogo version_04, campanha `MONITOR_01`):**
  1. M1/M2: mosaico mensal de NBR mínimo.
  2. M3: polígonos `fire` / `notFire` de cada mês.
  3. M4: rede MLP pixel a pixel.
  4. M5: probabilidade.
  5. M6–M9: pós-classificação no GEE.
- **Limite do MLP:** classifica cada pixel isolado e não usa a forma nem a textura da cicatriz.
- **Pergunta do teste:** uma rede que vê a vizinhança, a U-Net, melhora o mapa usando **as mesmas amostras e os mesmos mosaicos**?

## 2. Objetivo e escopo

- Bioma piloto: **Pantanal**. Período: **mensal**, começando em outubro de 2026 (`2026_10`). Campanha: `brazil.monitor_01`.
- A U-Net é um **experimento independente**, numa pasta própria (`network/catalog/experiments/unet_pantanal/`). Ela só lê os dados do catálogo; o fluxo MLP não muda.
- A saída é **compatível** com a pós-classificação atual: probabilidade × 1000 e `dayOfYear`, no mesmo formato dos tiles do M5.
- Tudo roda no Google Colab com GPU, lendo e gravando no Google Cloud Storage.

## 3. Dados de entrada (reaproveitados)

| Dado | Origem | Formato |
|---|---|---|
| Mosaico mensal | M1/M2 (`LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/2026_10/COG/`) | 1 COG por banda, refletância × 100 (uint8), 10 m; válido onde `dayOfYear > 0` |
| Amostras | M3 (`MONITOR_01/LIBRARY_SAMPLES/*.csv`) | polígonos com `fire` = 1/0, `region`, `period` (mês) |
| Região | `MONITOR_01/AUXILIARY/regiones_fuego_brazil_v1` (`region_nam = pantanal`) | limite para os blocos e para a inferência |

Bandas usadas: blue, red, nir, swir1, swir2.

**[preencher]** Número de polígonos e área por classe, que estão em `reports/chips_2026_10.json` → `samples.by_class` (um arquivo por mês).

## 4. Do pixel ao recorte (chip)

```
MLP:    1 pixel (5 bandas)          →  1 probabilidade
U-Net:  recorte 256 × 256 pixels    →  mapa 256 × 256 de probabilidades
        (≈ 2,6 × 2,6 km com pixel de 10 m)
```

- Os centros dos recortes são sorteados **dentro dos polígonos**, até 4 por polígono.
- O número de centros de fogo e de não fogo é igual em cada conjunto.
- **Rótulo de cada pixel do recorte:**
  - **1:** dentro de polígono de fogo.
  - **0:** dentro de polígono de não fogo.
  - **255:** sem informação (fora dos polígonos ou sem dado). Esses pixels são ignorados no treino e nas métricas.
- Onde fogo e não fogo se sobrepõem, vale fogo. Esses casos ficam registrados.

**Figura:** `reports/chips_preview_2026_10.png`, com 16 recortes em falsa-cor SWIR1/NIR/RED e o rótulo sobreposto (vermelho = fogo, azul = não fogo).

## 5. Divisão espacial antes de tudo

- A área é dividida em **blocos de 25 km**. Os blocos que têm amostras são sorteados com semente fixa em:
  - treino: 70%
  - validação: 15%
  - teste: 15%
- Um recorte que cruza blocos de conjuntos diferentes é descartado.
- Uma checagem no código garante que **nenhum recorte de teste toca um bloco de treino**.
- **Por quê:** recortes vizinhos são quase iguais. Sem essa separação, o teste mediria memória do modelo, não generalização.

**Figura:** mapa dos blocos (`splits/blocks_2026_10.gpkg`) com as cores treino / validação / teste.

**[preencher]** Blocos e recortes por conjunto, além dos descartados: `reports/chips_2026_10.json` → `blocks`, `chips`, `dropped`.

Se houver vários meses em `PERIODS`, os recortes de todos eles treinam **um único modelo**, e cada mês ganha seu próprio mapa.

## 6. Atributos espectrais

Além das 5 bandas, a rede recebe 6 índices calculados por **uma única função**, a mesma no treino e na inferência:

| Índice | Fórmula (refletância 0–1) |
|---|---|
| NBR | (NIR − SWIR2) / (NIR + SWIR2) |
| NBR2 | (SWIR1 − SWIR2) / (SWIR1 + SWIR2) |
| MIRBI | 10·SWIR2 − 9,8·SWIR1 + 2 |
| NDVI | (NIR − RED) / (NIR + RED) |
| BAI | 1 / ((0,1 − RED)² + (0,06 − NIR)²) |
| EVI | 2,5·(NIR − RED) / (NIR + 6·RED − 7,5·BLUE + 1) |

**Normalização:**
1. Os valores são cortados nos percentis 1–99.
2. Depois é aplicada média e desvio por canal.

As estatísticas vêm **só dos recortes de treino** e ficam salvas em `stats/norm_stats.json`.

## 7. A rede U-Net

```
entrada 256×256×11
  ├─ 32 ─┐                                  ┌─ 32 → saída 256×256×1 (sigmoide)
  │   ├─ 64 ─┐                          ┌─ 64
  │   │   ├─ 128 ─┐              ┌─ 128
  │   │   │   ├─ 256 ─┐   ┌─ 256
  │   │   │   │      512 (fundo)
        descida (convolução + max pooling)   subida (convolução transposta + atalhos)
```

- Os **atalhos (skip connections)** devolvem os detalhes finos, como bordas da cicatriz, que se perdem na descida.
- A rede é totalmente convolucional. Treina com recortes de 256 e classifica janelas maiores.
- Cada convolução usa normalização em lote e ReLU. O treino usa precisão mista na GPU.

## 8. Perda, métricas e treino

- **Perda:** entropia cruzada binária + Dice, calculadas só nos pixels rotulados (0 ou 1).
  - Dice compensa o desequilíbrio entre área queimada e não queimada.
- **Métricas da classe fogo:** IoU, F1, precisão e revocação, todas mascaradas.
- **Treino:**
  - Adam, taxa de aprendizado 1e-4, lotes de 8, até 60 épocas.
  - Para cedo quando a IoU de validação não melhora em 8 épocas e guarda o melhor modelo.
  - Aumento de dados só no treino: espelhamentos e rotações de 90°, aplicados juntos à imagem e ao rótulo.

**Figura:** curvas de perda e IoU por época (`models/unet_monitor_pantanal_v1_history.csv`).

## 9. Classificação do Pantanal inteiro

- **Janela deslizante:** cada janela lê 512 + 2×16 pixels e grava só o centro de 512 × 512.
  - Por isso não aparecem emendas entre as janelas.
- **Saída:** `predict/pantanal_2026_10_prob.tif`, um arquivo por mês.
  - COG com a mesma grade do mosaico.
  - Banda 1: probabilidade × 1000.
  - Banda 2: `dayOfYear` onde a probabilidade é > 0.
  - −9999 fora da região.
- Segue a mesma regra do M5, por isso o resultado entra no M7 (ft00–ft02) e no M8 sem mudanças. O corte final de probabilidade continua sendo escolhido no GEE.

**Figura:** mapa de probabilidade, mais um zoom comparando MLP e U-Net na mesma área.

**[preencher]** Tempo de inferência e pixels válidos: `reports/predict_2026_10.json`.

## 10. Comparação justa com o MLP

As três versões são avaliadas **nos mesmos pixels de teste**. O limiar de cada uma é escolhido na validação, pelo melhor F1.

| Modelo | O que vê | Limiar (val) | F1 | IoU | Omissão | Comissão |
|---|---|---|---|---|---|---|
| A — MLP | 5 bandas, pixel | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |
| B — MLP | 5 bandas + 6 índices, pixel | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |
| C — U-Net | 5 bandas + 6 índices, recorte 256×256 | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |

Fonte: `reports/comparison_2026_10.json` (os meses de `PERIODS`, separados por hífen). O campo `markdown` já traz a tabela pronta, incluindo os valores com limiar 0,5.

- **A × B** mostra o ganho dos índices.
- **B × C** mostra o ganho do contexto espacial, com os mesmos atributos.
- O MLP de referência repete a rede do M4 da version_04:
  - camadas 7-14-7-14-7, sem dropout;
  - 7000 iterações em lotes de 1000;
  - treinada com os pixels rotulados dos mesmos recortes de treino.

## 11. Garantias do experimento

- [x] Todas as bandas na mesma grade. Se não estiverem, o processo para com uma mensagem clara.
- [x] Rótulos 1 / 0 / 255, com figura de inspeção.
- [x] Nenhum recorte de teste sobre bloco de treino (checagem no código).
- [x] Normalização calculada só com o treino.
- [x] Mesma função de índices no treino e na inferência.
- [x] Inferência sem emendas e com a mesma grade do mosaico.
- [x] Semente fixa. Caminhos só pela configuração.
- [x] Nada é apagado nem alterado no GCS. A escrita fica restrita à pasta `UNET_01/`.

## 12. Cuidados e limitações

- **Água escura × cicatriz:** no Pantanal, lagoas e áreas alagadas podem parecer queimadas. Vale conferir os recortes de inspeção sobre corpos d'água.
- **Rótulos só no miolo dos polígonos:** a rede não vê bordas rotuladas. Se ela errar bordas, o próximo passo é **rotular por completo** alguns recortes, sem mudar a arquitetura.
- **Poucos meses e um bioma:** o resultado vale para o Pantanal nos meses testados. Generalizar exige outros meses e regiões.
- **Nuvens e sombras:** o mosaico mensal pode guardar restos de nuvem ou sombra; conferir nos recortes de inspeção.
- **Métrica por pixel:** a comparação usa pixels de teste rotulados. A validação do mapa final continua sendo feita com os dados de referência habituais.

## 13. Próximos passos

1. Rodar o teste completo e preencher a tabela A × B × C.
2. Inspecionar visualmente o mapa da U-Net contra o MLP e o mosaico (bordas, água, nuvens).
3. Se a U-Net ganhar, passar a saída pelo M7 (ft00–ft02) e comparar a área queimada final.
4. Testar outros meses, incluindo o pico da estação de fogo, e outro bioma.
5. Avaliar a rotulagem completa de alguns recortes.

---

### Onde está cada coisa

| Item | Local |
|---|---|
| Pasta do experimento | branch `classification-unet`, `network/catalog/experiments/unet_pantanal/` |
| Código | `unet_pantanal/src/` |
| Notebook Colab | `unet_pantanal/notebooks/unet_monitor_pantanal.ipynb` |
| Resultados | `gs://mapbiomas-fire/sudamerica/brazil/CATALOG_01/MONITOR_01/UNET_01/` |
| Guia técnico (inglês) | `unet_pantanal/README.md` |
