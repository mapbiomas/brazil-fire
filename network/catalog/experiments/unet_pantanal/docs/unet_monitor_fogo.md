# U-Net para área queimada no Monitor do Fogo (Pantanal, Sentinel-2 mensal)

Roteiro para slides: cada seção `##` é um slide. **Visual** sugere a figura; **Fala** são notas para quem apresenta (não vão no slide). Os campos **[preencher]** saem dos relatórios do notebook (`reports/` no GCS). Situação em novembro de 2026.

- **Parte A – Contexto** (slides 1–4): por que testar e o que se pergunta.
- **Parte B – Proposta** (slides 5–13): método e critérios de decisão.
- **Parte C – O que foi feito** (slides 14–19): notebook, versões, primeiro resultado e próximos passos.

---

# Parte A – Contexto

## 1. Capa

**U-Net para mapeamento de área queimada**
Teste no Monitor do Fogo: Pantanal, Sentinel-2, mosaicos mensais – MapBiomas Fogo

**Visual:** mosaico mensal Sentinel-2 do Pantanal em falsa-cor (SWIR1/NIR/RED), com cicatrizes do mês visíveis.

## 2. Fluxo atual do Monitor do Fogo

- Mapeamento **mensal** de cicatrizes de fogo com Sentinel-2 de 10 m (catálogo version_04, campanha `MONITOR_01`):
  1. M1/M2: mosaico mensal de NBR mínimo.
  2. M3: polígonos de fogo / não fogo de cada mês.
  3. M4: rede MLP **pixel a pixel**.
  4. M5: probabilidade.
  5. M6–M9: pós-classificação no GEE.

## 3. Por que testar outra abordagem

**Mensagem:** hoje a classificação olha cada pixel sozinho; a cicatriz tem forma e vizinhança.

- O MLP não sabe se o vizinho também queimou nem qual é o formato da mancha. Isso pode gerar:
  - ruído "sal e pimenta" dentro e fora das cicatrizes;
  - bordas irregulares;
  - confusão com alvos escuros pontuais (água, sombra).
- Parte disso é corrigida depois, nos filtros da pós-classificação (M7).

**Visual:** zoom de uma cicatriz classificada hoje, com pixels isolados e bordas serrilhadas.

**Fala:** não é crítica ao MLP, que funciona bem; a pergunta é se olhar a vizinhança melhora o mapa já na classificação.

## 4. A pergunta e o escopo do piloto

> Uma U-Net, treinada com **as mesmas amostras** e **os mesmos mosaicos** do catálogo, mapeia a área queimada mensal do Pantanal melhor que o MLP atual?

| Item | Escolha |
|---|---|
| Campanha | Brasil, Monitor do Fogo (`MONITOR_01`) |
| Região | Pantanal (`Pantanal_r1`) |
| Período | mensal; primeiro mês: outubro de 2026 (`2026_10`) |
| Sensor | Sentinel-2, 10 m, mosaico mensal de NBR mínimo |
| Onde roda | Google Colab com GPU T4 gratuita, dados no Cloud Storage |

- **Nenhuma coleta nova:** manter os dados iguais permite atribuir a diferença ao método.
- **Por que o Pantanal:** muito fogo, já coberto pelo Monitor, e um desafio conhecido: **água escura e áreas alagadas** que lembram cicatriz.

---

# Parte B – Proposta

## 5. O que é uma U-Net

```
MLP:    1 pixel (bandas)              →  1 probabilidade
U-Net:  recorte (chip) 256 × 256 px   →  mapa 256 × 256 de probabilidades
        (≈ 2,6 × 2,6 km com pixel de 10 m)
```

- Rede convolucional para **segmentação**: diz, pixel a pixel, o que é cada coisa.
- **Descida:** resume o recorte e aprende o contexto (forma, textura, vizinhança).
- **Subida:** reconstrói o mapa na resolução original.
- **Atalhos** entre as metades preservam detalhes finos, como a borda da cicatriz.

**Visual:** diagrama em "U".

## 6. Visão geral do fluxo

```
 Catálogo (já existe)        Teste U-Net (novo)                            Catálogo (já existe)
 M1/M2 mosaicos  ─┐
                  ├──►  chips e rótulos ─► treino ─► avaliação
 M3    amostras  ─┘                           │
                                              └─► mapa da região (formato M5) ─►  M6+ pós-classificação
```

- Notebook e pasta próprios no repositório; o fluxo atual **não muda**.
- Tudo o que o teste grava fica em `UNET_01/` no bucket.

## 7. Dados de entrada

| Dado | Origem |
|---|---|
| Mosaico mensal | `CATALOG_01/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/{mês}/COG/`, um COG por banda (refletância × 100, 10 m) |
| Bandas | blue, green, red, nir, swir1, swir2 + `dayOfYear` (pixel válido onde > 0) |
| Índices (opcionais) | NBR, NBR2, MIRBI, NDVI, BAI, EVI, calculados das bandas por uma única função (treino e mapa) |
| Amostras | `MONITOR_01/LIBRARY_SAMPLES/samples_01_pantanal_r1_2026_10.csv` (fogo = 1 / não fogo = 0) |
| Região | asset `regioes_biomas_col2`, `bioma + "_r" + id` → `Pantanal_r1` |

**Normalização:** corte nos percentis 1–99 e padronização, com estatísticas **só dos chips de treino**.

## 8. Das amostras aos chips

1. **Polígonos usados:** os dos CSVs escolhidos, do mês e da região (`Pantanal_r1`), com classe 0/1 e geometria válida.
2. **Centros:** até 4 pontos sorteados por polígono; mesma quantidade de fogo e não fogo.
3. **Rótulo de cada pixel:**
   - **1:** dentro de polígono de fogo;
   - **0:** dentro de polígono de não fogo;
   - **255:** sem informação. Não entra no treino nem nas métricas.
4. Onde fogo e não fogo se sobrepõem, vale fogo (registrado).

**Visual:** `reports/chips_preview_{mês}.png` – 16 chips em falsa-cor com os rótulos.

**Fala:** os polígonos não cobrem o chip inteiro, e tudo bem: a rede só aprende onde há informação.

## 9. Divisão espacial antes de tudo

- A região é dividida em **blocos de 25 km**; os blocos com amostras são sorteados (semente fixa) em treino 70% / validação 15% / teste 15%, garantindo as duas classes em cada grupo.
- Descartes:
  - chips que cruzam blocos de grupos diferentes;
  - chips com menos de 1% de pixels rotulados;
  - chips com mais de 50% sem dado.
- O código confere que **nenhum chip de teste encosta num bloco de treino**.

**Visual:** mapa dos blocos (`splits/blocks_{mês}.gpkg`) colorido por grupo.

**Fala:** chips vizinhos são quase iguais; sem essa separação o teste mediria memória, não generalização.

## 10. Rede e treino

```
entrada 256×256×C
  ├─ 32 ─┐                                  ┌─ 32 → saída 256×256×1 (sigmoide)
  │   ├─ 64 ─┐                          ┌─ 64
  │   │   ├─ 128 ─┐              ┌─ 128
  │   │   │   ├─ 256 ─┐   ┌─ 256
  │   │   │   │      512 (fundo)
```

- **Perda:** entropia cruzada + Dice, só nos pixels rotulados (o Dice compensa o desequilíbrio fogo × não fogo).
- **Treino:**
  - Adam (1e-4), lotes de 8, até 60 épocas;
  - precisão mista na GPU;
  - espelhamentos e rotações nos chips de treino;
  - parada antecipada pelo IoU de fogo na validação.

## 11. Mapa da região inteira

- **Janela deslizante:** cada janela lê 512 + 2×16 px e grava só o centro, então não aparecem emendas.
- **Saída com 3 bandas nomeadas**, um COG por mês na grade do mosaico:
  - `burned`: 1 = queimado, 0 = não queimado, no limiar de melhor F1 da validação (ex.: 0,63);
  - `prob`: probabilidade × 1000 (0–1000);
  - `doy`: dia do ano (`dayOfYear`), onde `prob` > 0;
  - −9999 fora da região.
- O limiar usado vai para o relatório e para a propriedade `threshold` da imagem no GEE; `prob` permite escolher outro corte depois.

## 12. Comparação justa com o MLP

Três modelos, **os mesmos pixels de teste**, limiar de cada um escolhido na validação (melhor F1):

| Modelo | O que vê | Limiar (val) | F1 | IoU | Omissão | Comissão |
|---|---|---|---|---|---|---|
| A – MLP | bandas, pixel | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |
| B – MLP | bandas + índices, pixel | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |
| C – U-Net | bandas + índices, chip 256×256 | [preencher] | [preencher] | [preencher] | [preencher] | [preencher] |

- **A × B:** ganho dos índices.
- **B × C:** ganho do contexto espacial, com os mesmos atributos.
- O MLP repete a rede do M4 (7-14-7-14-7, 7000 iterações em lotes de 1000) e é treinado com os pixels dos mesmos chips de treino.

Fonte: `reports/comparison_{meses}.json` (campo `markdown` com a tabela pronta).

## 13. Como vamos decidir

Critérios combinados **antes** de ver o resultado (para discussão com a equipe):

1. **Métrica:** C melhora F1 e IoU de fogo em relação a B.
2. **Erros:** sem muito mais comissão (área queimada a mais).
3. **Visual:** bordas mais limpas e menos confusão com água, em áreas escolhidas antes.
4. **Operação:** tempo de treino e classificação viável no Colab.

**[definir com a equipe]** Diferença mínima de F1/IoU que justifica seguir.

---

# Parte C – O que foi feito

## 14. Notebook em passos

| Passo | O que faz | Escolha do usuário |
|---|---|---|
| 0 | Preparação, login, configuração do projeto | – |
| 1 | Tabela de meses × bandas com mosaico no GCS | – |
| 2 | Escolhe o treino, confere as amostras e cria os chips | amostras, bandas, índices, versão |
| 3 | Normalização (só chips de treino) | – |
| 4 | Treino da U-Net | épocas, batch |
| 5 | Métricas de teste | – |
| 6 | Classifica a região inteira | versão do modelo, meses a classificar |
| 7 | Comparação com o MLP do catálogo | – |

- Layout igual aos notebooks do catálogo: guia recolhido no topo e uma interface (widgets) em cada passo que pede escolha.
- **Publicação no GEE (passo 6.3):** como o M6 do production, cada mapa vira uma imagem `{região}_{mês}` na coleção da versão (`…/MONITOR_01/UNET_01/{bioma}/{versão}`), com propriedades de modelo, região e período. Já vêm marcados os meses que ainda não estão no GEE.
- **Treino e classificação são separados:** treina em 2026_10 e mapeia qualquer mês com mosaico.
- **Resumos didáticos em português** em cada passo, com diagnóstico automático (✅ ⚠️ ❌ 💡), curvas de treino e matriz de confusão.

## 15. Versões de treino e pastas

- Cada treino é uma **versão** (como o M4 da development), organizada por **bioma e região**:

```
UNET_01/
  pantanal/
    pantanal_r1/
      training_001_pantanal_r1_sentinel2/
        config.json  chips/  stats/  models/  reports/  predict/
  cerrado/
    cerrado_r1/ ...
```

- O bioma e a região saem do `REGION` (`Pantanal_r1` → `pantanal/pantanal_r1/`).
- **ID** sugerido pelo arquivo de amostras (`samples_01_…` → 001); **shortname** pela região (`pantanal_r1`).
- A aba **Versões existentes** reabre um treino para avaliar, classificar ou comparar.
- Os chips ficam salvos na versão e só são refeitos quando mudam amostras, bandas ou região.

## 16. Primeiro resultado (versão inicial: 4 bandas, sem índices)

- **Chips:** treino 23 / validação 13 / teste 15 (1,5 milhão de pixels de treino).

| Limiar | Precisão | Recall | F1 | IoU |
|---|---|---|---|---|
| 0,63 (validação) | 0,991 | 0,435 | 0,605 | 0,434 |
| 0,50 | 0,205 | 0,996 | – | – |

- **No limiar da validação:** quase tudo o que marca é fogo, mas deixa de marcar mais da metade do fogo real (omissão alta).
- **Em 0,5:** marca quase tudo como fogo; as probabilidades ainda estão pouco separadas.
- **Leitura:** esperado com só **23 chips de treino**. Mostra que o pipeline funciona de ponta a ponta, não a qualidade final.

## 17. Garantias do experimento

- [x] Bandas na mesma grade (senão, para com mensagem clara).
- [x] Rótulos 1 / 0 / 255, com figura de inspeção.
- [x] Nenhum chip de teste sobre bloco de treino (checagem no código).
- [x] Normalização só com o treino; mesma função de índices no treino e no mapa.
- [x] Inferência sem emendas, na grade do mosaico.
- [x] Semente fixa; caminhos só pela configuração.
- [x] Nada do catálogo é alterado; a escrita fica restrita a `UNET_01/`.
- [x] Testes automáticos no repositório.

## 18. Cuidados e limitações

- **Água escura × cicatriz:** conferir chips sobre lagoas e áreas alagadas.
- **Rótulos só dentro dos polígonos:** se a rede errar bordas, rotular por completo alguns chips.
- **Poucas amostras por mês:** juntar meses no treino (um modelo, um mapa por mês).
- **Nuvens e sombras** no mosaico mensal também lembram cicatriz.
- **Métrica por pixel:** compara modelos; a validação do mapa final segue com os dados de referência habituais.

## 19. Próximos passos

1. Rodar a versão com **6 bandas + índices** e comparar com a de 4 bandas.
2. Rodar o passo 7 (**U-Net × MLP**) e preencher a tabela do slide 12.
3. **Mais amostras e mais meses** (incluindo o pico da estação de fogo): maior ganho esperado.
4. Reaproveitar chips entre versões com os mesmos dados; mostrar polígonos que ficaram sem chip.
5. Passar os mapas pela pós-classificação (M6+) e comparar a área queimada final.
6. Se funcionar: expandir para outros biomas, uma pasta por bioma e região.

---

### Onde está cada coisa

| Item | Local |
|---|---|
| Pasta do experimento | branch `classification-unet`, `network/catalog/experiments/unet_pantanal/` |
| Notebook Colab | `notebooks/unet_monitor_pantanal.ipynb` |
| Código | `src/` |
| Guia técnico (inglês) | `README.md` |
| Saídas no bucket | `gs://mapbiomas-fire/sudamerica/brazil/CATALOG_01/MONITOR_01/UNET_01/{bioma}/{região}/` |
