# U-Net no Monitor do Fogo – andamento (Pantanal, Sentinel-2 mensal)

Roteiro para slides: cada seção `##` é um slide. Situação em novembro de 2026.

---

## 1. O que é

- Uma **U-Net** para mapear **área queimada** no Monitor do Fogo.
- Teste piloto: **Pantanal** (`Pantanal_r1`), mês **2026_10**, campanha `brazil.monitor_01`.
- Usa os **mesmos mosaicos (M2) e amostras (M3)** do catálogo, lidos do Cloud Storage.
- A saída tem o **formato do M5**: probabilidade × 1000 + `dayOfYear` (int16, nodata −9999).
- O código fica em `network/catalog/experiments/unet_pantanal/`, na branch `classification-unet`, e roda no Google Colab com GPU T4 gratuita.

## 2. Por que uma U-Net

- O MLP do catálogo (M4) classifica **um pixel por vez**.
- A U-Net vê um **recorte da imagem (chip)** de 256 × 256 px (2,56 km). Ela aprende também a **forma, a borda e a textura** da cicatriz.
- Pergunta do teste: com **as mesmas amostras e os mesmos mosaicos**, a U-Net mapeia melhor?

## 3. Dados de entrada

| Dado | Origem |
|---|---|
| Mosaico mensal (minnbr) | `CATALOG_01/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/{mês}/COG/`, um COG por banda |
| Bandas de 2026_10 | blue, green, red, nir, swir1, swir2 + dayOfYear (máscara de pixel válido) |
| Índices (opcionais) | nbr, nbr2, mirbi, ndvi, bai, evi, calculados das bandas |
| Amostras | `MONITOR_01/LIBRARY_SAMPLES/samples_01_pantanal_r1_sentinel_2026_10.csv` (polígonos fogo = 1 / não-fogo = 0) |
| Região | asset `regioes_biomas_col2`, `region_name = bioma + "_r" + id` → `Pantanal_r1` |

## 4. Fluxo do notebook (passos 0–7)

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

O layout segue os notebooks do catálogo: guia recolhido no topo e uma interface (widgets) em cada passo que precisa de escolha.

## 5. Versões de treino (como o M4 da development)

- Cada treino é uma **versão** com pasta própria, organizada por bioma e região: `UNET_01/{bioma}/{região}/training_{id}_{shortname}_sentinel2/` (ex.: `UNET_01/pantanal/pantanal_r1/training_001_pantanal_r1_sentinel2/`).
- A pasta guarda chips, normalização, modelo, métricas, mapas e um `config.json` para reabrir a versão depois.
- O **ID** é sugerido a partir do arquivo de amostras (`samples_01_…` → 01). O **shortname** é sugerido a partir da região (`pantanal_r1`).
- A aba **Versões existentes** reabre um treino para avaliar, classificar ou comparar.
- Treino e classificação são **separados**: dá para treinar em 2026_10 e mapear outros meses que tenham mosaico.

## 6. Como as amostras viram chips

1. **Polígonos usados:** os dos CSVs escolhidos, do mês e da região (`Pantanal 60`), com classe 0/1 e geometria válida.
2. **Pontos:** até 4 pontos sorteados por polígono; cada ponto é o centro de um chip.
3. **Rótulo:** cada chip tem a imagem e um rótulo: 1 fogo, 0 não-fogo, 255 desconhecido (o treino ignora).
4. **Divisão espacial:** blocos de 25 km divididos em treino / validação / teste (70/15/15) **antes** dos chips, para evitar vazamento entre as divisões. O sorteio dos blocos garante as duas classes em cada divisão.
5. **Descartes:** chips que cruzam blocos de divisões diferentes, com menos de 1% de pixels rotulados ou com mais de 50% de nodata.

Os chips ficam salvos no GCS e só são refeitos quando mudam as amostras, as bandas ou a região.

## 7. Treino e avaliação

- **Arquitetura:** U-Net com filtros 32-64-128-256 e fundo 512.
- **Treino:** perda BCE + Dice só nos pixels rotulados, precisão mista na GPU e parada antecipada pelo IoU de fogo na validação.
- **Limiar:** o de melhor F1 na **validação**. O teste é reportado nesse limiar e em 0,5.
- **Classificação:** janela deslizante (512 px + 16 px de sobreposição) sobre a região, sem emendas visíveis.
- **Comparação justa com o MLP (M4, camadas 7-14-7-14-7):**
  - mesmos chips de treino;
  - limiar escolhido na validação;
  - nota nos **mesmos pixels de teste**.

## 8. Primeiro resultado (versão anterior: 4 bandas, sem índices)

- **Chips:** treino 23 / validação 13 / teste 15 (1,5 milhão de pixels de treino).

| Limiar | Precisão | Recall | F1 | IoU |
|---|---|---|---|---|
| 0,63 (validação) | 0,991 | 0,435 | 0,605 | 0,434 |
| 0,50 | 0,205 | 0,996 | – | – |

- **No limiar da validação:** quase tudo o que o modelo marca é fogo, mas ele deixa de marcar mais da metade do fogo real (omissão alta).
- **Em 0,5:** marca quase tudo como fogo. As probabilidades ainda estão pouco separadas.
- **Leitura:** resultado inicial, esperado com **23 chips de treino**. Mostra que o pipeline funciona de ponta a ponta. Não mostra ainda a qualidade final.

## 9. O que foi construído além do modelo

- **Resumos didáticos em português** em cada passo: o que o passo faz, tabelas e um diagnóstico automático (✅ ⚠️ ❌ 💡).
- **Curvas de treino** (perda e IoU) e **matriz de confusão** do teste.
- **Verificações:**
  - inventário de bandas e grade;
  - amostras por classe e área;
  - aviso quando uma divisão fica sem fogo.
- **Testes automáticos** no repositório (59 testes).

## 10. Próximos passos

1. Rodar a versão nova com **6 bandas + índices** e comparar com a de 4 bandas.
2. Rodar o passo 7 (**U-Net × MLP**) nos mesmos pixels de teste.
3. **Mais amostras e mais meses**: com 23 chips de treino, esse é o maior ganho esperado.
4. Reaproveitar os chips entre versões com os mesmos dados, sem refazer quando só muda o treino.
5. Mostrar quantos polígonos ficaram sem nenhum chip.
6. Subir os mapas ao GEE e passar pela pós-classificação (M6+).
7. Opção de idioma (pt / en / es) nos textos.
