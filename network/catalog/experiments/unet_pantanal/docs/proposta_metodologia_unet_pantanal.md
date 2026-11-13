# Proposta de teste: U-Net para área queimada no Pantanal (2024)

Roteiro para montar a apresentação da **metodologia proposta**, antes dos resultados. Cada seção `##` é um slide. Em cada um há:
- **Mensagem:** a frase que o slide precisa passar;
- o conteúdo do slide;
- **Visual:** sugestão de figura ou diagrama;
- **Fala:** notas para quem apresenta (não vão no slide).

---

## 1. Capa

**U-Net para mapeamento de área queimada**
Proposta de teste no Pantanal, 2024
MapBiomas Fogo, catálogo Landsat

**Visual:** mosaico Landsat do Pantanal em falsa-cor (SWIR1/NIR/RED), com cicatrizes de 2024 visíveis.

---

## 2. Por que testar outra abordagem

**Mensagem:** hoje a classificação olha cada pixel sozinho; a cicatriz de fogo tem forma e vizinhança.

- O fluxo atual classifica **pixel a pixel** com uma rede MLP (M4/M5).
- O MLP não sabe se o pixel vizinho também queimou nem qual é o formato da mancha.
- Isso pode gerar:
  - ruído "sal e pimenta" dentro e fora das cicatrizes;
  - bordas irregulares;
  - confusão com alvos escuros pontuais, como água e sombra.
- Hoje parte disso é corrigida depois, nos filtros da pós-classificação (M7).

**Visual:** zoom de uma cicatriz com a classificação atual mostrando pixels isolados e bordas serrilhadas.

**Fala:** não é uma crítica ao MLP, que funciona bem; a pergunta é se olhar a vizinhança melhora o mapa já na classificação.

---

## 3. A pergunta do teste

**Mensagem:** com os mesmos dados de hoje, uma rede que vê a vizinhança mapeia melhor?

> Uma U-Net, treinada com **as mesmas amostras** e **os mesmos mosaicos** do catálogo, mapeia a área queimada do Pantanal em 2024 melhor que o MLP atual?

- **Mesmas amostras:** polígonos de fogo e não fogo já coletados (M3).
- **Mesmos mosaicos:** mosaico anual Landsat de NBR mínimo (M1/M2).
- **Nenhuma coleta nova** para este teste.

**Fala:** manter os dados iguais é o que permite atribuir a diferença ao método, e não a dados novos.

---

## 4. Escopo do piloto

**Mensagem:** um bioma e um ano, para responder rápido.

| Item | Escolha |
|---|---|
| País / campanha | Brasil, `COLLECTION_06` |
| Bioma | Pantanal |
| Ano | 2024 |
| Sensor | Landsat, 30 m, mosaico anual de NBR mínimo |
| Onde roda | Google Colab com GPU |
| Onde ficam os dados | Google Cloud Storage do MapBiomas Fogo |

**Por que o Pantanal:**
- bioma com muito fogo em 2024;
- tem um desafio conhecido: **água escura e áreas alagadas** que lembram cicatriz. Bom teste para o contexto espacial.

**Visual:** mapa do Brasil com o Pantanal destacado.

---

## 5. O que é uma U-Net

**Mensagem:** em vez de um pixel, a rede recebe um recorte da imagem e devolve um mapa.

```
MLP:    1 pixel (bandas)              →  1 probabilidade
U-Net:  recorte de 128 × 128 pixels   →  mapa de 128 × 128 probabilidades
        (≈ 3,8 × 3,8 km)
```

- Rede convolucional muito usada para **segmentação de imagens**: dizer, pixel a pixel, o que é cada coisa.
- Tem duas metades:
  - **descida:** resume o recorte e aprende o contexto (forma, textura, vizinhança);
  - **subida:** reconstrói o mapa na resolução original.
- **Atalhos** ligam as duas metades e preservam detalhes finos, como a borda da cicatriz.

**Visual:** diagrama em "U" (entrada à esquerda, saída à direita, atalhos horizontais).

**Fala:** o nome vem do formato do diagrama.

---

## 6. Visão geral do fluxo proposto

**Mensagem:** o teste reaproveita a entrada do catálogo e devolve a saída no formato que o catálogo já usa.

```
 Catálogo (já existe)            Teste U-Net (novo)                         Catálogo (já existe)
 ─────────────────────           ─────────────────────────────────────      ─────────────────────
 M1/M2  mosaicos     ─┐
                      ├──►  1 recortes e rótulos ─► 2 treino ─► 3 avaliação
 M3     amostras     ─┘                                    │
                                                           └─► 4 mapa do Pantanal ─►  M7/M8 pós-classificação
                                                                  (formato M5)
```

- Tudo roda num **notebook próprio**, numa **pasta separada** do repositório.
- O fluxo atual **não é alterado**.
- Tudo o que o teste grava fica numa pasta própria no bucket (`UNET_01/`).

---

## 7. Passo 1: recortes e rótulos

**Mensagem:** os polígonos viram rótulos dentro de recortes da imagem.

- Centros dos recortes sorteados **dentro dos polígonos** (até 4 por polígono).
- Mesma quantidade de recortes de fogo e de não fogo.
- Cada pixel do recorte recebe um rótulo:
  - **1:** dentro de polígono de fogo;
  - **0:** dentro de polígono de não fogo;
  - **sem rótulo:** fora dos polígonos ou sem dado. **Não entra** no treino nem nas métricas.

**Visual:** um recorte em falsa-cor com os polígonos sobrepostos (vermelho = fogo, azul = não fogo, transparente = sem rótulo).

**Fala:** os polígonos não cobrem o recorte inteiro, e tudo bem: a rede só aprende onde há informação.

---

## 8. Passo 1: separação espacial de treino e teste

**Mensagem:** o teste é feito em áreas que a rede nunca viu.

- O Pantanal é dividido em **blocos de 25 km**.
- Os blocos com amostras são sorteados em:
  - **treino:** 70%;
  - **validação:** 15%;
  - **teste:** 15%.
- Recortes que cruzam blocos de grupos diferentes são descartados.
- O código confere que **nenhum recorte de teste encosta num bloco de treino**.

**Visual:** mapa do Pantanal com a grade de blocos colorida por grupo.

**Fala:** recortes vizinhos são quase iguais. Se treino e teste se misturassem, a nota seria alta porque a rede "lembra" da área, não porque aprendeu.

---

## 9. Passo 1: o que a rede recebe

**Mensagem:** as bandas do mosaico mais índices de fogo conhecidos.

- **5 bandas:** azul, vermelho, NIR, SWIR1, SWIR2.
- **6 índices:** NBR, NBR2, MIRBI, NDVI, BAI, EVI.
- A mesma função calcula os índices no treino e no mapa final.
- Normalização: corte nos percentis 1–99 e padronização, com estatísticas **só do treino**.

**Visual:** grade com as 11 camadas de um mesmo recorte.

---

## 10. Passo 2: treino

**Mensagem:** a rede é treinada para acertar a forma da cicatriz, não só a contagem de pixels.

- **Perda:** entropia cruzada + Dice, só nos pixels com rótulo.
  - O Dice ajuda quando há muito mais não fogo que fogo.
- **Aumento de dados:** espelhamentos e rotações dos recortes de treino.
- **Parada antecipada:** o treino para quando a validação deixa de melhorar e guarda o melhor modelo.
- Semente fixa: o teste pode ser repetido com o mesmo resultado de divisão.

**Visual:** curva de aprendizado ilustrativa (treino × validação).

---

## 11. Passo 3: comparação justa com o MLP

**Mensagem:** três modelos, os mesmos pixels de teste.

| Modelo | O que vê | Para que serve |
|---|---|---|
| **A** MLP | 5 bandas, pixel a pixel | referência: mesma rede do M4 |
| **B** MLP | 5 bandas + 6 índices, pixel a pixel | mede o ganho dos índices |
| **C** U-Net | 5 bandas + 6 índices, recorte 128 × 128 | mede o ganho do contexto espacial |

- **A × B:** quanto os índices ajudam.
- **B × C:** quanto a vizinhança ajuda, com os mesmos atributos.
- Métricas da classe fogo: **F1, IoU, omissão e comissão**.
- O limiar de cada modelo é escolhido na validação; o teste é reportado nesse limiar e em 0,5.

**Fala:** o MLP de referência é treinado com os pixels dos mesmos recortes de treino, para que a única diferença seja o modelo.

---

## 12. Passo 4: mapa do Pantanal inteiro

**Mensagem:** o resultado sai no mesmo formato do catálogo e pode seguir para a pós-classificação.

- Classificação em **janelas deslizantes** com sobreposição: só o centro de cada janela é gravado, então não aparecem emendas.
- Saída igual à do M5:
  - banda 1: probabilidade × 1000;
  - banda 2: dia do ano da queimada;
  - mesma grade do mosaico.
- Pode entrar no M7/M8 sem mudanças. O corte de probabilidade continua sendo escolhido no GEE.

**Visual:** mapa de probabilidade do Pantanal, com zoom lado a lado MLP × U-Net.

---

## 13. Como vamos decidir

**Mensagem:** critérios combinados antes de ver o resultado.

Sugestão para discussão com a equipe:

1. **Métrica:** a U-Net (C) melhora F1 e IoU de fogo em relação ao MLP com os mesmos atributos (B) nos pixels de teste.
2. **Erros:** a melhora não vem às custas de muito mais comissão (área queimada a mais).
3. **Inspeção visual:** bordas mais limpas e menos confusão com água, em áreas escolhidas antes.
4. **Operação:** tempo de treino e de classificação viável no Colab.

**[definir com a equipe]** Diferença mínima de F1/IoU que justifica seguir.

**Fala:** definir o critério antes evita interpretar o resultado a favor do método novo.

---

## 14. Riscos e cuidados

**Mensagem:** já sabemos onde olhar.

- **Água escura × cicatriz:** lagoas e áreas alagadas do Pantanal. Vamos conferir os recortes sobre corpos d'água.
- **Rótulos só dentro dos polígonos:** a rede não vê bordas rotuladas. Se errar bordas, o próximo passo é rotular por completo alguns recortes.
- **Um bioma e um ano:** o resultado vale para o Pantanal em 2024; generalizar exige outros testes.
- **Métrica por pixel:** compara modelos nos pixels de teste. A validação do mapa final segue com os dados de referência habituais.

---

## 15. Próximos passos

1. Exportar os mosaicos e as amostras do Pantanal 2024 (em andamento).
2. Rodar o notebook: recortes, treino, avaliação e mapa.
3. Preencher a tabela A × B × C e fazer a inspeção visual.
4. Discutir o resultado com a equipe usando os critérios do slide 13.
5. Se a U-Net ganhar:
   - passar a saída pelo M7 e comparar a área queimada final;
   - testar outros anos (por exemplo, 2019 e 2020) e outro bioma.

---

### Anexo: onde está cada coisa

| Item | Local |
|---|---|
| Pasta do experimento | branch `classification-unet`, `network/catalog/experiments/unet_pantanal/` |
| Notebook Colab | `unet_pantanal/notebooks/unet_pantanal_2024.ipynb` |
| Guia técnico (inglês) | `unet_pantanal/README.md` |
| Roteiro com resultados | `unet_pantanal/docs/apresentacao_unet_pantanal.md` |
| Saídas no bucket | `gs://mapbiomas-fire/sudamerica/brazil/CATALOG_01/COLLECTION_06/UNET_01/` |
