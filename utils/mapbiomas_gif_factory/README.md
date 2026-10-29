# MapBiomas GIF Factory — piloto

Protótipo para produzir GIFs cartográficos de imagens multibanda do Earth Engine no Google Colab. O catálogo inicial inclui produtos experimentais de Fogo do Brasil e Peru; assets, permissões, legendas e resultados precisam ser conferidos antes do uso em produção.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/utils/mapbiomas_gif_factory/notebooks/piloto.ipynb)

## Como usar

1. Abra o notebook acima. Na primeira célula, `GEE_PROJECT` usa `mapbiomas-brazil`; altere para um projeto Earth Engine ao qual você tenha acesso, se necessário.
2. Execute as células em ordem. Edite `LAYERS` e `TERRITORIES` se quiser acrescentar itens personalizados; as bandas de imagens multibanda são descobertas automaticamente.
3. Na interface, escolha **uma iniciativa** (Brasil ou Peru) e navegue pelas guias **Produtos**, **Territórios**, **Saídas e destino** e **Resultados**. Um lote nunca mistura iniciativas: os produtos e territórios exibidos pertencem à iniciativa ativa, e trocar de iniciativa limpa as seleções. Territórios personalizados continuam disponíveis em qualquer iniciativa.
4. Selecione pelo menos um produto, um território e os tipos de arquivo. Países inteiros e biomas individuais estão disponíveis para Brasil e Peru.
5. Escolha se a biblioteca principal será local ou, marcando envio automático, o Drive. Use **Verificar disponibilidade** para consultar o estado das combinações selecionadas. Execute a célula `ui.start_batch()` para iniciar.
6. Se a biblioteca principal já tiver resultados completos e compatíveis, eles não serão reprocessados. Caso haja conflito, o processo avisa e não sobrescreve. Para enviar arquivos locais mais tarde, selecione-os na seção de envio; os metadados necessários são copiados junto.

O armazenamento `/content` desaparece quando a sessão Colab termina. Os dados temporários não são enviados ao Drive automaticamente.

## Pastas e disponibilidade

A biblioteca é permanente durante a sessão local, ou no Drive se este for escolhido como destino. Não há uma pasta de entrega por execução:

```text
mapbiomas_gif_factory/
├── Brasil_Fogo_Colecao_5/Area_queimada_anual/Amazonia/
│   ├── GIFs/
│   ├── Colagens/
│   ├── Frames/
│   └── metadados/
└── Peru_Fuego_Coleccion_2/Area_quemada_anual/Peru/
```

Os intermediários ficam visíveis em `/content/temp/<iniciativa_tema_colecao>/<produto>/<territorio>/<período>/<execução>/processamento/`. Eles permanecem na sessão para inspeção, inclusive após falhas; o projeto não apaga automaticamente pastas antigas.

- **Disponível:** os tipos de saída selecionados têm metadados compatíveis e todos os arquivos registrados conferem.
- **Parcial:** há metadados compatíveis, mas falta algum arquivo ou tipo de saída; pode ser necessário gerar novamente no temporário. Arquivos existentes não são substituídos.
- **Não gerado:** não há resultado registrado para a seleção.
- **Verificar:** metadados ausentes/inválidos, arquivos alterados ou configuração diferente. Não há sobrescrita automática; examine a pasta no destino.

Com envio automático habilitado, o Drive é a referência. Apagar um arquivo no Drive faz a próxima verificação indicar resultado parcial, mesmo que exista cópia local. Com envio automático desativado, a biblioteca local é a referência. Um envio posterior seletivo leva os metadados e pode aparecer como parcial até todos os arquivos de um tipo serem copiados.

Uma série completa usa `completo` nos metadados e não recebe período extra no nome do GIF/colagem. Um intervalo parcial contínuo acrescenta, por exemplo, `_2020-2023`; uma seleção irregular recebe um identificador, e os nomes exatos das bandas ficam nos metadados. Frames parciais ficam em subpasta própria. A conferência verifica configuração, lista de bandas, tamanho e SHA-256 dos arquivos; ela não detecta uma alteração no conteúdo do asset Earth Engine se o endereço e os nomes das bandas permanecerem iguais.

## Camadas e territórios personalizados

As células `LAYERS` e `TERRITORIES` são listas Python explícitas. Altere-as e execute novamente a interface. Para uma camada `ee.Image` multibanda:

```python
LAYERS = [{
    "id": "minha_camada",  # identificador único: letras, números, _ ou -
    "name": "Minha camada",
    "theme": "Meu tema",             # opcional
    "collection": "Minha coleção",   # opcional
    "asset": "projects/.../minha_imagem",
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"},
               {"label": "Fogo", "color": "800000"}],
    # "start_year": 2020,  # opcionais; omitidos = todas as bandas
    # "end_year": 2023,
}]
```

`id`, `name`, `asset`, `visualization` e `legend` são obrigatórios. `bands` é opcional para seleção explícita. As cores da legenda devem corresponder à paleta; para classes não consecutivas, não presumir que uma paleta curta represente corretamente os valores. Os exemplos de Fogo 4 e MATOPIBA são experimentais.

### Eixo da sequência: tempo ou profundidade

Por padrão a animação percorre os anos das bandas. Produtos de Solo organizados por camada de profundidade usam outro eixo, com `axis` e `depth_band`:

```python
LAYERS = [{
    "id": "argila_col3",
    "name": "Fração de argila",
    "theme": "Solo",
    "collection": "Coleção 3",
    "asset": "projects/.../mapbiomas_brazil_collection3_soil_clay_fraction_v1",
    "axis": "depth",          # "time" (padrão) ou "depth"
    "depth_band": "clay_fraction",
    "visualization": {"min": 0, "max": 100, "palette": ["ffffcc", "800026"]},
    "legend": [{"label": "0", "color": "ffffcc"}, {"label": "100", "color": "800026"}],
}]
```

Nesse eixo, a fábrica lê a propriedade `depth` de cada imagem da coleção, monta a sequência e rotula os frames como `0–10 cm`. `start_year`/`end_year` são rejeitados, e colagens e GIFs especiais por ano são desligados. Os produtos de profundidade do catálogo estão desabilitados: ainda falta validar legenda e visualização de cada variável.

Para séries cumulativas, como `fire_frequency_1985_1999`, o rótulo do frame usa o ano final quando todas as janelas começam no mesmo ano; caso contrário, o intervalo completo é mantido.

Um território é uma `ee.FeatureCollection`. O filtro opcional usa `ee.Filter.inList`:

```python
TERRITORIES = [{
    "id": "meus_biomas",
    "name": "Cerrado e Caatinga",
    "asset": "projects/ee-ipam/assets/IBGE/limite_biomas_IBGE_2026",
    "filter": {"field": "Bioma", "values": ["Cerrado", "Caatinga"]},
}]
```

Para apenas um valor, pode usar `"value": "Cerrado"` no lugar de `values`. Omita `filter` para usar toda a coleção. Confirme acesso ao asset e os valores do atributo. O projeto ainda não verifica automaticamente se o território intersecta os dados.

## Limitações e próximos passos

- O adaptador suporta `ee.Image` multibanda em eixo de tempo e coleções de Solo em eixo de profundidade. ImageCollection anual (como Atmosfera), mosaicos, RGB e produtos que exigem processadores específicos ainda requerem adaptação. Peru, Frequência de fogo e Solo são experimentais e devem ser conferidos no Colab.
- Bandas com ano ou intervalo identificável são ordenadas temporalmente; seleção ambígua é interrompida, em vez de produzir uma sequência presumida.
- A biblioteca não importa automaticamente os resultados gerados pela versão anterior; arquivos antigos sem metadados compatíveis aparecem como `Verificar` se ocuparem o novo caminho.
- A estimativa de tempo cobre apenas o download atual; a barra de lote conta combinações processadas, inclusive falhas.
- A verificação de disponibilidade consulta metadados e arquivos; ela não valida ausência de interseção entre território e cobertura dos dados.
- Próximas expansões: legendas verificadas de uso e cobertura 10/10.1, produtos com processadores próprios, produtos de profundidade de Solo e ImageCollection anual, incluindo Atmosfera.

## Desenvolvimento

`pip install -e .` e `python -m unittest discover -s tests`. A validação visual e de ponta a ponta requer Earth Engine, permissões dos assets, Google Drive e uma sessão Colab.
