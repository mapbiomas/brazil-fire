# MapBiomas GIF Factory — piloto

Protótipo independente para criar GIFs no Google Colab a partir de imagens multibanda do Earth Engine. Começa com Fogo coleção 5 e biomas brasileiros; o exemplo precisa de validação visual e de acesso antes de uso em produção. Não representa o catálogo completo do MapBiomas.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/utils/mapbiomas_gif_factory/notebooks/piloto.ipynb)

## Como executar

1. Abra `utils/mapbiomas_gif_factory/notebooks/piloto.ipynb` do repositório `mapbiomas/brazil-fire` no Colab.
2. Na primeira célula, o projeto Earth Engine padrão é `mapbiomas-brazil`. Se você não tiver acesso a ele, altere `GEE_PROJECT` para outro projeto habilitado ao qual tenha acesso. Execute as células em ordem. Opcionalmente fixe `REPO_REF` em um commit validado.
3. Edite as listas de camadas e territórios e encontre-as na guia Personalizado; exemplos experimentais também aparecem na guia Brasil. As bandas são descobertas automaticamente; para limitar anos de uma camada personalizada, use `start_year` e/ou `end_year` no objeto da camada.
4. Selecione as opções na interface e execute a célula seguinte, `ui.start_batch()`. Por padrão apenas o GIF fica em `/content/mapbiomas_gif_factory_output`.
5. A primeira célula tenta montar o Drive para envio automático ou posterior. Se não quiser usá-lo, comente o bloco de montagem; caso a montagem falhe, ainda é possível gerar arquivos em `/content`. Use a pasta sugerida ou outra de sua preferência. Na guia de envio, atualize a lista e selecione os arquivos desejados. Arquivos existentes no destino não são sobrescritos.

O armazenamento `/content` é temporário e desaparece quando a sessão do Colab termina. O envio posterior exige que a sessão e seus arquivos ainda existam. Cópias com falha são reportadas e podem ser novamente selecionadas.

## Configurações personalizadas

As células `LAYERS` e `TERRITORIES` do notebook são listas Python explícitas. Edite-as e execute novamente a célula da interface. As camadas aparecem na guia **Personalizado**; os territórios podem ser usados com qualquer camada selecionada. Os exemplos de Fogo 4 e MATOPIBA são experimentais: confirme acesso e resultado no Earth Engine.

Uma camada de imagem multibanda tem o seguinte formato:

```python
LAYERS = [{
    "id": "minha_camada",                # identificador único: letras, números, _ ou -
    "name": "Minha camada",              # nome exibido
    "theme": "Meu tema",                  # opcional
    "collection": "Minha coleção",        # opcional
    "asset": "projects/.../minha_imagem", # ee.Image multibanda
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [
        {"label": "Não observado", "color": "fdfdfd"},
        {"label": "Fogo", "color": "800000"},
    ],
}]
```

`id`, `name`, `asset`, `visualization` e `legend` são obrigatórios. `theme` e `collection` são opcionais. A fábrica descobre as bandas; para limitar o período desta camada, adicione `"start_year": 2020` e/ou `"end_year": 2023`. Sem estes campos, usa todas as bandas disponíveis. Para selecionar bandas manualmente, inclua opcionalmente `"bands": ["nome_da_banda"]`. A ordem e as cores de `legend` devem corresponder à paleta de `visualization`: o renderizador usa a paleta para as cores e os rótulos da legenda. Não use este exemplo binário como legenda genérica para produtos categóricos: classes não consecutivas exigem correspondência validada entre valores, cores e rótulos.

A última célula é o único disparo do lote; aguarde sua conclusão antes de executá-la novamente. A primeira célula ativa o gerenciador de widgets do Colab.

Um território usa uma FeatureCollection e pode ter filtro de igualdade:

```python
TERRITORIES = [{
    "id": "meu_territorio",             # identificador único
    "name": "Meu território",
    "asset": "projects/.../meu_limite", # ee.FeatureCollection
    # Opcional; remova se quiser usar toda a coleção:
    "filter": {"field": "nome_do_campo", "value": "valor"},
}]
```

Confira a existência e as permissões do asset e, quando houver filtro, o nome e o valor do atributo. O projeto não valida automaticamente se um território intersecta os dados da camada.

## Organização dos resultados

Cada disparo cria uma pasta com o nome de entrega escolhido e data/hora, sem alterar resultados de execuções anteriores. Dentro dela, os arquivos ficam em `Brasil/Fogo/Colecao_5/Produto/Territorio/`, com biomas do mesmo produto como pastas irmãs. Camadas do notebook aparecem em `Personalizado/Tema/Colecao/Produto/Territorio/`. Se tema ou coleção não forem informados, são usados `Sem_tema` e `Sem_colecao`.

Cada território contém somente as pastas selecionadas: `GIFs`, `Colagens` e/ou `Frames/Mapa_basico`, `Frames/Mapa_com_ano_e_escala`, `Frames/Mapa_completo`. A entrega inclui um `LEIA-ME.txt` com os produtos, territórios e períodos concluídos. O arquivo também pode ser selecionado para envio posterior ao Drive. O nome da pasta de entrega recebe data e hora automaticamente para evitar misturar execuções; arquivos existentes no destino não são sobrescritos.

## Escopo e limitações

- Somente `ee.Image` multibanda, com descoberta automática de bandas, visualização simples e legenda fornecida pelo usuário. Bandas com ano ou intervalo identificável são ordenadas temporalmente; quando há ambiguidade a execução é interrompida com instruções. `bands` é opcional para seleção avançada explícita. ImageCollection, RGB e produtos que exigem processadores próprios ainda não são suportados. A correspondência de códigos e cores para classes não consecutivas precisa de validação específica; não presumir que uma paleta curta corresponde aos valores de classe.
- A legenda do exemplo vem da configuração existente da Fábrica de GIFs; os parâmetros, nomes de bandas e resultado ainda precisam ser testados no Earth Engine.
- Os biomas usam os assets `projects/ee-ipam/assets/IBGE/limite_biomas_IBGE_2026` e, para Mata Atlântica, `users/wallacesilva/mata_atlantica_`, conforme a configuração do projeto original. A conta autenticada precisa ter permissão de leitura; disponibilidade e filtros devem ser confirmados no Earth Engine.
- O pipeline cartográfico original é incluído no pacote; os níveis correspondem a `frames_pure`, `frames_clean` e `frames_maps`. O pipeline original redimensiona os arquivos em `frames_pure` após o download. A equivalência visual entre plataformas ainda deve ser testada, inclusive fontes e acesso aos assets.
- A geração ocorre sequencialmente em uma thread; não há retomada automática de processamento ou manifesto persistente. Cada disparo cria uma pasta própria. Os resultados selecionados são copiados e enviados ao Drive após cada combinação, não durante seu processamento. Erros de Drive não interrompem a geração.
- A estimativa apresentada é apenas para o download dos frames da combinação atual, calculada com a média observada; não inclui composição, GIF ou envio ao Drive.
- A barra de lote conta combinações processadas, inclusive falhas; a barra de download conta somente frames baixados. O relógio atualiza durante as esperas, e tentativas e erros são registrados no painel.
- Não há integração automática com o catálogo de referência. Acrescentar coleções e legendas requer revisão e testes individuais.
- A célula personalizada contém um exemplo editável de Fogo 4 anual. Uso e cobertura 10/10.1 e os demais produtos de Fogo 4 ainda não integram o catálogo permanente; precisam de validação de simbologia e transformações.
- Os territórios personalizados são independentes das iniciativas. Os exemplos MATOPIBA e MATOPIBA no Cerrado vêm das configurações locais e não tiveram acesso verificado nesta versão. Troque os assets na célula se necessário.

## Próximas etapas de expansão

1. Conferir assets, nomes de bandas e legenda oficial das coleções de uso e cobertura 10 e 10.1. Implementar mapeamento explícito valor → cor → rótulo para classes não consecutivas.
2. Validar os produtos da coleção 4 de Fogo individualmente, incluindo os que exigem decodificação; incorporar gradualmente ao catálogo permanente.
3. Reaproveitar configurações de Peru Fogo 2, Paraguai Fogo 1 e degradação do projeto original, verificando assets, territórios, legendas e processadores antes de publicar cada preset.
4. Adicionar ImageCollection temporal: por exemplo Atmosfera com uma imagem por ano, selecionando por `year` quando existir ou por `system:time_start`, e detectando anos ausentes e duplicados. Mosaicos, RGB e séries mensais requerem estratégias separadas.
5. Verificar interseção entre território e cobertura dos dados antes da geração; a seleção atual permite qualquer combinação, mas não garante que contenha pixels válidos.

## Desenvolvimento

`pip install -e .` e `python -m unittest discover -s tests`. O teste de ponta a ponta requer credenciais Earth Engine, acesso aos assets e uma sessão Colab.
