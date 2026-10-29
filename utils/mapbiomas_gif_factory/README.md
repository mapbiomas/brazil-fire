# MapBiomas GIF Factory — piloto

Protótipo independente para criar GIFs no Google Colab a partir de imagens multibanda do Earth Engine. Começa com Fogo coleção 5 e DF; o exemplo precisa de validação visual e de acesso antes de uso em produção. Não representa o catálogo completo do MapBiomas.

## Como executar

1. Abra `utils/mapbiomas_gif_factory/notebooks/piloto.ipynb` do repositório `mapbiomas/brazil-fire` no Colab.
2. Na primeira célula configure um projeto Google Cloud habilitado para Earth Engine. Execute as células em ordem. Opcionalmente fixe `REPO_REF` em um commit validado.
3. Edite as listas de camadas e territórios, ou escolha o exemplo experimental na interface. Confira os nomes das bandas no asset; o exemplo usa 2023 e 2024.
4. Selecione as saídas e execute. Por padrão apenas o GIF fica em `/content/mapbiomas_gif_factory_output`.
5. Para envio automático ou posterior, monte o Drive pela célula indicada e use a pasta sugerida ou outra de sua preferência. Na guia de envio, atualize a lista e selecione os arquivos desejados. Arquivos existentes no destino não são sobrescritos.

O armazenamento `/content` é temporário e desaparece quando a sessão do Colab termina. O envio posterior exige que a sessão e seus arquivos ainda existam. Cópias com falha são reportadas e podem ser novamente selecionadas.

## Escopo e limitações

- Somente `ee.Image` multibanda, com bandas explicitamente informadas, visualização simples e legenda fornecida pelo usuário. ImageCollection, RGB e produtos que exigem processadores próprios ainda não são suportados.
- A legenda do exemplo vem da configuração existente da Fábrica de GIFs; os parâmetros, nomes de bandas e resultado ainda precisam ser testados no Earth Engine.
- Frames brutos são thumbnails recortados e visualizados pelo Earth Engine; preparados são atualmente uma cópia RGB desses thumbnails; finais acrescentam título e legenda. Esta implementação não reproduz todos os elementos cartográficos da Fábrica original.
- A geração ocorre sequencialmente; não há retomada automática de processamento ou manifesto persistente. Cada disparo cria uma pasta própria. O envio automático ocorre a cada arquivo finalizado, mas erros de Drive não interrompem a geração.
- Não há integração automática com o catálogo de referência. Acrescentar coleções e legendas requer revisão e testes individuais.

## Desenvolvimento

`pip install -e .` e `python -m unittest discover -s tests`. O teste de ponta a ponta requer credenciais Earth Engine, acesso aos assets e uma sessão Colab.
