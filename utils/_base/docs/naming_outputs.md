# Nomes de pastas e arquivos de saída

## Decisão proposta

Nomes de **saída** novos usam **inglês, `snake_case`**, para manter a consistência com o resto do repositório e facilitar o uso pela rede internacional.

Exemplo: `Brasil_Fogo_Colecao_5/Area_queimada_anual/` → `brazil_fire_collection_5/annual_burned_area/`.

## Critério

- Saída que sai da equipe (outros países, GitHub, entrega pública): inglês.
- Arquivo de uso interno e temporário: pode ficar em português, mas sem acentos nem espaços.
- Nunca: acentos, espaços, maiúsculas misturadas.
- País/coleção/produto sempre nesta ordem: `<country>_<initiative>_<collection>/<product>/`.

## Em aberto (decisão da Vera)

- Aprovar a regra acima.
- `mapbiomas_gif_factory` hoje gera `Brasil_Fogo_Colecao_5`. Renomear quebra bibliotecas já criadas; na Fase 2, aceitar os dois nomes na leitura ou migrar com script.
