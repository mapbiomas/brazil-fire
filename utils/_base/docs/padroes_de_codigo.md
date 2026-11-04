# Padrões de código e documentação

## Nomes

- Pastas e arquivos de ferramentas: inglês, `snake_case` (`fire_monitor_highlights`, `build_legends.py`).
- Exceções herdadas: arquivos de HTML por país podem ter o nome no idioma do boletim (`boletim-fogo.html`, `boletin-fuego.html`); não renomeie sem necessidade.
- Notebooks: `<etapa>_<campanha>_<pais>.ipynb`, como em `network/catalog/`.

## Idioma

| Item | Idioma |
|---|---|
| Código, nomes de variáveis e funções | inglês |
| Comentários | português ou inglês; o que a equipe da ferramenta lê melhor, sem misturar no mesmo arquivo |
| README de ferramenta | inglês se a ferramenta é para a rede internacional; português se é só para a equipe do Brasil |
| Interface para o usuário final | no idioma do público, com textos de `utils/_base/i18n/` |

(Hoje: `fire_monitor_highlights` e `legends_mapbiomas_fire` têm README em inglês, `mapbiomas_gif_factory` em português. A Fase 2 pode alinhar.)

## Onde mora cada documento

- `README.md`: uso, para o usuário. Link da versão publicada na primeira linha.
- `CLAUDE.md`: decisões técnicas, para desenvolvedores e agentes.
- `CHANGELOG.md` (opcional): mudanças visíveis ao usuário.
- Não repita conteúdo entre eles; um aponta para o outro.

## Código

- Biblioteca padrão do Python sempre que possível.
- Cores, logos e textos compartilhados: ler de `utils/_base/`.
- Arquivos gerados dizem no topo que são gerados e por qual script, com teste que confere se estão em dia (como em `legends_mapbiomas_fire`).
- Saídas: ver `naming_outputs.md`.
