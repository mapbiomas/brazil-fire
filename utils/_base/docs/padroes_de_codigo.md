# Padrões de código e documentação

## Nomes

- Pastas e arquivos de ferramentas: inglês, `snake_case`. Ferramentas de `utils/` se chamam `fire_<o_que_faz>` (`fire_monitor_highlights`, `fire_legends`).
- Exceções herdadas: arquivos de HTML por país podem ter o nome no idioma do boletim (`boletim-fogo.html`, `boletin-fuego.html`); não renomeie sem necessidade.
- Notebooks: `<etapa>_<campanha>_<pais>.ipynb`, como em `network/catalog/`.

## Idioma

| Item | Idioma |
|---|---|
| Código, nomes de variáveis e funções | inglês |
| Comentários | português ou inglês; o que a equipe da ferramenta lê melhor, sem misturar no mesmo arquivo |
| **README** | **sempre inglês** |
| Interface para o usuário final | no idioma do público, com textos de `utils/_base/i18n/` |

(Hoje: `mapbiomas_gif_factory` ainda tem README em português e nome fora do padrão `fire_<o_que_faz>`; alinhar quando for migrada.)

## Onde mora cada documento

- `README.md`: uso, para o usuário. Link da versão publicada na primeira linha.
- `CLAUDE.md`: decisões técnicas, para desenvolvedores e agentes.
- `CHANGELOG.md` (opcional): mudanças visíveis ao usuário.
- Não repita conteúdo entre eles; um aponta para o outro.

## Código

- Biblioteca padrão do Python sempre que possível.
- Cores, logos e textos compartilhados: ler de `utils/_base/`.
- Arquivos gerados dizem no topo que são gerados e por qual script, com teste que confere se estão em dia (como em `fire_legends`).
- Saídas: ver `naming_outputs.md`.
