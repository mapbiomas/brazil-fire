# CLAUDE.md — fire_legends

Notas técnicas. O uso está no `README.md`.

## Regras

- Não fazer commit nem push sem autorização explícita (`utils/_base/docs/politica_commit_push.md`).
- Os CSVs em `legends/` e o `registry.csv` são a fonte de verdade. `LEGENDS.md`, `preview.html` e `app/palettes.json` são **gerados** por `build_legends.py`; nunca editar à mão. O teste `tests/test_legends.py` falha se estiverem desatualizados.

## Arquitetura

- Python (biblioteca padrão) valida e gera os arquivos; o app é um HTML estático (`app/index.html`, derivado de um artefato do claude.ai) que lê `app/palettes.json`.
- Chaves do app: `app_country`/`app_collection`/`app_product` no `registry.csv` formam a chave `country__collection__product` em `palettes.json`.
- O app publicado (GitHub Pages, `/legends/`) é só de consulta; o cadastro de legendas se faz pelos CSVs.

## Publicação

`.github/workflows/pages.yml` copia `app/` para `_site/legends/`. A URL pública não depende do nome da pasta.

## Problemas conhecidos

- Classe 0 de `brazil_col5_annual_burned_area`: o site lista `#fffff` (5 dígitos); confirmar.
- PDF do Brasil Col5, intervalo de retorno: coluna PT deslocada a partir do pixel 20 (corrigido nos CSVs).

## Próximas etapas

- [ ] Registrar cores das coleções que ainda faltam.
- [ ] Gerar QML (QGIS) e código GEE a partir dos CSVs.
