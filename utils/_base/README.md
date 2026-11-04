# utils/_base — padrões compartilhados

Fonte única de logos, paletas de cores, textos de interface (i18n), modelos de documentação e regras de código para as ferramentas de `utils/`. Serve à equipe do Brasil e aos países da rede MapBiomas Fogo.

**Regra de ouro:** não redefina cores, logos ou textos dentro de uma ferramenta. Aponte para esta pasta.

## Estrutura

```
utils/_base/
├── README.md
├── logos/          # Monitor_PT/EN/ES.png, variacoes/ (vazia por enquanto)
├── palette/        # vazia por enquanto (paletas a definir)
├── i18n/           # pt-BR.json, en.json, es.json (mesmas chaves)
├── templates/      # README, CLAUDE e CHANGELOG para novas ferramentas
└── docs/           # padrões de código, política de commit, nomes de saída
```

## Como usar

| Preciso de… | Use |
|---|---|
| logo | `logos/Monitor_<PT|EN|ES>.png` (regras em `logos/README.md`) |
| cor de classe de fogo | CSVs em `utils/legends_mapbiomas_fire/legends/` (paletas em `palette/` ainda por definir) |
| texto de interface | `i18n/<idioma>.json`, chave em `strings` |
| nova ferramenta | copie os arquivos de `templates/` |

## Estado

Fase 1: pasta criada. As três ferramentas ainda **não** consomem estes arquivos (Fase 2). A pasta `palette/` ficou vazia de propósito, para ser preenchida depois.
