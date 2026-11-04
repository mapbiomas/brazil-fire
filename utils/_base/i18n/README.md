# i18n

Textos de interface compartilhados em três idiomas: `pt-BR.json`, `en.json`, `es.json`. Os três têm exatamente as mesmas chaves.

```json
{ "lang": "es", "months": ["Enero", "..."], "strings": { "bulletin.title": "Boletín del Monitor de Fuego" } }
```

- Códigos de idioma alinhados a `M_lang.py` (`network/catalog/production/src/core/`): `pt`, `en`, `es`. Os idiomas `fr` e `id` são específicos do pipeline do Fire Monitor e não entram aqui.
- Primeira versão: títulos e rótulos já usados em `fire_monitor_highlights` (PT e ES) e traduzidos para EN. O HTML existente **não** foi migrado (Fase 2).
- Revisão de falante nativo pendente em EN e ES.
