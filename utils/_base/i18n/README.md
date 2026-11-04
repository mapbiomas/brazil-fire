# i18n

Shared interface texts in three languages: `pt-BR.json`, `en.json`, `es.json`. The three files have exactly the same keys.

```json
{ "lang": "es", "months": ["Enero", "..."], "strings": { "bulletin.title": "Boletín del Monitor de Fuego" } }
```

- Language codes follow `M_lang.py` (`network/catalog/production/src/core/`): `pt`, `en`, `es`. `fr` and `id` are specific to the Fire Monitor pipeline and are not included.
- First version: titles and labels already used in `fire_monitor_highlights` (PT and ES), translated to EN. The existing HTML was **not** migrated yet.
- EN and ES need review by a native speaker.
