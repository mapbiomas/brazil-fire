## 🔗 Open the bulletin online

| 🇧🇷 Brazil · Portuguese | 🇵🇪 Peru · Spanish |
|:---:|:---:|
| **[Boletim do Monitor do Fogo →](https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/brazil/)** | **[Boletín del Monitor de Fuego →](https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/peru/)** |

_Landing page with both countries: https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/_

**Locally**: open [src/brazil/boletim-fogo.html](src/brazil/boletim-fogo.html) or [src/peru/boletin-fuego.html](src/peru/boletin-fuego.html) in a browser, keeping the logo next to the HTML. No build or install needed.

# Fire Monitor Highlights 🔥

Monthly fire bulletin built from the MapBiomas Fire Monitor API, printable as a PDF (5 A4 landscape sheets).

- **Brazil**: bulletin of the month, plus an interactive panel (*Destaque dinâmico*) for any territory and period.
- **Peru**: bulletin of the month in Spanish. The Peru API only has data since July 2025 (no historical average) and none for May 2026; the page flags this.

## 🗂️ Folder structure

```
utils/fire_monitor_highlights/
├── README.md
├── CLAUDE.md                     # technical documentation
├── .gitignore
└── src/
    ├── brazil/                   # Portuguese · API plataforma.monitorfogo.mapbiomas.org
    │   ├── boletim-fogo.html     # bulletin of the month + Destaque dinâmico
    │   └── Monitor_PT.png
    └── peru/                     # Spanish · API fogo-peru.geodatin.com
        ├── boletin-fuego.html    # bulletin of the month only
        └── Monitor_ES.png
```

The official logos live in [`utils/_base/logos/`](../_base/logos/); each country folder keeps a copy next to its HTML so the page also works when opened locally.

## ➕ Add a country

1. Copy the closest folder in `src/` (for example `src/peru/`) to `src/<country>/`.
2. Adapt the API URL, territory types, language and logo (UI texts: [`utils/_base/i18n/`](../_base/i18n/)).
3. Add the country to the "Assemble site" step of [`.github/workflows/pages.yml`](../../.github/workflows/pages.yml) and to the table at the top of this README.

Each country has its own independent HTML file, because the APIs, territory types, available data and language differ. Details are in [CLAUDE.md](CLAUDE.md) (in Portuguese).

---

Data: [MapBiomas Fire](https://www.monitorfogo.mapbiomas.org/).