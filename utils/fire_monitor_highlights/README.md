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
├── logos/                        # official logos: Monitor_EN.png, Monitor_ES.png, Monitor_PT.png
└── src/
    ├── brazil/                   # Portuguese · API plataforma.monitorfogo.mapbiomas.org
    │   ├── boletim-fogo.html     # bulletin of the month + Destaque dinâmico
    │   └── Monitor_PT.png
    └── peru/                     # Spanish · API fogo-peru.geodatin.com
        ├── boletin-fuego.html    # bulletin of the month only
        └── Monitor_ES.png
```

Each country has its own folder with an independent HTML file, because the APIs, territory types, available data and language differ. To add a country, copy the closest folder and adapt the API and texts. Details are in [CLAUDE.md](CLAUDE.md) (in Portuguese).

---

Data: [MapBiomas Fire](https://www.monitorfogo.mapbiomas.org/).