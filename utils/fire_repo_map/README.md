# MapBiomas Fire · Repository map

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/repo_map/](https://mapbiomas.github.io/brazil-fire/repo_map/)**

Browsable map of this repository for users and developers: the main areas, every folder with the description taken from its `README.md`, its files (with Colab links for notebooks), what kind of code it holds, and which folders are published apps or legacy/production versions. Interface in Portuguese, English and Spanish.

## 🗂️ Folder structure

```
utils/fire_repo_map/
├── README.md
├── build_repo_map.py   # scans the repository and writes app/repo_map.json
└── app/
    ├── index.html      # the app (static, no build)
    ├── repo_map.json   # generated folder tree
    └── logos/          # copies of utils/_base/logos (the app opens standalone)
```

| Item | Purpose |
|---|---|
| `build_repo_map.py` | Walks the repository (standard library only) and records, per folder: README title and first paragraph, files, file kinds, status (legacy, production, active, pilot) and published app URL. |
| `app/index.html` | Area overview, searchable folder tree and detail panel. Links: `#<path>` opens a folder, `?lang=pt\|en\|es` sets the language, `?embed=1` hides the header (used by the showcase page). |

## ▶️ Run locally

1. `python utils/fire_repo_map/build_repo_map.py` (from the repository root)
2. `python -m http.server -d utils/fire_repo_map/app` and open http://localhost:8000

The Pages workflow (`.github/workflows/pages.yml`) runs the script on every push to `main`, so the published map always matches the current tree. The committed `repo_map.json` is only for local use.

## ➕ Keep the map useful

- **Folder descriptions** come from the first paragraph of each folder's `README.md`. A folder without README shows "No README in this folder". Writing one is the way to describe it.
- **Status** comes from the README title (`[LEGACY]`, `[ACTIVE]`, `Production`, `pilot`, `exploratory`).
- **Published apps** are listed in `PAGES_APPS` in `build_repo_map.py`; keep it in sync with the workflow.
- **Top-level area texts** (the cards at the top) are in `I18N.*.areas` in `app/index.html`, in the three languages. Add an entry when a new top-level folder is created.
