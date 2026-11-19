# MapBiomas Fire · Showcase

## 🔗 Open the app: **[https://mapbiomas.github.io/brazil-fire/](https://mapbiomas.github.io/brazil-fire/)**

Root page of this repository's GitHub Pages. Two tabs: **Apps** (cards with a screenshot of every published app, filters by tag, and the main Colab notebooks and tools) and **Repository map** (the [`fire_repo_map`](../fire_repo_map/) app embedded). Interface in Portuguese, English and Spanish.

## 🗂️ Folder structure

```
utils/fire_showcase/
├── README.md
└── app/
    ├── index.html   # the page (static, no build); content lists APPS and NOTEBOOKS
    ├── thumbs/      # 640x400 WebP screenshots, one per app
    └── logos/       # copies of utils/_base/logos (the page opens standalone)
```

The Pages workflow copies `app/` to the root of the site (`_site/`). Links in the page are relative to that root (`dashboard/`, `repo_map/`...), so open it through the published site or a locally assembled `_site/`.

Links: `?lang=pt|en|es` sets the language, `?tag=<tag>` filters the apps, `#map` opens the map tab.

## ➕ Add an app

1. Publish it in `.github/workflows/pages.yml` (copy its files to `_site/<app>/`).
2. Add an item to `APPS` in `app/index.html`: `url`, `thumb`, `src` (folder in the repo), `tags` and the title `t` and description `d` in `pt`, `en` and `es`.
3. Add the screenshot, for example with headless Chrome:
   ```
   chrome --headless=new --hide-scrollbars --window-size=1280,800 --virtual-time-budget=15000 --screenshot=shot.png <app URL>
   ```
   then resize to 640x400 and save as `app/thumbs/<app>.webp`.
4. Add the folder to `PAGES_APPS` in `utils/fire_repo_map/build_repo_map.py`.

Notebooks and tools follow the same idea in `NOTEBOOKS` (`path` to the `.ipynb`, or to a folder with `folder: true`).
