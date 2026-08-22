import ipywidgets as widgets
from IPython.display import display, clear_output

from . import config
from .state import list_months_in_collection, build_state

L = widgets.Layout

_STATUS_CSS = widgets.HTML("""<style>
.mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
.mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
.mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
</style>""")

def _palette():
    return {
        "panel_bg": "#ffffff",
        "panel_border": "#cccccc",
        "header_bg": "#ffffff",
        "header_border": "#333333",
        "title": "#333333",
        "subtitle": "#6c757d",
        "grid_header_bg": "#343a40",
        "grid_header_fg": "#ffffff",
        "row_a": "#fcfcfc",
        "row_b": "#ffffff",
        "date_bg": "#e9ecef",
        "date_fg": "#212529",
        "legend_bg": "#f8f9fa",
        "legend_fg": "#6c757d",
        "hint_bg": "#fffbe6",
        "hint_border": "#ffe58f",
        "hint_fg": "#495057",
        "inst_bg": "#e8f4fd",
        "inst_border": "#bee5eb",
        "inst_fg": "#0c5460",
        "guide_bg": "#ffffff",
        "guide_border": "#dddddd",
        "guide_fg": "#333333",
        "border": "#cccccc",
        "sep": "#dee2e6",
    }


def _badge(ok):
    if ok:
        return (
            '<span style="background:#28a745;color:#fff;padding:2px 7px;'
            'border-radius:3px;font-size:11px;font-weight:700;line-height:16px;'
            'display:inline-block;box-sizing:border-box;">OK</span>'
        )
    return (
        '<span style="background:#e9ecef;color:#6c757d;padding:2px 7px;'
        'border-radius:3px;font-size:11px;line-height:16px;'
        'display:inline-block;box-sizing:border-box;">MISS</span>'
    )


def _badge_link(url):
    return (
        f'<a href="{url}" target="_blank" rel="noopener" title="Download" '
        f'style="background:#28a745;color:#fff;padding:2px 7px;border-radius:3px;'
        f'font-size:11px;font-weight:700;line-height:16px;display:inline-block;box-sizing:border-box;'
        f'text-decoration:underline;text-underline-offset:2px;cursor:pointer;'
        f'box-shadow:0 0 0 1px rgba(40,167,69,.35);">🔗 OK</a>'
    )


def _badge_copy(asset_id):
    js = asset_id.replace("'", "\\'")
    return (
        f'<a href="#" title="Copy asset ID" '
        f'onclick="navigator.clipboard.writeText(\'{js}\');return false;" '
        f'style="background:#28a745;color:#fff;padding:2px 7px;border-radius:3px;'
        f'font-size:11px;font-weight:700;line-height:16px;display:inline-block;box-sizing:border-box;'
        f'text-decoration:none;cursor:pointer;">OK</a>'
    )


# (chave do estado, titulo da coluna, numero da etapa, tipo de badge)
_COLS = [
    ("exported",         "Export",          1, "badge"),
    ("mosaiced",         "Mosaic",          2, "link_mosaic"),
    ("vectorized_gcs",   "Vector GCS",      3, "link_vector"),
    ("vectorized_gee",   "Vector GEE",      4, "copy_asset"),
    ("published_mosaic", "Public mosaic",   5, "link_pub_mosaic"),
    ("published_vector", "Public vector",   6, "link_pub_vector"),
    ("temp_cleaned",     "Clean temp",      7, "badge"),
]


def _empty():
    return {
        "exported": False,
        "mosaiced": False,
        "vectorized_gcs": False,
        "vectorized_gee": False,
        "published_mosaic": False,
        "published_vector": False,
        "temp_cleaned": False,
    }


def _is_complete(v):
    return bool(
        v.get("exported") and v.get("mosaiced") and v.get("vectorized_gcs")
        and v.get("vectorized_gee") and v.get("published_mosaic")
        and v.get("published_vector") and v.get("temp_cleaned")
    )


# ---------------------------------------------------------------------------
# Guias (7 idiomas)
# ---------------------------------------------------------------------------
LANG_ORDER = ["PT", "EN", "FR", "ID", "ES", "NL", "ZH"]

GUIDES = {
    "PT": {
        "name": "Português",
        "what": "Este aplicativo exporta os mapas mensais de área queimada do Monitor do Fogo: "
                "do Earth Engine para o GCS como tiles 0/1 leves, monta um COG mensal, vetoriza "
                "(shapefile + unique_id, zipado), publica no Earth Engine, espelha COG e vetor no "
                "bucket público e, por fim, remove os tiles temporários.",
        "howto_title": "Como usar",
        "steps": [
            "Etapa 1 — Export: envia as imagens do GEE como tiles 0/1 para o GCS (temp/).",
            "Etapa 2 — Mosaic: monta um COG por mês.",
            "Etapa 3 — Vectorize: converte o raster em shapefile (unique_id) e zipa.",
            "Etapa 4 — Upload GEE: publica o ZIP como FeatureCollection.",
            "Etapa 5 — Public mosaic: espelha os COGs no bucket público.",
            "Etapa 6 — Public vector: espelha os ZIPs no bucket público.",
            "Etapa 7 — Clean temp: remove tiles temporários dos meses consolidados.",
        ],
        "cols_title": "Colunas da grade",
        "cols": [
            ("Export", "tiles 0/1 no GCS (temp/)"),
            ("Mosaic", "COG mensal"),
            ("Vector GCS", "vetor zipado no GCS"),
            ("Vector GEE", "FeatureCollection no Earth Engine"),
            ("Public mosaic", "COG espelhado no bucket público"),
            ("Public vector", "ZIP espelhado no bucket público"),
            ("Clean temp", "tiles temporários removidos após consolidação"),
        ],
        "links": "Badges OK de Mosaic/Vector/Publico abrem link direto de download "
                 "(storage.googleapis.com).",
        "theme": "Use o botão 🌙/☀️ no cabeçalho para alternar entre claro e escuro.",
        "legend": "OK = etapa concluída  |  MISS = etapa pendente",
    },
    "EN": {
        "name": "English",
        "what": "This app exports the Fire Monitor's monthly burned-area maps: from Earth Engine "
                "to cloud storage as lightweight 0/1 tiles, builds a monthly COG, vectorizes it "
                "(shapefile + unique_id, zipped), publishes it to Earth Engine, mirrors COG and "
                "vector to the public bucket, and finally removes the temporary tiles.",
        "howto_title": "How to use",
        "steps": [
            "Step 1 — Export: sends the GEE images as 0/1 tiles to GCS (temp/).",
            "Step 2 — Mosaic: builds one COG per month.",
            "Step 3 — Vectorize: raster to shapefile (unique_id) and zips it.",
            "Step 4 — Upload GEE: publishes the ZIP as a FeatureCollection.",
            "Step 5 — Public mosaic: mirrors COGs to the public bucket.",
            "Step 6 — Public vector: mirrors vector ZIPs to the public bucket.",
            "Step 7 — Clean temp: removes temp tiles of consolidated months.",
        ],
        "cols_title": "Grid columns",
        "cols": [
            ("Export", "0/1 tiles in GCS (temp/)"),
            ("Mosaic", "monthly COG"),
            ("Vector GCS", "zipped vector in GCS"),
            ("Vector GEE", "FeatureCollection in Earth Engine"),
            ("Public mosaic", "COG mirrored to public bucket"),
            ("Public vector", "ZIP mirrored to public bucket"),
            ("Clean temp", "temp tiles removed after consolidation"),
        ],
        "links": "OK badges in the Mosaic/Vector/Publico columns open a direct download link "
                 "(storage.googleapis.com).",
        "theme": "Use the 🌙/☀️ button in the header to switch between light and dark.",
        "legend": "OK = stage done  |  MISS = stage pending",
    },
    "FR": {
        "name": "Français",
        "what": "Cette application exporte les cartes mensuelles des surfaces brûlées du Fire "
                "Monitor : depuis Earth Engine vers le GCS en tuiles 0/1 légères, construit un "
                "COG mensuel, vectorise (shapefile + unique_id, compressé), publie dans Earth "
                "Engine, reflète le COG et le vecteur dans le bucket public et supprime les "
                "tuiles temporaires.",
        "howto_title": "Comment utiliser",
        "steps": [
            "Étape 1 — Export : envoie les images GEE en tuiles 0/1 vers le GCS (temp/).",
            "Étape 2 — Mosaic : construit un COG par mois.",
            "Étape 3 — Vectorize : raster → shapefile (unique_id) puis zip.",
            "Étape 4 — Upload GEE : publie le ZIP comme FeatureCollection.",
            "Étape 5 — Public mosaic : reflète les COG dans le bucket public.",
            "Étape 6 — Public vector : reflète les ZIP dans le bucket public.",
            "Étape 7 — Clean temp : supprime les tuiles temporaires des mois consolidés.",
        ],
        "cols_title": "Colonnes de la grille",
        "cols": [
            ("Export", "tuiles 0/1 dans GCS (temp/)"),
            ("Mosaic", "COG mensuel"),
            ("Vector GCS", "vecteur zippé dans GCS"),
            ("Vector GEE", "FeatureCollection dans Earth Engine"),
            ("Public mosaic", "COG reflété dans le bucket public"),
            ("Public vector", "ZIP reflété dans le bucket public"),
            ("Clean temp", "tuiles temporaires supprimées après consolidation"),
        ],
        "links": "Les badges OK de Mosaic/Vector/Publico ouvrent un lien de téléchargement "
                 "direct (storage.googleapis.com).",
        "theme": "Utilisez le bouton 🌙/☀️ dans l'en-tête pour basculer clair/sombre.",
        "legend": "OK = étape terminée  |  MISS = étape en attente",
    },
    "ID": {
        "name": "Bahasa Indonesia",
        "what": "Aplikasi ini mengekspor peta bulanan area terbakar Fire Monitor: dari Earth "
                "Engine ke cloud storage sebagai tile 0/1 ringan, membangun COG bulanan, "
                "memvektorisasi (shapefile + unique_id, zip), mempublikasikan ke Earth Engine, "
                "menyalin COG dan vektor ke bucket publik, lalu menghapus tile sementara.",
        "howto_title": "Cara penggunaan",
        "steps": [
            "Langkah 1 — Export: mengirim gambar GEE sebagai tile 0/1 ke GCS (temp/).",
            "Langkah 2 — Mosaic: membangun satu COG per bulan.",
            "Langkah 3 — Vectorize: raster → shapefile (unique_id) lalu zip.",
            "Langkah 4 — Upload GEE: memublikasikan ZIP sebagai FeatureCollection.",
            "Langkah 5 — Public mosaic: menyalin COG ke bucket publik.",
            "Langkah 6 — Public vector: menyalin ZIP vektor ke bucket publik.",
            "Langkah 7 — Clean temp: menghapus tile sementara bulan yang terkonsolidasi.",
        ],
        "cols_title": "Kolom grid",
        "cols": [
            ("Export", "tile 0/1 di GCS (temp/)"),
            ("Mosaic", "COG bulanan"),
            ("Vector GCS", "vektor zip di GCS"),
            ("Vector GEE", "FeatureCollection di Earth Engine"),
            ("Public mosaic", "COG disalin ke bucket publik"),
            ("Public vector", "ZIP disalin ke bucket publik"),
            ("Clean temp", "tile sementara dihapus setelah konsolidasi"),
        ],
        "links": "Lencana OK di kolom Mosaic/Vector/Publico membuka tautan unduhan langsung "
                 "(storage.googleapis.com).",
        "theme": "Gunakan tombol 🌙/☀️ di header untuk beralih terang/gelap.",
        "legend": "OK = tahap selesai  |  MISS = tahap tertunda",
    },
    "ES": {
        "name": "Español",
        "what": "Esta aplicación exporta los mapas mensuales de área quemada del Fire Monitor: "
                "desde Earth Engine a GCS como tiles 0/1 ligeros, construye un COG mensual, "
                "vectoriza (shapefile + unique_id, comprimido), publica en Earth Engine, refleja "
                "el COG y el vector en el bucket público y elimina los tiles temporales.",
        "howto_title": "Cómo usar",
        "steps": [
            "Paso 1 — Export: envía las imágenes GEE como tiles 0/1 a GCS (temp/).",
            "Paso 2 — Mosaic: construye un COG por mes.",
            "Paso 3 — Vectorize: ráster → shapefile (unique_id) y comprime.",
            "Paso 4 — Upload GEE: publica el ZIP como FeatureCollection.",
            "Paso 5 — Public mosaic: refleja los COG en el bucket público.",
            "Paso 6 — Public vector: refleja los ZIP en el bucket público.",
            "Paso 7 — Clean temp: elimina tiles temporales de meses consolidados.",
        ],
        "cols_title": "Columnas de la cuadrícula",
        "cols": [
            ("Export", "tiles 0/1 en GCS (temp/)"),
            ("Mosaic", "COG mensual"),
            ("Vector GCS", "vector comprimido en GCS"),
            ("Vector GEE", "FeatureCollection en Earth Engine"),
            ("Public mosaic", "COG reflejado en el bucket público"),
            ("Public vector", "ZIP reflejado en el bucket público"),
            ("Clean temp", "tiles temporales eliminados tras consolidación"),
        ],
        "links": "Las insignias OK de Mosaic/Vector/Publico abren un enlace de descarga directa "
                 "(storage.googleapis.com).",
        "theme": "Use el botón 🌙/☀️ en el encabezado para alternar claro/oscuro.",
        "legend": "OK = etapa completada  |  MISS = etapa pendiente",
    },
    "NL": {
        "name": "Nederlands",
        "what": "Deze app exporteert de maandelijkse verbrande-oppervlaktekaarten van de Fire "
                "Monitor: van Earth Engine naar cloud storage als lichte 0/1 tiles, bouwt een "
                "maandelijkse COG, vectoriseert (shapefile + unique_id, gezipt), publiceert naar "
                "Earth Engine, spiegelt COG en vector naar de publieke bucket en verwijdert "
                "tijdelijke tiles.",
        "howto_title": "Hoe te gebruiken",
        "steps": [
            "Stap 1 — Export: stuurt de GEE-beelden als 0/1 tiles naar GCS (temp/).",
            "Stap 2 — Mosaic: bouwt één COG per maand.",
            "Stap 3 — Vectorize: raster → shapefile (unique_id) en zipt.",
            "Stap 4 — Upload GEE: publiceert de ZIP als FeatureCollection.",
            "Stap 5 — Public mosaic: spiegelt COG's naar de publieke bucket.",
            "Stap 6 — Public vector: spiegelt vector-ZIP's naar de publieke bucket.",
            "Stap 7 — Clean temp: verwijdert tijdelijke tiles van geconsolideerde maanden.",
        ],
        "cols_title": "Grid-kolommen",
        "cols": [
            ("Export", "0/1 tiles in GCS (temp/)"),
            ("Mosaic", "maandelijkse COG"),
            ("Vector GCS", "gezipte vector in GCS"),
            ("Vector GEE", "FeatureCollection in Earth Engine"),
            ("Public mosaic", "COG gespiegeld naar publieke bucket"),
            ("Public vector", "ZIP gespiegeld naar publieke bucket"),
            ("Clean temp", "tijdelijke tiles verwijderd na consolidatie"),
        ],
        "links": "OK-badges in de kolommen Mosaic/Vector/Publico openen een directe "
                 "downloadlink (storage.googleapis.com).",
        "theme": "Gebruik de 🌙/☀️-knop in de header om licht/donker te wisselen.",
        "legend": "OK = fase klaar  |  MISS = fase in afwachting",
    },
    "ZH": {
        "name": "中文",
        "what": "此应用导出 Fire Monitor 的月度过火面积地图：从 Earth Engine 到云端存储作为轻量 0/1 瓦片，构建月度 COG，矢量化为带 unique_id 的压缩 shapefile，发布到 Earth Engine，镜像 COG 和矢量到公共存储桶，最后删除临时瓦片。",
        "howto_title": "使用方法",
        "steps": [
            "步骤 1 — Export：将 GEE 影像作为 0/1 瓦片发送到 GCS（temp/）。",
            "步骤 2 — Mosaic：每月构建一个 COG。",
            "步骤 3 — Vectorize：栅格转 shapefile（unique_id）并压缩。",
            "步骤 4 — Upload GEE：将 ZIP 发布为 FeatureCollection。",
            "步骤 5 — Public mosaic：将 COG 镜像到公共存储桶。",
            "步骤 6 — Public vector：将矢量 ZIP 镜像到公共存储桶。",
            "步骤 7 — Clean temp：删除已合并月份的临时瓦片。",
        ],
        "cols_title": "网格列",
        "cols": [
            ("Export", "GCS 中的 0/1 瓦片 (temp/)"),
            ("Mosaic", "月度 COG"),
            ("Vector GCS", "GCS 中的压缩矢量"),
            ("Vector GEE", "Earth Engine 中的 FeatureCollection"),
            ("Public mosaic", "镜像到公共存储桶的 COG"),
            ("Public vector", "镜像到公共存储桶的 ZIP"),
            ("Clean temp", "合并后删除的临时瓦片"),
        ],
        "links": "Mosaic/Vector/Publico 列的 OK 徽章会打开直接下载链接（storage.googleapis.com）。",
        "theme": "使用标题栏中的 🌙/☀️ 按钮在明暗模式间切换。",
        "legend": "OK = 阶段完成 | MISS = 阶段待处理",
    },
}


def _guide_html(lang):
    g = GUIDES[lang]
    p = _palette()
    cols_rows = "".join(
        f'<tr><td style="padding:3px 8px;border:1px solid {p["guide_border"]};white-space:nowrap;">'
        f'<b>{c}</b></td>'
        f'<td style="padding:3px 8px;border:1px solid {p["guide_border"]};">{m}</td></tr>'
        for c, m in g["cols"]
    )
    steps = "".join(f"<li>{s}</li>" for s in g["steps"])
    return (
        f'<div style="font-size:12px;color:{p["guide_fg"]};line-height:1.6;">'
        f'<p><b>{g["name"]}</b> — {g["what"]}</p>'
        f'<h4 style="margin:10px 0 4px 0;">{g["howto_title"]}</h4>'
        f'<ol style="margin:0 0 8px 0;padding-left:20px;">{steps}</ol>'
        f'<p><span style="background:#28a745;color:#fff;padding:1px 6px;border-radius:3px;'
        f'font-size:10px;">OK</span> = {g["legend"].split("|")[0].split("=")[1].strip()} &nbsp;|&nbsp; '
        f'<span style="background:#e9ecef;color:#6c757d;padding:1px 6px;border-radius:3px;'
        f'font-size:10px;">MISS</span> = {g["legend"].split("|")[1].split("=")[1].strip()}</p>'
        f'<h4 style="margin:10px 0 4px 0;">{g["cols_title"]}</h4>'
        f'<table style="border-collapse:collapse;">{cols_rows}</table>'
        f'<p>{g["links"]}</p>'
        f'</div>'
    )


class MonitorUI:
    _DATE_W = "100px"
    _CELL_W = "76px"
    _SEL_W  = "64px"

    def __init__(self):
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.log_area = widgets.Output()

        self.grid_container = widgets.VBox([
            widgets.HTML(
                '<div style="padding:20px;text-align:center;color:#6c757d;font-size:13px;">'
                '<i>Loading months from the collection...</i></div>'
            )
        ])

        self.btn_sync = widgets.Button(
            description="Sync", button_style="success", icon="refresh",
            layout=L(width="120px", height="34px")
        )
        self.btn_sync.on_click(self._on_sync)

        self.btn_select_pending = widgets.Button(
            description="Select Pending", button_style="info",
            layout=L(width="150px", height="34px")
        )
        self.btn_select_pending.on_click(self._on_select_pending)

        self.btn_clear = widgets.Button(
            description="Clear", button_style="warning",
            layout=L(width="90px", height="34px")
        )
        self.btn_clear.on_click(self._on_clear)

        self.btn_select_all = widgets.Button(
            description="Select All", button_style="info",
            layout=L(width="120px", height="34px")
        )
        self.btn_select_all.on_click(self._on_select_all)

        self.year_filter = None
        self.year_dropdown = widgets.Dropdown(
            options=["All years"], value="All years",
            description="Year:", layout=L(width="220px")
        )
        self.year_dropdown.observe(self._on_year_change, names="value")

        self.loader = widgets.HTML(
            value='<span id="mon-loader" style="display:none;margin-left:10px;color:#3498db;font-size:13px;">Syncing...</span>'
        )

        self.toolbar = widgets.HBox([
            self.year_dropdown, self.btn_select_pending, self.btn_select_all,
            self.btn_clear, self.btn_sync, self.loader,
        ], layout=L(margin="0 0 8px 0", gap="8px", align_items="center"))

        self.container = widgets.VBox([
            _STATUS_CSS,
            self.toolbar,
            self.grid_container,
            self.log_area,
        ])

        self._render_panel()

    def _render_panel(self):
        p = _palette()
        self.container.layout = L(
            border=f"1px solid {p['panel_border']}", padding="10px",
            border_radius="5px", margin="10px 0", background=p["panel_bg"]
        )
        self.log_area.layout = L(background=p["panel_bg"])
        self._render_grid()

    def display(self):
        display(self.container)

    def _log(self, message, type="info"):
        colors = {"info": "#3498db", "success": "#27ae60", "error": "#d32f2f", "warning": "#e67e22"}
        color = colors.get(type, "#333")
        with self.log_area:
            display(widgets.HTML(
                f'<span style="color:{color};font-size:12px;">[{type.upper()}] {message}</span>'
            ))

    def start(self):
        months = list_months_in_collection()
        if months:
            for m in months:
                self.state[m] = _empty()
            self._render_grid()
            self._log(f"{len(months)} months in collection. Syncing automatically...", "info")
        else:
            self._log("Could not query the collection. Check GEE authentication.", "warning")
        self._on_sync(None)

    def _on_sync(self, _):
        if self.is_refreshing:
            return
        self.is_refreshing = True
        self.btn_sync.disabled = True
        self.btn_sync.description = "Syncing..."
        self.loader.value = self.loader.value.replace("display:none", "display:flex")
        self._log("Checking files in GCS and assets in GEE...", "info")
        try:
            selected = self._get_selected_keys()
            self.state = build_state(logger=self._log)
            self._render_grid()
            self._restore_selected(selected)
            completed = sum(1 for k, v in self.state.items() if k != "updated_at" and _is_complete(v))
            total = len([k for k in self.state if k != "updated_at"])
            self._log(f"Sync complete: {completed}/{total} months complete.", "success")
        except Exception as e:
            self._log(f"Sync error: {e}", "error")
        finally:
            self.is_refreshing = False
            self.btn_sync.disabled = False
            self.btn_sync.description = "Sync"
            self.loader.value = self.loader.value.replace("display:flex", "display:none")

    def _all_months(self):
        return sorted(
            [k for k in self.state.keys() if k != "updated_at"],
            reverse=True
        )

    def _filtered_months(self):
        months = self._all_months()
        if self.year_filter is not None:
            months = [k for k in months if k.startswith(f"{self.year_filter}_")]
        return months

    def _refresh_year_dropdown(self):
        years = sorted({int(k.split("_")[0]) for k in self._all_months()}, reverse=True)
        options = ["All years"] + [str(y) for y in years]
        if self.year_dropdown.value not in options:
            self.year_dropdown.value = "All years"
        self.year_dropdown.options = options

    def _on_year_change(self, change):
        value = change.get("new")
        self.year_filter = int(value) if value != "All years" else None
        selected = self._get_selected_keys()
        self._render_grid()
        self._restore_selected(selected)

    def _col_content(self, kind, ok, y, m):
        if not ok:
            return _badge(False)
        if kind == "link_mosaic":
            url = f"https://storage.googleapis.com/{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(y, m)}.tif"
            return _badge_link(url)
        if kind == "link_vector":
            url = f"https://storage.googleapis.com/{config.BUCKET}/{config.vector_prefix()}/{config.vector_name(y, m)}.zip"
            return _badge_link(url)
        if kind == "link_pub_mosaic":
            url = f"https://storage.googleapis.com/{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(y, m)}.tif"
            return _badge_link(url)
        if kind == "link_pub_vector":
            url = f"https://storage.googleapis.com/{config.PUBLIC_BUCKET}/{config.vector_prefix()}/{config.vector_name(y, m)}.zip"
            return _badge_link(url)
        if kind == "copy_asset":
            asset_id = f"{config.vector_asset_prefix()}/{config.vector_name(y, m)}"
            return _badge_copy(asset_id)
        return _badge(True)

    def _render_grid(self):
        self.chk_dict = {}
        p = _palette()

        def _header_cell(width, title, etapa):
            return widgets.HTML(
                f'<div style="width:{width};text-align:center;font-weight:700;font-size:10px;'
                f'color:{p["grid_header_fg"]};line-height:1.25;box-sizing:border-box;'
                f'border-right:1px solid {p["sep"]};">'
                f'{title}<br><span style="font-size:9px;font-weight:400;opacity:.85;">Step {etapa}</span>'
                f'</div>'
            )

        header_row = widgets.HBox(
            [widgets.HTML(f'<div style="width:{self._DATE_W};font-weight:700;font-size:12px;color:{p["grid_header_fg"]};'
                          f'box-sizing:border-box;border-right:1px solid {p["sep"]};">Date</div>')]
            + [_header_cell(self._CELL_W, t, e) for _, t, e, _ in _COLS]
            + [widgets.HTML(f'<div style="width:{self._SEL_W};text-align:center;font-weight:700;font-size:11px;color:{p["grid_header_fg"]};">Select</div>')],
            layout=L(
                background=p["grid_header_bg"], padding="6px 10px", min_height="44px",
                align_items="center", overflow="visible"
            )
        )

        rows = [header_row]

        self._refresh_year_dropdown()
        months = self._filtered_months()

        row_layout = L(
            align_items="center", min_height="38px",
            border_bottom=f"1px solid {p['border']}", padding="3px 10px",
            overflow="visible", width="100%"
        )

        for i, m in enumerate(months):
            info = self.state.get(m, {})
            y, mm = int(m.split("_")[0]), int(m.split("_")[1])
            bg = p["row_a"] if i % 2 == 0 else p["row_b"]

            date_cell = widgets.HTML(
                f'<div style="width:{self._DATE_W};font-family:monospace;font-size:13px;color:{p["date_fg"]};font-weight:600;'
                f'background:{p["date_bg"]};padding:2px 6px;border-radius:3px;box-sizing:border-box;'
                f'border-right:1px solid {p["sep"]};">{m}</div>'
            )

            cells = [date_cell]
            for key, _t, _e, kind in _COLS:
                ok = info.get(key, False)
                cells.append(widgets.HTML(
                    f'<div style="width:{self._CELL_W};text-align:center;box-sizing:border-box;'
                    f'border-right:1px solid {p["sep"]};">{self._col_content(kind, ok, y, mm)}</div>'
                ))

            chk = widgets.Checkbox(value=False, indent=False, layout=L(width="20px", height="20px"))
            chk_wrapper = widgets.HBox([chk], layout=L(
                width=self._SEL_W, justify_content="center",
                align_items="center", overflow="hidden"
            ))
            self.chk_dict[m] = chk

            row = widgets.HBox(cells + [chk_wrapper], layout=row_layout)
            row.layout.background = bg
            rows.append(row)

        n_all = len(self._all_months())
        n_complete = sum(1 for k in self._all_months() if _is_complete(self.state[k]))
        n_visible = len(months)
        n_visible_complete = sum(1 for k in months if _is_complete(self.state[k]))

        if self.year_filter is not None:
            label = (
                f'{n_visible} months of {self.year_filter} in filter &nbsp;|&nbsp; '
                f'<span style="color:#28a745;font-weight:700;">{n_visible_complete}</span> complete &nbsp;|&nbsp; '
                f'<span style="color:#6c757d;">{n_visible - n_visible_complete}</span> pending'
            )
        else:
            label = (
                f'{n_all} months in collection &nbsp;|&nbsp; '
                f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> complete &nbsp;|&nbsp; '
                f'<span style="color:#6c757d;">{n_all - n_complete}</span> pending'
            )

        legend = widgets.HTML(
            f'<div style="font-size:11px;color:{p["legend_fg"]};margin:6px 0 0 10px;padding:6px 10px;'
            f'background:{p["legend_bg"]};border-radius:4px;">'
            f'{label}'
            f'</div>'
        )

        hint = widgets.HTML(
            f'<div style="font-size:11px;color:{p["hint_fg"]};margin:4px 0 0 10px;padding:6px 10px;'
            f'background:{p["hint_bg"]};border:1px solid {p["hint_border"]};border-radius:4px;line-height:1.5;">'
            f'<strong>MISS &rarr; OK:</strong> '
            f'<b>Export</b>=Step 1 cell &nbsp;|&nbsp; '
            f'<b>Mosaic</b>=Step 2 cell &nbsp;|&nbsp; '
            f'<b>Vector GCS</b>=Step 3 cell &nbsp;|&nbsp; '
            f'<b>Vector GEE</b>=Step 4 cell &nbsp;|&nbsp; '
            f'<b>Public mosaic</b>=Step 5 cell &nbsp;|&nbsp; '
            f'<b>Public vector</b>=Step 6 cell &nbsp;|&nbsp; '
            f'<b>Clean temp</b>=Step 7 cell (after both published)'
            f'</div>'
        )

        self.grid_container.children = [
            widgets.VBox(rows, layout=L(
                max_height="460px", width="100%",
                overflow_y="auto", overflow_x="auto",
                padding="0", border=f"1px solid {p['border']}",
                background_color=p["row_b"]
            )),
            legend,
            hint,
        ]

    def _on_select_pending(self, _):
        for key, chk in self.chk_dict.items():
            if not _is_complete(self.state.get(key, {})):
                chk.value = True

    def _on_select_all(self, _):
        for chk in self.chk_dict.values():
            chk.value = True

    def _on_clear(self, _):
        for chk in self.chk_dict.values():
            chk.value = False

    def get_selected_months(self):
        result = []
        for key, chk in self.chk_dict.items():
            if chk.value:
                parts = key.split("_")
                if len(parts) >= 2:
                    result.append((int(parts[0]), int(parts[1])))
        return result

    def _get_selected_keys(self):
        return [k for k, chk in self.chk_dict.items() if chk.value]

    def _restore_selected(self, keys):
        for k in keys:
            if k in self.chk_dict:
                self.chk_dict[k].value = True

    def sync(self):
        selected = self._get_selected_keys()
        self.state = build_state(logger=self._log)
        self._render_grid()
        self._restore_selected(selected)


class CountryTabs:
    """Abas por pais dentro da aba Interface. Cada aba tem seu proprio MonitorUI."""

    def __init__(self, countries):
        self.countries = list(countries)
        if not self.countries:
            raise ValueError("No countries configured for the tabs.")
        for c in self.countries:
            if c not in config.COUNTRIES:
                raise ValueError(f"Country '{c}' not in config.COUNTRIES.")

        self._panels = {}
        self._active_code = self.countries[0]
        self._active_panel = None

        self._placeholders = [widgets.VBox([]) for _ in self.countries]
        self.tab = widgets.Tab(children=self._placeholders)
        for i, c in enumerate(self.countries):
            self.tab.set_title(i, f"{config.flag(c)} {c.title()}")

        self.tab.observe(self._on_tab_change, names="selected_index")

    def _on_tab_change(self, change):
        idx = change.get("new")
        if idx is None:
            return
        self._activate(idx)

    def _activate(self, idx):
        code = self.countries[idx]
        self._active_code = code
        if code not in self._panels:
            config.set_country(code, verbose=False)
            panel = MonitorUI()
            panel.start()
            self._panels[code] = panel
            self._placeholders[idx].children = [panel.container]
        else:
            panel = self._panels[code]
            panel.sync()
        self._active_panel = panel

    def __getattr__(self, name):
        panel = self.__dict__.get("_active_panel")
        if panel is None:
            raise AttributeError(name)
        return getattr(panel, name)

    def display(self):
        display(self.tab)


class FireMonitorApp:
    """App em guias: Interface (abas de pais) + guias em 7 idiomas."""

    def __init__(self, countries):
        self.interface = CountryTabs(countries)

        self.header = widgets.HTML()
        self.guide_widgets = [widgets.HTML() for _ in LANG_ORDER]

        self.tab = widgets.Tab(children=[self.interface.tab] + self.guide_widgets)
        self.tab.set_title(0, "Interface")
        for i, lang in enumerate(LANG_ORDER, start=1):
            self.tab.set_title(i, GUIDES[lang]["name"])

        self.container = widgets.VBox([self.header, self.tab])
        self._render()

    def _render(self):
        p = _palette()
        self.header.value = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;width:100%;'
            f'padding:10px 14px;background:{p["header_bg"]};border:1px solid {p["header_border"]};'
            f'border-radius:5px;margin-bottom:8px;">'
            f'<div>'
            f'<span style="font-weight:bold;font-size:17px;color:{p["title"]};">Export &amp; Vectorization</span>'
            f'<span style="color:{p["subtitle"]};font-size:12px;margin-left:14px;">MapBiomas Fire Monitor</span>'
            f'</div>'
            f'<div style="color:{p["subtitle"]};font-size:12px;">{config.flag(config.COUNTRY)} {config.COUNTRY.title()}</div>'
            f'</div>'
        )
        self.container.children = [self.header, self.tab]
        for i, lang in enumerate(LANG_ORDER):
            self.guide_widgets[i].value = (
                f'<div style="padding:12px;background:{p["guide_bg"]};border:1px solid {p["guide_border"]};'
                f'border-radius:5px;">{_guide_html(lang)}</div>'
            )

    def __getattr__(self, name):
        return getattr(self.interface, name)

    def display(self):
        display(self.container)


def run_ui(countries=None):
    countries = countries or config.COUNTRIES_AVAILABLE
    app = FireMonitorApp(countries)
    # exibe o shell (cabecalho + abas) imediatamente; depois monta/sincroniza
    # o painel do primeiro pais (nao bloqueia a renderizacao).
    app.display()
    app.interface._activate(0)
    return app
