import datetime
import random
import threading
import ipywidgets as widgets
from IPython.display import display, clear_output

from . import config
from .state import list_months_in_collection, build_state

L = widgets.Layout

_STATUS_CSS = widgets.HTML("""<style>
@keyframes mfm-spin { to { transform: rotate(360deg); } }
@keyframes mfm-pulse-outline {
    0% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.8); }
    50% { box-shadow: 0 0 0 10px rgba(255, 193, 7, 0); }
    100% { box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }
}
.mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
.mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
.mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
.mfm-btn-unloaded {
    animation: mfm-pulse-outline 1.5s infinite;
}
</style>""")

STORY_SEQUENCES = [
    ["[ 🌲 🌲 🌲 🌲 🌲 ]", "[ 🌲 🌲 🦅 🌲 🌲 ]", "[ 🌲 🌲 🔥 🌲 🌲 ]",
     "[ 🛰️ 🔥 🔥 🔥 🌲 ]", "[ 💻 🧠 ⚙️ ☁️ ☁️ ]", "[ 🗺️ 📍 ✅ ✨ ✨ ]"],
    ["🕛", "🕒", "🕕", "🕘"],
    ["🌍", "🛰️", "🔥", "🤖", "🗺️"],
    ["🛰️", "📡", "🔥", "💻", "🗺️"],
    ["🌲", "🔎", "🔥", "🚨", "🧠", "🗺️"],
    ["🛰️ 🌍", "🛰️ 🔎", "🛰️ 🔥", "📡 💻", "🤖 🧠", "📊 🗺️"],
]


class StoryLoader:
    """Animacao de carregamento que nao bloqueia o kernel do notebook."""

    def __init__(self, label="Loading...", interval=0.7):
        self.label = label
        self.interval = interval
        self.widget = widgets.HTML()
        self._running = False
        self._thread = None

    def _render(self, frame):
        self.widget.value = (
            '<div style="padding:14px 18px;color:#3498db;font-size:18px;line-height:1.5;">'
            f'<code>{frame}</code> <span style="font-size:12px;">{self.label}</span></div>'
        )

    def _run(self):
        sequence = random.choice(STORY_SEQUENCES)
        index = 0
        repeats = 0
        while self._running:
            self._render(sequence[index])
            index = (index + 1) % len(sequence)
            if index == 0:
                repeats += 1
                if repeats >= 2:
                    sequence = random.choice(STORY_SEQUENCES)
                    repeats = 0
            threading.Event().wait(self.interval)

    def start(self):
        if self._running:
            return self.widget
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.widget

    def stop(self, message=None):
        self._running = False
        self._thread = None
        if message:
            self._render(message)


class LogDrawer:
    """Guia global com uma visao do ultimo log e outra do historico."""

    def __init__(self):
        self.output = widgets.Output()
        self.history = []
        self.last = widgets.HTML(value="<div style='padding:5px 8px;color:#6c757d;'>No messages yet.</div>")
        self.output = widgets.Output()
        self.tab = widgets.Tab(children=[self.last, self.output])
        self.tab.set_title(0, "Last log")
        self.tab.set_title(1, "Log history")
        self.container = widgets.VBox([self.tab])
        self.output.layout = L(max_height="240px", overflow="auto", padding="4px",
                               border="1px solid #cccccc", background="#ffffff")

    def append(self, html):
        self.history.append(html)
        self.last.value = f'<div style="padding:5px 8px;color:#495057;font-size:12px;">{html}</div>'
        with self.output:
            display(widgets.HTML(html))


def _palette():
    return {
        "panel_bg": "#ffffff",
        "panel_border": "#cccccc",
        "header_bg": "#ffffff",
        "header_border": "#333333",
        "title": "#333333",
        "subtitle": "#6c757d",
        "grid_header_bg": "#263238",
        "grid_header_fg": "#ffffff",
        "grid_header_muted_bg": "#3a3a3a",
        "grid_header_muted_fg": "#888888",
        "row_a": "#f7f9fb",
        "row_b": "#ffffff",
        "date_bg": "#e8eef3",
        "date_fg": "#212529",
        "legend_bg": "#f8f9fa",
        "legend_fg": "#6c757d",
        "hint_bg": "#fffbe6",
        "hint_border": "#ffe58f",
        "hint_fg": "#495057",
        "guide_bg": "#ffffff",
        "guide_border": "#dddddd",
        "guide_fg": "#333333",
        "border": "#cccccc",
        "sep": "#cbd5df",
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


def _badge_na():
    return (
        '<span style="background:#f1f3f5;color:#adb5bd;padding:2px 7px;'
        'border-radius:3px;font-size:11px;line-height:16px;'
        'display:inline-block;box-sizing:border-box;">N/A</span>'
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


def _loading_html(label="Loading..."):
    return widgets.HTML(
        value=(f'<div style="padding:18px;color:#3498db;font-size:13px;">'
               f'<span style="display:inline-block;width:12px;height:12px;margin-right:8px;'
               f'border:2px solid #b9d7f0;border-top-color:#3498db;border-radius:50%;'
               f'animation:mfm-spin 0.8s linear infinite;vertical-align:-2px;"></span>{label}</div>')
    )


# (state key, column title, step number, badge type)
# New order: 1=Export, 2=Mosaic, 3=Clean Temp, 4=Vector GCS, 5=Vector GEE, 6=Public Mosaic, 7=Public Vector
# Vector steps (4,5,7) only shown for vectorizable products
_COLS = [
    ("exported",         "Export",          1, "badge"),
    ("mosaiced",         "Mosaic",          2, "link_mosaic"),
    ("temp_cleaned",     "Clean temp",      3, "badge"),
    ("vectorized_gcs",   "Vector GCS",      4, "link_vector"),
    ("vectorized_gee",   "Vector GEE",      5, "copy_asset"),
    ("published_mosaic", "Public mosaic",   6, "link_pub_mosaic"),
    ("published_vector", "Public vector",   7, "link_pub_vector"),
]

_VECTOR_KINDS = {"link_vector", "copy_asset", "link_pub_vector"}


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
    """Check if unit is complete. Supports both v1 (old step order) and v2 (new step order)."""
    # v2 keys (new order)
    base_v2 = bool(v.get("exported") and v.get("mosaiced")
                   and v.get("published_mosaic") and v.get("temp_cleaned"))
    # v1 keys (old order: temp_cleaned was step 7)
    base_v1 = bool(v.get("exported") and v.get("mosaiced")
                   and v.get("published_mosaic") and v.get("temp_cleaned"))
    base = base_v2 or base_v1
    if config.is_vectorizable():
        return base and v.get("vectorized_gcs") and v.get("vectorized_gee")
    return base


# ---------------------------------------------------------------------------
# Guias (7 idiomas)
# ---------------------------------------------------------------------------
LANG_ORDER = ["PT", "ES", "EN", "ID", "FR", "NL", "ZH"]

GUIDES = {
    "PT": {
        "name": "Português",
        "tab_title": "Guia: Português",
        "welcome_note": "Bem-vindo! A interface do aplicativo está em inglês por padrão, mas preparamos esta documentação completa em Português para orientar você em cada etapa — desde a escolha do país e produto até a publicação final. Sinta-se à vontade para explorar as abas acima e consultar este guia sempre que precisar.",
        "what": "Este aplicativo exporta os mapas de área queimada/incêndio do MapBiomas Fire: "
                "do Earth Engine para o GCS, monta mosaicos por unidade (banda/imagem), vetoriza "
                "(quando aplicável), publica no Earth Engine e no bucket público e remove os "
                "arquivos temporários.",
        "howto_title": "Como usar",
        "steps": [
            "Escolha o país, o tema (ex.: fire), a coleção e o produto.",
            "No produto, marque as unidades (bandas ou imagens) desejadas.",
            "Execute as etapas em ordem: Export, Mosaico, Vetorização (se houver), Upload GEE (se houver), Publicar mosaico, Publicar vetor, Limpar temp.",
        ],
        "cols_title": "Colunas da grade",
        "cols": [
            ("Export", "unidade exportada do GEE (temp/)"),
            ("Mosaic", "COG montado"),
            ("Vector GCS", "vetor zipado (só produtos vetorizáveis)"),
            ("Vector GEE", "FeatureCollection no Earth Engine"),
            ("Public mosaic", "COG espelhado no bucket público"),
            ("Public vector", "ZIP espelhado no bucket público"),
            ("Clean temp", "tiles temporários removidos após consolidação"),
        ],
        "links": "Badges 🔗 OK abrem o link de download.",
        "legend": "OK = etapa concluída  |  MISS = etapa pendente  |  N/A = não se aplica",
    },
    "EN": {
        "name": "English",
        "tab_title": "Guide: English",
        "welcome_note": "Welcome! The application interface defaults to English, and this documentation is provided in English to walk you through every step — from choosing the country and product to final publication. Feel free to explore the tabs above and refer to this guide whenever you need.",
        "what": "This app exports the MapBiomas Fire burned/fire maps: from Earth Engine to cloud "
                "storage, builds per-unit mosaics (band/image), vectorizes (when applicable), "
                "publishes to Earth Engine and the public bucket, and removes temporary files.",
        "howto_title": "How to use",
        "steps": [
            "Pick the country, theme (e.g., fire), collection and product.",
            "In the product, check the units (bands or images) you want.",
            "Run the steps in order: Export, Mosaic, Vectorize (if any), Upload GEE (if any), Publish mosaic, Publish vector, Clean temp.",
        ],
        "cols_title": "Grid columns",
        "cols": [
            ("Export", "unit exported from GEE (temp/)"),
            ("Mosaic", "built COG"),
            ("Vector GCS", "zipped vector (vectorizable products only)"),
            ("Vector GEE", "FeatureCollection in Earth Engine"),
            ("Public mosaic", "COG mirrored to public bucket"),
            ("Public vector", "ZIP mirrored to public bucket"),
            ("Clean temp", "temp tiles removed after consolidation"),
        ],
        "links": "🔗 OK badges open the download link.",
        "legend": "OK = stage done  |  MISS = stage pending  |  N/A = not applicable",
    },
    "FR": {
        "name": "Français",
        "tab_title": "Guide: Français",
        "welcome_note": "Bienvenue ! L'interface de l'application est en anglais par défaut, mais nous avons préparé cette documentation complète en Français pour vous accompagner à chaque étape — du choix du pays et du produit jusqu'à la publication finale. N'hésitez pas à explorer les onglets ci-dessus et à consulter ce guide quand vous en avez besoin.",
        "what": "Cette application exporte les cartes de brûlage MapBiomas Fire : d'Earth Engine "
                "vers le stockage cloud, construit des mosaïques par unité (bande/image), vectorise "
                "(si applicable), publie dans Earth Engine et le bucket public et supprime les "
                "fichiers temporaires.",
        "howto_title": "Comment utiliser",
        "steps": [
            "Choisissez le pays, le thème, la collection et le produit.",
            "Dans le produit, cochez les unités (bandes ou images) souhaitées.",
            "Exécutez les étapes en ordre : Export, Mosaic, Vectorize (si applicable), Upload GEE, Public mosaic, Public vector, Clean temp.",
        ],
        "cols_title": "Colonnes de la grille",
        "cols": [
            ("Export", "unité exportée (temp/)"),
            ("Mosaic", "COG construit"),
            ("Vector GCS", "vecteur zippé (produits vectorisables)"),
            ("Vector GEE", "FeatureCollection dans Earth Engine"),
            ("Public mosaic", "COG reflété dans le bucket public"),
            ("Public vector", "ZIP reflété dans le bucket public"),
            ("Clean temp", "tuiles temporaires supprimées"),
        ],
        "links": "Les badges 🔗 OK ouvrent le lien de téléchargement.",
        "legend": "OK = étape terminée  |  MISS = en attente  |  N/A = non applicable",
    },
    "ID": {
        "name": "Bahasa Indonesia",
        "tab_title": "Panduan: Bahasa Indonesia",
        "welcome_note": "Selamat datang! Antarmuka aplikasi ini menggunakan bahasa Inggris secara default, namun kami menyediakan dokumentasi lengkap dalam Bahasa Indonesia untuk memandu Anda di setiap langkah — dari memilih negara dan produk hingga publikasi akhir. Silakan jelajahi tab di atas dan rujuk panduan ini kapan saja diperlukan.",
        "what": "Aplikasi ini mengekspor peta kebakaran MapBiomas Fire: dari Earth Engine ke cloud "
                "storage, membangun mozaik per unit (band/citra), vektorisasi (jika berlaku), "
                "mempublikasikan ke Earth Engine dan bucket publik, lalu menghapus file sementara.",
        "howto_title": "Cara penggunaan",
        "steps": [
            "Pilih negara, tema, koleksi, dan produk.",
            "Di produk, centang unit (band atau citra) yang diinginkan.",
            "Jalankan langkah: Export, Mosaic, Vectorize (jika ada), Upload GEE, Public mosaic, Public vector, Clean temp.",
        ],
        "cols_title": "Kolom grid",
        "cols": [
            ("Export", "unit diekspor (temp/)"),
            ("Mosaic", "COG dibangun"),
            ("Vector GCS", "vektor zip (produk yang dapat divektor)"),
            ("Vector GEE", "FeatureCollection di Earth Engine"),
            ("Public mosaic", "COG disalin ke bucket publik"),
            ("Public vector", "ZIP disalin ke bucket publik"),
            ("Clean temp", "tile sementara dihapus"),
        ],
        "links": "Lencana 🔗 OK membuka tautan unduhan.",
        "legend": "OK = tahap selesai  |  MISS = tertunda  |  N/A = tidak berlaku",
    },
    "ES": {
        "name": "Español",
        "tab_title": "Guía: Español",
        "welcome_note": "¡Bienvenido! La interfaz de la aplicación está en inglés por defecto, pero hemos preparado esta documentación completa en Español para guiarle en cada paso — desde la selección del país y producto hasta la publicación final. No dude en explorar las pestañas superiores y consultar esta guía cuando lo necesite.",
        "what": "Esta aplicación exporta los mapas de fuego MapBiomas Fire: desde Earth Engine a "
                "cloud storage, construye mosaicos por unidad (banda/imagen), vectoriza (si "
                "aplica), publica en Earth Engine y el bucket público y elimina archivos temporales.",
        "howto_title": "Cómo usar",
        "steps": [
            "Elija país, tema, colección y producto.",
            "En el producto, marque las unidades (bandas o imágenes) deseadas.",
            "Ejecute las etapas: Export, Mosaic, Vectorize (si aplica), Upload GEE, Public mosaic, Public vector, Clean temp.",
        ],
        "cols_title": "Columnas de la cuadrícula",
        "cols": [
            ("Export", "unidad exportada (temp/)"),
            ("Mosaic", "COG construido"),
            ("Vector GCS", "vector comprimido (productos vectorizables)"),
            ("Vector GEE", "FeatureCollection en Earth Engine"),
            ("Public mosaic", "COG reflejado en el bucket público"),
            ("Public vector", "ZIP reflejado en el bucket público"),
            ("Clean temp", "tiles temporales eliminados"),
        ],
        "links": "Las insignias 🔗 OK abren el enlace de descarga.",
        "legend": "OK = etapa completada  |  MISS = pendiente  |  N/A = no aplica",
    },
    "NL": {
        "name": "Nederlands",
        "tab_title": "Handleiding: Nederlands",
        "welcome_note": "Welkom! De interface van de applicatie is standaard in het Engels, maar we hebben deze complete documentatie in het Nederlands voorbereid om u te begeleiden bij elke stap — van het kiezen van het land en product tot de uiteindelijke publicatie. Verken gerust de tabbladen bovenaan en raadpleeg deze handleiding wanneer u dat wilt.",
        "what": "Deze app exporteert de MapBiomas Fire-brandkaarten: van Earth Engine naar cloud "
                "storage, bouwt mozaïeken per eenheid (band/beeld), vectoriseert (indien van "
                "toepassing), publiceert naar Earth Engine en de publieke bucket en verwijdert "
                "tijdelijke bestanden.",
        "howto_title": "Hoe te gebruiken",
        "steps": [
            "Kies land, thema, collectie en product.",
            "Vink in het product de eenheden (banden of beelden) aan.",
            "Voer de stappen uit: Export, Mosaic, Vectorize (indien van toepassing), Upload GEE, Public mosaic, Public vector, Clean temp.",
        ],
        "cols_title": "Grid-kolommen",
        "cols": [
            ("Export", "eenheid geëxporteerd (temp/)"),
            ("Mosaic", "gebouwde COG"),
            ("Vector GCS", "gezipte vector (vectoriseerbare producten)"),
            ("Vector GEE", "FeatureCollection in Earth Engine"),
            ("Public mosaic", "COG gespiegeld naar publieke bucket"),
            ("Public vector", "ZIP gespiegeld naar publieke bucket"),
            ("Clean temp", "tijdelijke tiles verwijderd"),
        ],
        "links": "🔗 OK-badges openen de downloadlink.",
        "legend": "OK = fase klaar  |  MISS = in afwachting  |  N/A = niet van toepassing",
    },
    "ZH": {
        "name": "中文",
        "tab_title": "指南: 中文",
        "welcome_note": "欢迎使用！应用界面默认为英语，但我们准备了完整的中文文档，引导您完成每一个步骤——从选择国家和产品到最终发布。请随意探索上方的标签页，并在需要时随时查阅本指南。",
        "what": "此应用导出 MapBiomas Fire 火灾地图：从 Earth Engine 到云存储，按单元（波段/影像）构建镶嵌图，在适用时进行矢量化，发布到 Earth Engine 和公共存储桶，并删除临时文件。",
        "howto_title": "使用方法",
        "steps": [
            "选择国家、主题、集合和产品。",
            "在产品中勾选所需的单元（波段或影像）。",
            "按顺序执行：导出、镶嵌、矢量化（如适用）、上传 GEE、公开镶嵌、公开矢量、清理临时文件。",
        ],
        "cols_title": "网格列",
        "cols": [
            ("Export", "导出的单元 (temp/)"),
            ("Mosaic", "构建的 COG"),
            ("Vector GCS", "压缩矢量（可矢量化产品）"),
            ("Vector GEE", "Earth Engine 中的 FeatureCollection"),
            ("Public mosaic", "镜像到公共存储桶的 COG"),
            ("Public vector", "镜像到公共存储桶的 ZIP"),
            ("Clean temp", "合并后删除的临时瓦片"),
        ],
        "links": "🔗 OK 徽章打开下载链接。",
        "legend": "OK = 阶段完成 | MISS = 待处理 | N/A = 不适用",
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
    legend = g["legend"]
    welcome = g.get("welcome_note", "")
    welcome_html = (
        f'<div style="background:{p["hint_bg"]};border:1px solid {p["hint_border"]};'
        f'border-radius:4px;padding:10px;margin-bottom:12px;font-size:12px;color:{p["hint_fg"]};line-height:1.5;">'
        f'{welcome}</div>'
    ) if welcome else ""
    return (
        f'<div style="font-size:12px;color:{p["guide_fg"]};line-height:1.6;">'
        f'{welcome_html}'
        f'<p><b>{g["name"]}</b> — {g["what"]}</p>'
        f'<h4 style="margin:10px 0 4px 0;">{g["howto_title"]}</h4>'
        f'<ol style="margin:0 0 8px 0;padding-left:20px;">{steps}</ol>'
        f'<p><span style="background:#28a745;color:#fff;padding:1px 6px;border-radius:3px;'
        f'font-size:10px;">OK</span> / '
        f'<span style="background:#e9ecef;color:#6c757d;padding:1px 6px;border-radius:3px;'
        f'font-size:10px;">MISS</span> / '
        f'<span style="background:#f1f3f5;color:#adb5bd;padding:1px 6px;border-radius:3px;'
        f'font-size:10px;">N/A</span> — {legend}</p>'
        f'<h4 style="margin:10px 0 4px 0;">{g["cols_title"]}</h4>'
        f'<table style="border-collapse:collapse;">{cols_rows}</table>'
        f'<p>{g["links"]}</p>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Grid de unidades de um produto
# ---------------------------------------------------------------------------
def _catalog_units(country, theme, collection, product):
    try:
        from .catalog import build_inventory
        inv = build_inventory([country])
        for p in inv.get(country, {}).get(theme, {}).get(collection, []):
            if p.get("name") == product:
                return [u.get("key") for u in (p.get("units") or [])]
    except Exception:
        pass
    return []


class UnitGridPanel:
    _DATE_W = "230px"
    _CELL_W = "88px"
    _SEL_W  = "72px"

    def __init__(self, country, theme, collection, log_area=None, on_data_loaded_change=None):
        self.country = country
        self.theme = theme
        self.collection = collection
        self.product = None
        self.units = []
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.year_filter = None
        self.log_area = log_area
        self._on_data_loaded_change = on_data_loaded_change

        self.grid_container = widgets.VBox([
            _loading_html("Loading units...")
        ])

        self.btn_sync = widgets.Button(description="Sync", button_style="success", icon="refresh",
                                       layout=L(width="120px", height="34px"))
        self.btn_sync.on_click(self._on_sync)
        self.btn_select_pending = widgets.Button(description="Select Pending", button_style="info",
                                                 layout=L(width="150px", height="34px"))
        self.btn_select_pending.on_click(self._on_select_pending)
        self.btn_select_all = widgets.Button(description="Select All", button_style="info",
                                             layout=L(width="120px", height="34px"))
        self.btn_select_all.on_click(self._on_select_all)
        self.btn_clear = widgets.Button(description="Clear", button_style="warning",
                                        layout=L(width="90px", height="34px"))
        self.btn_clear.on_click(self._on_clear)

        self.year_dropdown = widgets.Dropdown(options=["All units"], value="All units",
                                              description="Year:", layout=L(width="200px"))
        self.year_dropdown.observe(self._on_year_change, names="value")

        self.story_loader = StoryLoader("Checking GCS and Earth Engine...")
        self.btn_load_data = widgets.Button(
            description="Load Data", button_style="danger", icon="download",
            layout=L(width="130px", height="34px"),
            tooltip="Discover bands/units from GEE and GCS for this product"
        )
        self.btn_load_data.on_click(self._on_load_data)
        self.toolbar = widgets.HBox([
            self.year_dropdown, self.btn_select_pending, self.btn_select_all,
            self.btn_clear, self.btn_sync, self.btn_load_data, self.story_loader.widget,
        ], layout=L(margin="0 0 8px 0", gap="8px", align_items="center"))

        self._data_loaded = False
        self.container = widgets.VBox([_STATUS_CSS, self.toolbar, self.grid_container])
        self._render_panel()

    def _render_panel(self):
        p = _palette()
        self.container.layout = L(border=f"1px solid {p['panel_border']}", padding="10px",
                                  border_radius="5px", margin="6px 0", background=p["panel_bg"])
        self._render_grid()

    def _log(self, message, type="info"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colors = {"info": "#3498db", "success": "#27ae60", "error": "#d32f2f", "warning": "#e67e22"}
        color = colors.get(type, "#333")
        html = f'<span style="color:{color};font-size:12px;">[{ts}] [{type.upper()}] {message}</span>'
        if self.log_area:
            self.log_area.append(html)

    def _activate_product(self, product):
        if not product:
            return
        config.set_country(self.country, verbose=False)
        config.set_theme(self.theme)
        config.set_collection(self.collection)
        config.set_product(product)
        self.product = product
        self.units = _catalog_units(self.country, self.theme, self.collection, product)
        if not self.units and config.product_kind() == "monthly":
            self.units = list_months_in_collection()
        # Keep existing state if available, don't auto-sync
        if not hasattr(self, 'state') or not self.state:
            self.state = {"updated_at": None}
        self._data_loaded = False
        self._update_load_data_button_style()
        # Notify parent ProductTabs to update tab style
        self._notify_tab_style()

    def _on_load_data(self, _):
        """Explicitly load data (discover bands/units from GEE/GCS)."""
        if self.is_refreshing:
            return
        if not self.product:
            return
        self._data_loaded = True
        self._update_load_data_button_style()
        self._notify_tab_style()
        self._on_sync(None)

    def _update_load_data_button_style(self):
        """Update Load Data button style based on loaded state."""
        if self._data_loaded:
            self.btn_load_data.button_style = "success"
            self.btn_load_data.description = "Data Loaded"
            self.btn_load_data.icon = "check"
            # Remove pulsing outline
            if hasattr(self.btn_load_data, 'remove_class'):
                self.btn_load_data.remove_class('mfm-btn-unloaded')
        else:
            self.btn_load_data.button_style = "danger"
            self.btn_load_data.description = "Load Data"
            self.btn_load_data.icon = "download"
            # Add pulsing outline
            if hasattr(self.btn_load_data, 'add_class'):
                self.btn_load_data.add_class('mfm-btn-unloaded')

    def _notify_tab_style(self):
        """Notify parent ProductTabs to update tab style for this product."""
        if self._on_data_loaded_change:
            self._on_data_loaded_change(self.product, self._data_loaded)

    def _all_units(self):
        keys = set(self.units) | {k for k in self.state.keys() if k != "updated_at"}
        return sorted(keys, reverse=True)

    def _filtered_units(self):
        units = self._all_units()
        if self.year_filter is not None:
            units = [u for u in units if str(u).startswith(f"{self.year_filter}")]
        return units

    def _refresh_year_dropdown(self):
        years = set()
        for u in self._all_units():
            s = str(u)
            if s[:4].isdigit():
                years.add(s[:4])
        options = ["All units"] + sorted(years, reverse=True)
        if self.year_dropdown.value not in options:
            self.year_dropdown.value = "All units"
        self.year_dropdown.options = options

    def _on_year_change(self, change):
        value = change.get("new")
        self.year_filter = int(value) if value != "All units" else None
        selected = self._get_selected_keys()
        self._render_grid()
        self._restore_selected(selected)

    def _col_content(self, kind, ok, unit):
        if not config.is_vectorizable() and kind in _VECTOR_KINDS:
            return _badge_na()
        if not ok:
            return _badge(False)
        if kind == "link_mosaic":
            url = f"https://storage.googleapis.com/{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name_unit(unit)}.tif"
            return _badge_link(url)
        if kind == "link_vector":
            url = f"https://storage.googleapis.com/{config.BUCKET}/{config.vector_prefix()}/{config.vector_name_unit(unit)}.zip"
            return _badge_link(url)
        if kind == "link_pub_mosaic":
            url = f"https://storage.googleapis.com/{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name_unit(unit)}.tif"
            return _badge_link(url)
        if kind == "link_pub_vector":
            url = f"https://storage.googleapis.com/{config.PUBLIC_BUCKET}/{config.vector_prefix()}/{config.vector_name_unit(unit)}.zip"
            return _badge_link(url)
        if kind == "copy_asset":
            asset_id = f"{config.vector_asset_prefix()}/{config.vector_name_unit(unit)}"
            return _badge_copy(asset_id)
        return _badge(True)

    def _render_grid(self):
        self.chk_dict = {}
        p = _palette()

        def _header_cell(width, title, etapa, is_vector=False):
            vec_muted = is_vector and not config.is_vectorizable()
            bg = p["grid_header_bg"] if not vec_muted else p.get("grid_header_muted_bg", "#4a4a4a")
            fg = p["grid_header_fg"] if not vec_muted else p.get("grid_header_muted_fg", "#999999")
            return widgets.HTML(
                f'<div style="width:{width};height:42px;background:{bg};'
                f'text-align:center;font-weight:700;font-size:11px;padding:5px 3px;'
                f'color:{fg};line-height:1.25;box-sizing:border-box;'
                f'border-right:1px solid {p["sep"]};">'
                f'{title}<br><span style="font-size:9px;font-weight:400;opacity:.6;">Step {etapa}</span>'
                f'</div>'
            )

        vectorizable = config.is_vectorizable()
        header_row = widgets.HBox(
            [widgets.HTML(f'<div style="width:{self._DATE_W};height:42px;background:{p["grid_header_bg"]};'
                          f'font-weight:700;font-size:12px;color:{p["grid_header_fg"]};padding:12px 6px;'
                          f'box-sizing:border-box;border-left:1px solid {p["sep"]};'
                          f'border-right:1px solid {p["sep"]};">Unit</div>')]
            + [_header_cell(self._CELL_W, t, e, kind in _VECTOR_KINDS) for _, t, e, kind in _COLS]
            + [widgets.HTML(f'<div style="width:{self._SEL_W};height:42px;background:{p["grid_header_bg"]};'
                            f'text-align:center;font-weight:700;font-size:11px;color:{p["grid_header_fg"]};padding:12px 3px;'
                            f'box-sizing:border-box;border-right:1px solid {p["sep"]};">Select</div>')],
            layout=L(background=p["grid_header_bg"], padding="6px 10px", min_height="44px",
                     align_items="center", overflow="visible")
        )

        rows = [header_row]
        self._refresh_year_dropdown()
        units = self._filtered_units()
        row_layout = L(align_items="center", min_height="38px",
                       border_bottom=f"1px solid {p['border']}", padding="3px 10px",
                       overflow="visible", width="100%")

        for i, unit in enumerate(units):
            info = self.state.get(unit, {})
            bg = p["row_a"] if i % 2 == 0 else p["row_b"]
            unit_text = str(unit)
            unit_short = unit_text if len(unit_text) <= 34 else unit_text[:31] + "..."
            date_cell = widgets.HTML(
                f'<div title="{unit_text}" style="width:{self._DATE_W};font-family:monospace;font-size:12px;color:{p["date_fg"]};font-weight:600;'
                f'background:{p["date_bg"]};padding:2px 6px;border-radius:3px;box-sizing:border-box;'
                f'border-left:1px solid {p["sep"]};border-right:1px solid {p["sep"]};overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{unit_short}</div>'
            )
            cells = [date_cell]
            for key, _t, _e, kind in _COLS:
                ok = info.get(key, False)
                cells.append(widgets.HTML(
                    f'<div style="width:{self._CELL_W};text-align:center;box-sizing:border-box;'
                    f'border-right:1px solid {p["sep"]};">{self._col_content(kind, ok, unit)}</div>'
                ))
            chk = widgets.Checkbox(value=False, indent=False, layout=L(width="20px", height="20px"))
            chk_wrapper = widgets.HBox([chk], layout=L(
                width=f"calc({self._SEL_W} - 2px)", justify_content="center",
                align_items="center", overflow="hidden", border=f"1px solid {p['sep']}"
            ))
            self.chk_dict[unit] = chk
            row = widgets.HBox(cells + [chk_wrapper], layout=row_layout)
            row.layout.background = bg
            rows.append(row)

        n_all = len(self._all_units())
        n_visible = len(units)
        n_complete = sum(1 for u in self._all_units() if _is_complete(self.state.get(u, {})))

        if self.year_filter is not None:
            label = (f'{n_visible} units of {self.year_filter} in filter &nbsp;|&nbsp; '
                     f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> complete')
        else:
            label = (f'{n_all} units &nbsp;|&nbsp; '
                     f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> complete')

        legend = widgets.HTML(
            f'<div style="font-size:11px;color:{p["legend_fg"]};margin:6px 0 0 10px;padding:6px 10px;'
            f'background:{p["legend_bg"]};border-radius:4px;">{label}</div>'
        )
        hint = widgets.HTML(
            f'<div style="font-size:11px;color:{p["hint_fg"]};margin:4px 0 0 10px;padding:6px 10px;'
            f'background:{p["hint_bg"]};border:1px solid {p["hint_border"]};border-radius:4px;line-height:1.5;">'
            f'<strong>MISS &rarr; OK:</strong> Export=Step 1 &nbsp;|&nbsp; Mosaic=Step 2 &nbsp;|&nbsp; '
            f'Vector GCS=Step 3 &nbsp;|&nbsp; Vector GEE=Step 4 &nbsp;|&nbsp; Public mosaic=Step 5 &nbsp;|&nbsp; '
            f'Public vector=Step 6 &nbsp;|&nbsp; Clean temp=Step 7 (after both published)'
            f'</div>'
        )
        meta = config.product_context()
        example = self.units[0] if self.units else "-"
        metadata = widgets.HTML(
            f'<div style="font-size:11px;color:{p["guide_fg"]};margin:6px 0 0 10px;padding:8px 10px;'
            f'background:{p["guide_bg"]};border:1px solid {p["guide_border"]};border-radius:4px;line-height:1.5;">'
            f'<b>Product:</b> {meta["product"]} &nbsp;|&nbsp; '
            f'<b>Example unit/band:</b> <code title="{example}">{example}</code><br>'
            f'<b>Asset:</b> <code>{meta["assetid"]}</code> &nbsp;|&nbsp; '
            f'<b>Type:</b> {meta["type"]} &nbsp;|&nbsp; <b>Scale:</b> {meta["scale"]} m &nbsp;|&nbsp; '
            f'<b>Vectorize:</b> {"yes" if meta["vectorize"] else "no"}</div>'
        )
        self.grid_container.children = [
            widgets.VBox(rows, layout=L(max_height="460px", width="100%",
                                        overflow_y="auto", overflow_x="auto", padding="0",
                                        border=f"1px solid {p['border']}", background_color=p["row_b"])),
            legend, metadata, hint,
        ]

    def _on_sync(self, _):
        if self.is_refreshing:
            return
        if not self.product:
            return
        config.set_country(self.country, verbose=False)
        config.set_theme(self.theme)
        config.set_collection(self.collection)
        config.set_product(self.product)
        self.is_refreshing = True
        self.btn_sync.disabled = True
        self.btn_sync.description = "Syncing..."
        self.story_loader.label = "Checking GCS and Earth Engine..."
        self.story_loader.start()
        self._log("Checking files in GCS and assets in GEE...", "info")
        try:
            selected = self._get_selected_keys()
            self.state = build_state(
                country=self.country,
                theme=self.theme,
                collection=self.collection,
                product=self.product,
                logger=self._log,
            )
            self._log(f"[DEBUG] UI units={self.units}", "info")
            self._log(f"[DEBUG] state units={[u for u in self.state if u != 'updated_at']}", "info")
            self._render_grid()
            self._restore_selected(selected)
            n_ok = sum(1 for u in self._all_units() if _is_complete(self.state.get(u, {})))
            result = f"Sync complete: {n_ok}/{len(self._all_units())} units complete."
            self.story_loader.stop(result)
            self._log(result, "success")
            # Mark data as loaded and update UI
            self._data_loaded = True
            self._update_load_data_button_style()
            self._notify_tab_style()
        except Exception as e:
            self.story_loader.stop("Sync failed")
            self._log(f"Sync error: {e}", "error")
        finally:
            self.is_refreshing = False
            self.btn_sync.disabled = False
            self.btn_sync.description = "Sync"

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

    def get_selected_units(self):
        return [k for k, chk in self.chk_dict.items() if chk.value]

    def _get_selected_keys(self):
        return [k for k, chk in self.chk_dict.items() if chk.value]

    def _restore_selected(self, keys):
        for k in keys:
            if k in self.chk_dict:
                self.chk_dict[k].value = True

    def sync(self):
        self._on_sync(None)


# ---------------------------------------------------------------------------
# Navegacao: pais -> tema -> colecao -> produto (abas) -> grid de unidades
# ---------------------------------------------------------------------------
class ProductTabs:
    """Abas de produtos visiveis + um grid independente por produto."""

    def __init__(self, country, theme, collection, log_area=None, on_data_loaded_change=None):
        self.country = country
        self.theme = theme
        self.collection = collection
        self.log_area = log_area
        self._on_data_loaded_change = on_data_loaded_change
        products = config.list_products(country, theme, collection)
        self.products = [p["product"] for p in products if p.get("visible", True)]
        self._panels = {}
        self._placeholders = [widgets.VBox([]) for _ in self.products]
        self.tab = widgets.Tab(children=self._placeholders)
        for index, product in enumerate(self.products):
            self.tab.set_title(index, product)
        self.tab.observe(self._on_product_tab, names="selected_index")
        self._active_panel = None
        self._loaded_products = set()
        if self.products:
            self._activate_product(0)
        self.container = self.tab

    def _on_product_tab(self, change):
        index = change.get("new")
        if index is not None:
            self._activate_product(index)

    def _activate_product(self, index):
        if index < 0 or index >= len(self.products):
            return
        product = self.products[index]
        if product not in self._panels:
            panel = UnitGridPanel(
                self.country, self.theme, self.collection, self.log_area,
                on_data_loaded_change=self._on_panel_data_loaded
            )
            panel._activate_product(product)
            self._panels[product] = panel
            self._placeholders[index].children = [panel.container]
        else:
            panel = self._panels[product]
        self._active_panel = panel
        # Update tab style for this product
        self._update_tab_style(product)

    def _on_panel_data_loaded(self, product, loaded):
        """Callback when a panel's data loaded state changes."""
        if loaded:
            self._loaded_products.add(product)
        self._update_tab_style(product)

    def _update_tab_style(self, product):
        """Update tab style based on whether data is loaded for this product."""
        # Button now handles its own pulsing; tabs don't need special styling
        pass

    def __getattr__(self, name):
        if name == "_active_panel":
            raise AttributeError(name)
        panel = self.__dict__.get("_active_panel")
        if panel is None:
            raise AttributeError(name)
        return getattr(panel, name)


class CollectionTabs:
    """Abas de colecao dentro de um tema."""

    def __init__(self, country, theme, log_area=None):
        self.country = country
        self.theme = theme
        self.log_area = log_area
        colls = [c for c, prods in config.OBJ.get(country, {}).get(theme, {}).items()
                 if [p for p in prods if p.get("visible", True)]]
        self.collections = colls
        self._panels = {}
        self._placeholders = [widgets.VBox([]) for _ in colls]
        self.tab = widgets.Tab(children=self._placeholders)
        for i, c in enumerate(colls):
            self.tab.set_title(i, c)
        self.tab.observe(self._on_tab_change, names="selected_index")
        self._active_panel = None
        if colls:
            self._activate(0)

    def _on_tab_change(self, change):
        idx = change.get("new")
        if idx is None:
            return
        self._activate(idx)

    def _activate(self, idx):
        coll = self.collections[idx]
        if coll not in self._panels:
            self._placeholders[idx].children = [_loading_html("Loading products...")]
            pp = ProductTabs(self.country, self.theme, coll, self.log_area)
            self._panels[coll] = pp
            self._placeholders[idx].children = [pp.container]
        else:
            pp = self._panels[coll]
        self._active_panel = pp

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class ThemeTabs:
    """Abas de tema dentro de um pais."""

    def __init__(self, country, log_area=None):
        self.country = country
        self.log_area = log_area
        themes = [t for t, colls in config.OBJ.get(country, {}).items()
                  if any([p for p in prods if p.get("visible", True)] for prods in colls.values())]
        self.themes = themes
        self._panels = {}
        self._placeholders = [widgets.VBox([]) for _ in themes]
        self.tab = widgets.Tab(children=self._placeholders)
        for i, t in enumerate(themes):
            self.tab.set_title(i, t)
        self.tab.observe(self._on_tab_change, names="selected_index")
        self._active_panel = None
        if themes:
            self._activate(0)

    def _on_tab_change(self, change):
        idx = change.get("new")
        if idx is None:
            return
        self._activate(idx)

    def _activate(self, idx):
        theme = self.themes[idx]
        if theme not in self._panels:
            self._placeholders[idx].children = [_loading_html("Loading collections...")]
            ct = CollectionTabs(self.country, theme, self.log_area)
            self._panels[theme] = ct
            self._placeholders[idx].children = [ct.tab]
        else:
            ct = self._panels[theme]
        self._active_panel = ct

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class CountryTabs:
    """Abas de pais -> tema -> colecao -> produto -> unidades."""

    def __init__(self, countries, log_area=None):
        self.countries = list(countries)
        self.log_area = log_area
        if not self.countries:
            raise ValueError("No countries configured for the tabs.")
        for c in self.countries:
            if c not in config.OBJ:
                raise ValueError(f"Country '{c}' not in config.OBJ.")

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
            self._placeholders[idx].children = [_loading_html("Loading themes...")]
            tt = ThemeTabs(code, self.log_area)
            self._panels[code] = tt
            self._placeholders[idx].children = [tt.tab]
        else:
            tt = self._panels[code]
        self._active_panel = tt

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class FireMonitorApp:
    """App em guias: Interface (navegacao) + guias em 7 idiomas."""

    def __init__(self, countries):
        self.header = widgets.HTML()
        self.log_area = LogDrawer()
        self.interface = CountryTabs(countries, self.log_area)
        self.guide_widgets = [widgets.HTML() for _ in LANG_ORDER]

        self.tab = widgets.Tab(children=[self.interface.tab] + self.guide_widgets)
        self.tab.set_title(0, "Interface")
        for i, lang in enumerate(LANG_ORDER, start=1):
            self.tab.set_title(i, GUIDES[lang]["tab_title"])

        self.container = widgets.VBox([self.header, self.tab, self.log_area.container])
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
        self.container.children = [self.header, self.tab, self.log_area.container]
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
    app.display()
    app.interface._activate(0)
    return app
