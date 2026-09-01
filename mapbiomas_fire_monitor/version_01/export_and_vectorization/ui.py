import datetime
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import ipywidgets as widgets
from IPython.display import display, clear_output

from . import catalog
from . import config
from .state import build_state, load_state

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
.mfm-tab-btn { border:1px solid #ced4da !important; }
.mfm-tab-active { box-shadow: inset 0 0 0 2px #263238 !important; }
</style>""")

class ProgressLoader:
    """Spinner padrao (CSS mfm-spin) + status de progresso real.

    A rotacao e feita por CSS (sem thread); o texto muda via `set_status`
    conforme as etapas do scan avancam. Nao bloqueia o kernel.
    """

    def __init__(self, label="Ready."):
        self.label = label
        self._status = ""
        self._active = False
        self.widget = widgets.HTML()
        self._render()

    def _render(self, message=None, color="#6c757d"):
        status = message if message is not None else (self._status or self.label)
        if self._active:
            html = (
                '<div style="padding:8px 14px;color:#3498db;font-size:13px;'
                'display:flex;align-items:center;gap:8px;line-height:1.5;">'
                '<span style="display:inline-block;width:14px;height:14px;flex:0 0 auto;'
                'border:2px solid #b9d7f0;border-top-color:#3498db;border-radius:50%;'
                'animation:mfm-spin 0.8s linear infinite;"></span>'
                f'<span>{status}</span></div>'
            )
        else:
            # Ocioso: texto estatico sem spinner (nao fica girando a toa).
            html = (f'<div style="padding:8px 14px;color:{color};font-size:12px;">'
                    f'{status}</div>')
        self.widget.value = html

    def start(self, status=None):
        if status:
            self._status = status
        self._active = True
        self._render()
        return self.widget

    def set_status(self, status):
        self._status = status
        if self._active:
            self._render()

    def stop(self, message=None):
        self._active = False
        if message:
            self._render(message, color="#28a745")
        else:
            self.widget.value = ""


class LogDrawer:
    """Guia global de logs: cauda limitada em tela + historico completo exportavel.

    Custo por mensagem e O(1) em memoria de widget: um unico HTML e
    re-renderizado (throttled) com a cauda do ring buffer, em vez de
    empilhar widgets no DOM. O historico completo da sessao fica em uma
    lista Python pura e pode ser baixado como .txt.
    """

    MAX_RENDERED = 500      # linhas mantidas na aba "Log history"
    FLUSH_INTERVAL = 0.4    # segundos entre renders (throttle)

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=self.MAX_RENDERED)
        self._full_history = []
        self._last_flush = 0.0
        self._flush_timer = None
        self._flush_gen = 0

        self.last = widgets.HTML(
            value="<div style='padding:5px 8px;color:#6c757d;'>No messages yet.</div>")
        self.log_view = widgets.HTML()
        btn_export = widgets.Button(
            description="\u2913 Export log (.txt)", icon="download",
            layout=L(height="28px"),
            tooltip="Download the complete session log as a .txt file")
        btn_export.on_click(self._export_log)
        self.btn_catalog = widgets.Button(
            description="\u2913 Catalog cache (.json)", icon="download",
            layout=L(height="28px"),
            tooltip="Download catalog_cache.json (units/bands loaded this session) to version in GitHub")
        self.btn_catalog.on_click(self._download_catalog)
        toolbar = widgets.HBox(
            [btn_export, self.btn_catalog],
            layout=L(display="flex", justify_content="flex-end", gap="8px"))
        self.output = widgets.VBox(
            [toolbar, self.log_view],
            layout=L(max_height="260px", overflow="auto", padding="4px",
                     border="1px solid #cccccc", background="#ffffff"))
        self.tab = widgets.Tab(children=[self.last, self.output])
        self.tab.set_title(0, "Last log")
        self.tab.set_title(1, "Log history")
        self.container = widgets.VBox([self.tab])

    @property
    def history(self):
        """Historico completo da sessao (todas as mensagens, sem poda)."""
        return self._full_history

    # ------------------------------------------------------------------ API
    def append(self, message, level=None):
        verbose = bool(getattr(config, "LOG_VERBOSE", False))
        if not verbose and "[DEBUG]" in str(message):
            return
        message = str(message)
        urgent = (level in ("warning", "error")
                  or "[ERROR]" in message or "[WARN]" in message)
        now = time.monotonic()
        with self._lock:
            self._buffer.append(message)
            self._full_history.append(message)
            self.last.value = (
                f'<div style="padding:5px 8px;color:#495057;font-size:12px;">'
                f'{message}</div>')
            if urgent or (now - self._last_flush) >= self.FLUSH_INTERVAL:
                self._render_locked()
            elif self._flush_timer is None:
                delay = max(0.05, self.FLUSH_INTERVAL - (now - self._last_flush))
                self._schedule_flush(delay)

    # ------------------------------------------------------------ internals
    def _schedule_flush(self, delay):
        gen = self._flush_gen
        timer = threading.Timer(delay, self._deferred_flush, args=(gen,))
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _deferred_flush(self, gen):
        with self._lock:
            self._flush_timer = None
            if gen != self._flush_gen:
                return  # um render mais recente ja aconteceu
            self._render_locked()

    def _render_locked(self):
        self._flush_gen += 1
        self._last_flush = time.monotonic()
        self.log_view.value = "".join(
            f'<div style="padding:1px 8px;color:#495057;font-size:12px;'
            f'border-bottom:1px solid #f1f3f5;">{m}</div>'
            for m in self._buffer)

    def _export_log(self, _=None):
        tag_re = re.compile(r"<[^>]+>")
        with self._lock:
            lines = [tag_re.sub("", m)
                     .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                     for m in self._full_history]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"mapbiomas_fire_log_{stamp}.txt"
        try:
            from google.colab import files as colab_files  # noqa: F401
            path = fname
        except Exception:
            path = os.path.join(os.getcwd(), fname)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except Exception as exc:
            print("Log export failed:", exc)
            return
        try:
            from google.colab import files as colab_files
            colab_files.download(path)
        except Exception:
            print(f"Log saved to: {path}")

    def _download_catalog(self, _=None):
        set_button_busy(self.btn_catalog, True, "Preparing...")
        try:
            catalog.download_cache(countries=config.COUNTRIES_AVAILABLE, logger=self.append)
        except Exception as exc:
            self.append(f"[ERROR] Catalog download failed: {exc}", "error")
        finally:
            set_button_busy(self.btn_catalog, False, "\u2913 Catalog cache (.json)")


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


def set_button_busy(button, busy, busy_text=None):
    """Estado padronizado de botao durante operacao longa: desabilita e mostra
    o icone de spinner; ao concluir, restaura icone/descricao originais."""
    if busy:
        button._busy_icon = getattr(button, "icon", "")
        button._busy_text = getattr(button, "description", "")
        button.disabled = True
        button.icon = "spinner"
        if busy_text:
            button.description = busy_text
    else:
        button.disabled = False
        button.icon = getattr(button, "_busy_icon", "")
        button.description = getattr(button, "_busy_text", getattr(button, "description", ""))


# (state key, column title, step number, badge type)
# Step order:
#   annual_burned (vectorizable): 1=Export, 2=Mosaic, 3=Public Mosaic, 4=Clean Temp,
#                                 5=Vector GCS, 6=Vector GEE, 7=Public Vector
#   other products (default)     : 1=Export, 2=Mosaic, 3=Public Mosaic, 4=Clean Temp
# Vector steps (5,6,7) are optional and only shown for vectorizable products.
_COLS_VECTORIZABLE = [
    ("exported",         "Export",          1, "badge"),
    ("mosaiced",         "Mosaic",          2, "link_mosaic"),
    ("published_mosaic", "Public mosaic",   3, "link_pub_mosaic"),
    ("temp_cleaned",     "Clean temp",      4, "badge"),
    ("vectorized_gcs",   "Vector GCS",      5, "link_vector"),
    ("vectorized_gee",   "Vector GEE",      6, "copy_asset"),
    ("published_vector", "Public vector",   7, "link_pub_vector"),
]

_COLS_DEFAULT = [
    ("exported",         "Export",          1, "badge"),
    ("mosaiced",         "Mosaic",          2, "link_mosaic"),
    ("published_mosaic", "Public mosaic",   3, "link_pub_mosaic"),
    ("temp_cleaned",     "Clean temp",      4, "badge"),
]


def _product_cols():
    return _COLS_VECTORIZABLE if config.is_vectorizable() else _COLS_DEFAULT

_VECTOR_KINDS = {"link_vector", "copy_asset", "link_pub_vector"}


def _unit_pending_key(entry):
    """Primeira etapa (chave de estado) pendente de uma unidade, na ordem das
    colunas do produto. `None` = unidade completa. Por definicao, uma etapa so
    e 'pendente' quando todas as anteriores ja foram processadas."""
    for key, _t, _n, _b in _product_cols():
        if not (entry or {}).get(key, False):
            return key
    return None


def _step_title(key):
    for k, t, _n, _b in _product_cols():
        if k == key:
            return t
    return key


def _wrap_tab_bar(titles, on_activate, per_line=None):
    """Barra de abas em botoes com quebra deterministica por linha.

    Divide os titulos em linhas de ate `per_line` (default
    config.PRODUCT_TABS_PER_LINE). Cada botao mantem o tamanho exato do titulo
    (flex 0 0 auto) — nada de comprimir/elidir; se os `per_line` nao couberem
    na largura da tela, a linha ainda quebra antes via flex_wrap. Retorna o
    container (VBox de linhas) e a lista plana de botoes (indice paralelo aos
    titulos, para estilizar por indice)."""
    per_line = per_line or getattr(config, "PRODUCT_TABS_PER_LINE", 10)
    btns = []
    for i, title in enumerate(titles):
        b = widgets.Button(description=title,
                           layout=L(height="32px", margin="0 2px 2px 0",
                                    flex="0 0 auto"))
        b.style.button_color = "#f8f9fa"      # nao carregado
        b.style.font_color = "#212529"         # texto escuro (contraste)
        if hasattr(b, "add_class"):
            b.add_class("mfm-tab-btn")
        b.on_click(lambda _b, idx=i: on_activate(idx))
        btns.append(b)

    rows = []
    for start in range(0, len(btns), per_line):
        row = widgets.HBox(btns[start:start + per_line],
                           layout=L(flex_wrap="wrap", gap="4px",
                                    margin="0 0 2px 0", align_items="center"))
        rows.append(row)
    bar = widgets.VBox(rows, layout=L(margin="0 0 6px 0"))
    return bar, btns


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

GUIDES = {'PT': {'name': 'Português',
        'tab_title': 'Guia: Português',
        'welcome_note': 'Bem-vindo! Esta guia em português explica o aplicativo Export & Vectorization do '
                        'Monitor do Fogo MapBiomas: como navegar (país → tema → coleção → produto), '
                        'descobrir as unidades, processar cada etapa e publicar os mapas. As abas acima '
                        'mostram a interface; use esta guia sempre que precisar.',
        'what': 'O aplicativo exporta os mapas de área queimada/incêndio do MapBiomas Fire: do Earth Engine '
                '(GEE) para o GCS, monta mosaicos (COG) por unidade (banda ou imagem), vetoriza quando '
                'aplicável, publica no Earth Engine e no bucket público e remove os arquivos temporários.',
        'howto_title': 'Como usar',
        'steps': ['<b>Navegue</b> pelas abas: país → tema → coleção → produto. Com muitos '
                  'produtos, as guias quebram em várias linhas — verde = carregado, '
                  'cinza = não carregado.',
                  'Clique em <b>Load Data</b> para carregar o produto atual (unidades da memória, '
                  'descobrir dados novos e verificar o status, com limite de '
                  '<code>SCAN_TIMEOUT</code>) — ou em <b>Load Collection</b> para carregar todos os '
                  'produtos da coleção em fila.',
                  'Na grade (checkbox na primeira coluna), marque as unidades desejadas. Use '
                  '<b>Select Pending</b> para selecionar por estágio — o título mostra o estágio-alvo '
                  'e cada clique avança no ciclo (mais avançado primeiro). Também há <b>Select All</b> '
                  'e <b>Select All Collection</b> (todos os produtos da coleção).',
                  'Execute as etapas na ordem: <b>Export → Mosaico → Publicar mosaico → Limpar temp</b>. '
                  'A vetorização é opcional (Steps 5–7, só produtos vetorizáveis, ex.: annual_burned): '
                  '<b>Vetor GCS → Vetor GEE → Publicar vetor</b>.',
                  'Para refazer uma etapa, ative <code>FORCE_&lt;ETAPA&gt; = True</code> na célula da etapa '
                  'e selecione as unidades na grade.',
                  'Para versionar a memória do catálogo (quando houver dados novos no '
                  '<code>config.py</code>), use o botão <b>⤓ Catalog cache (.json)</b> na barra inferior e '
                  'suba o arquivo no GitHub.'],
        'cols_title': 'Colunas da grade',
        'cols': [['Export', 'unidade exportada do GEE (temp/)'],
                 ['Mosaic', 'COG montado'],
                 ['Public mosaic', 'COG espelhado no bucket público'],
                 ['Vector GCS', 'vetor zipado (só produtos vetorizáveis)'],
                 ['Vector GEE', 'FeatureCollection no Earth Engine'],
                 ['Public vector', 'ZIP espelhado no bucket público'],
                 ['Clean temp', 'tiles temporários removidos após consolidação']],
        'links': 'Badges <b>🔗 OK</b> abrem o link de download; <b>Vector GEE</b> copia o asset ID.',
        'legend': 'OK = etapa concluída | MISS = etapa pendente | N/A = não se aplica',
        'graphs_title': 'Gráficos',
        'graphs': [{'title': 'Navegação',
                    'lines': ['País',
                              '└─ Tema',
                              '   └─ Coleção',
                              '      └─ Produto',
                              '         └─ Unidades (bandas ou imagens)']},
                   {'title': 'Etapas (produto vetorizável)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mosaico → COG',
                              ' ↓',
                              '3 Publicar mosaico → bucket público',
                              ' ↓',
                              '4 Limpar temp',
                              ' ↓',
                              '5 Vetor GCS → ZIP',
                              ' ↓',
                              '6 Vetor GEE → FeatureCollection',
                              ' ↓',
                              '7 Publicar vetor → ZIP público']},
                   {'title': 'Etapas (demais produtos)',
                    'lines': ['1 Export',
                              ' ↓',
                              '2 Mosaico',
                              ' ↓',
                              '3 Publicar mosaico',
                              ' ↓',
                              '4 Limpar temp']},
                   {'title': 'Memória do catálogo',
                    'lines': ['config.py (dado cru)',
                              ' ↓',
                              'Load Data (descobre unidades sob demanda)',
                              ' ↓',
                              'catalog_cache.json (memória entre sessões)',
                              ' ↓',
                              'Botão ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (versionar)']},
                   {'title': 'Fluxo de dados',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mosaico → COG',
                              ' ↓',
                              'Publicar → bucket público',
                              ' ↓',
                              '(se vetorizável) Vetorização → ZIP + upload GEE']}]},
 'ES': {'name': 'Español',
        'tab_title': 'Guía: Español',
        'welcome_note': '¡Bienvenido! Esta guía en español explica la aplicación Export & Vectorization del '
                        'Monitor del Fuego de MapBiomas: cómo navegar (país → tema → colección → producto), '
                        'descubrir las unidades, procesar cada Etapa y publicar los mapas. Las pestañas de '
                        'arriba muestran la interfaz; use esta guía siempre que la necesite.',
        'what': 'La aplicación exporta los mapas de área quemada/incendio de MapBiomas Fire: de Earth Engine '
                '(GEE) a GCS, monta mosaicos (COG) por unidad (banda o imagen), vectoriza cuando '
                'corresponde, publica en Earth Engine y en el bucket público y elimina los archivos '
                'temporales.',
        'howto_title': 'Cómo usar',
        'steps': ['<b>Navegue</b> por las pestañas: país → tema → colección → producto.',
                  'Haga clic en <b>Load Data</b> (botón rojo parpadeante) para descubrir las '
                  '<b>unidades</b>: bandas (imagen multibanda) o imágenes (ImageCollection). El '
                  'descubrimiento es bajo demanda; el caché no se llena con datos no cargados.',
                  'En la cuadrícula, marque las unidades deseadas. El filtro <code>Unit:</code> '
                  '(predeterminado “All units”) restringe por prefijo de unidad.',
                  'Haga clic en <b>Sync</b> para verificar el estado de las etapas. El escaneo se ejecuta en '
                  'segundo plano, con indicador de progreso (el kernel no se bloquea).',
                  'Ejecute las etapas en orden: <b>Export → Mosaico → Publicar mosaico → Vector GCS → Vector '
                  'GEE → Publicar vector → Limpiar temp</b>. Etapas 4–6 solo para productos vectorizables '
                  '(ej.: annual_burned); en los demás: <b>Export → Mosaico → Publicar mosaico → Limpiar '
                  'temp</b>.',
                  'Para rehacer una Etapa, active <code>FORCE_&lt;ETAPA&gt; = True</code> en la celda de la '
                  'Etapa y seleccione las unidades en la cuadrícula.',
                  'Para versionar la memoria del Catálogo (cuando haya datos nuevos en '
                  '<code>config.py</code>), use el botón <b>⤓ Catalog cache (.json)</b> en la barra inferior '
                  'y suba el archivo a GitHub.'],
        'cols_title': 'Columnas de la cuadrícula',
        'cols': [['Export', 'unidad exportada de GEE (temp/)'],
                 ['Mosaic', 'COG ensamblado'],
                 ['Public mosaic', 'COG replicado en el bucket público'],
                 ['Vector GCS', 'vector comprimido (solo productos vectorizables)'],
                 ['Vector GEE', 'FeatureCollection en Earth Engine'],
                 ['Public vector', 'ZIP replicado en el bucket público'],
                 ['Clean temp', 'tiles temporales eliminados tras consolidación']],
        'links': 'Insignias <b>🔗 OK</b> abren el enlace de descarga; <b>Vector GEE</b> copia el asset ID.',
        'legend': 'OK = Etapa completada | MISS = Etapa pendiente | N/A = no aplica',
        'graphs_title': 'Gráficos',
        'graphs': [{'title': 'Navegación',
                    'lines': ['País',
                              '└─ Tema',
                              '   └─ Colección',
                              '      └─ Producto',
                              '         └─ Unidades (bandas o imágenes)']},
                   {'title': 'Etapas (producto vectorizable)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mosaico → COG',
                              ' ↓',
                              '3 Publicar mosaico → bucket público',
                              ' ↓',
                              '4 Vector GCS → ZIP',
                              ' ↓',
                              '5 Vector GEE → FeatureCollection',
                              ' ↓',
                              '6 Publicar vector → ZIP público',
                              ' ↓',
                              '7 Limpiar temp']},
                   {'title': 'Etapas (demás productos)',
                    'lines': ['1 Export',
                              ' ↓',
                              '2 Mosaico',
                              ' ↓',
                              '3 Publicar mosaico',
                              ' ↓',
                              '4 Limpiar temp']},
                   {'title': 'Memoria del Catálogo',
                    'lines': ['config.py (dato crudo)',
                              ' ↓',
                              'Load Data (descubre unidades bajo demanda)',
                              ' ↓',
                              'catalog_cache.json (memoria entre sesiones)',
                              ' ↓',
                              'Botón ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (versionar)']},
                   {'title': 'Flujo de datos',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mosaico → COG',
                              ' ↓',
                              'Publicar → bucket público',
                              ' ↓',
                              '(si vectorizable) Vectorización → ZIP + upload GEE']}]},
 'EN': {'name': 'English',
        'tab_title': 'Guide: English',
        'welcome_note': 'Welcome! This English guide explains the MapBiomas Fire Export & Vectorization app: '
                        'how to navigate (country → theme → collection → product), discover the units, '
                        'process each step, and publish the maps. The tabs above show the interface; use '
                        'this guide whenever you need it.',
        'what': 'The app exports burned area/fire maps from MapBiomas Fire: from Earth Engine (GEE) to GCS, '
                'builds mosaics (COG) per unit (band or image), vectorizes when applicable, publishes to '
                'Earth Engine and the public bucket, and removes temporary files.',
        'howto_title': 'How to use',
        'steps': ['<b>Navigate</b> through the tabs: country → theme → collection → product.',
                  'Click <b>Load Data</b> (pulsing red button) to discover the <b>units</b>: bands '
                  '(multiband image) or images (ImageCollection). Discovery is on-demand — the cache is not '
                  'filled with unloaded data.',
                  'In the grid, check the desired units. The <code>Unit:</code> filter (default “All units”) '
                  'restricts by unit prefix.',
                  'Click <b>Sync</b> to check the status of the steps. The scan runs in the background, with '
                  'a progress indicator (the kernel does not block).',
                  'Execute the steps in order: <b>Export → Mosaic → Publish mosaic → Vector GCS → Vector GEE '
                  '→ Publish vector → Clean temp</b>. Steps 4–6 are only for vectorizable products (e.g., '
                  'annual_burned); for others: <b>Export → Mosaic → Publish mosaic → Clean temp</b>.',
                  'To redo an step, activate <code>FORCE_&lt;STEP&gt; = True</code> in the step cell and '
                  'select the units in the grid.',
                  'To version the Catalog memory (when there is new data in <code>config.py</code>), use the '
                  '<b>⤓ Catalog cache (.json)</b> button in the bottom bar and upload the file to GitHub.'],
        'cols_title': 'Grid columns',
        'cols': [['Export', 'unit exported from GEE (temp/)'],
                 ['Mosaic', 'assembled COG'],
                 ['Public mosaic', 'COG mirrored in the public bucket'],
                 ['Vector GCS', 'zipped vector (vectorizable products only)'],
                 ['Vector GEE', 'FeatureCollection in Earth Engine'],
                 ['Public vector', 'ZIP mirrored in the public bucket'],
                 ['Clean temp', 'temporary tiles removed after consolidation']],
        'links': '<b>🔗 OK</b> badges open the download link; <b>Vector GEE</b> copies the asset ID.',
        'legend': 'OK = step completed | MISS = step pending | N/A = not applicable',
        'graphs_title': 'Graphs',
        'graphs': [{'title': 'Navigation',
                    'lines': ['Country',
                              '└─ Theme',
                              '   └─ Collection',
                              '      └─ Product',
                              '         └─ Units (bands or images)']},
                   {'title': 'Steps (vectorizable product)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mosaic → COG',
                              ' ↓',
                              '3 Publish mosaic → public bucket',
                              ' ↓',
                              '4 Vector GCS → ZIP',
                              ' ↓',
                              '5 Vector GEE → FeatureCollection',
                              ' ↓',
                              '6 Publish vector → public ZIP',
                              ' ↓',
                              '7 Clean temp']},
                   {'title': 'Steps (other products)',
                    'lines': ['1 Export', ' ↓', '2 Mosaic', ' ↓', '3 Publish mosaic', ' ↓', '4 Clean temp']},
                   {'title': 'Catalog memory',
                    'lines': ['config.py (raw data)',
                              ' ↓',
                              'Load Data (discovers units on demand)',
                              ' ↓',
                              'catalog_cache.json (memory between sessions)',
                              ' ↓',
                              'Button ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (versioning)']},
                   {'title': 'Data flow',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mosaic → COG',
                              ' ↓',
                              'Publish → public bucket',
                              ' ↓',
                              '(if vectorizable) Vectorization → ZIP + upload GEE']}]},
 'ID': {'name': 'Bahasa Indonesia',
        'tab_title': 'Panduan: Bahasa Indonesia',
        'welcome_note': 'Selamat datang! Panduan dalam bahasa Indonesia ini menjelaskan aplikasi Export & '
                        'Vectorization dari MapBiomas Fire: cara navigasi (negara → tema → koleksi → '
                        'produk), menemukan unit, memproses setiap langkah, dan memublikasikan peta. Tab di '
                        'atas menampilkan antarmuka; gunakan panduan ini kapan pun Anda butuhkan.',
        'what': 'Aplikasi ini mengekspor peta area terbakar/kebakaran dari MapBiomas Fire: dari Earth Engine '
                '(GEE) ke GCS, membuat mosaik (COG) per unit (band atau citra), melakukan vektorisasi jika '
                'berlaku, memublikasikan ke Earth Engine dan bucket publik, serta menghapus file sementara.',
        'howto_title': 'Cara penggunaan',
        'steps': ['<b>Navigasi</b> melalui tab: negara → tema → koleksi → produk.',
                  'Klik <b>Load Data</b> (tombol merah berdenyut) untuk menemukan <b>unit</b>: band (citra '
                  'multiband) atau citra (ImageCollection). Penemuan dilakukan berdasarkan permintaan — '
                  'cache tidak diisi dengan data yang tidak dimuat.',
                  'Pada kisi, centang unit yang diinginkan. Filter <code>Unit:</code> (default “All units”) '
                  'membatasi berdasarkan awalan unit.',
                  'Klik <b>Sync</b> untuk memeriksa status langkah-langkah. Pemindaian berjalan di latar '
                  'belakang, dengan indikator kemajuan (kernel tidak terblokir).',
                  'Jalankan langkah-langkah secara berurutan: <b>Export → Mosaik → Publikasikan mosaik → '
                  'Vektor GCS → Vektor GEE → Publikasikan vektor → Bersihkan temp</b>. langkah-langkah 4–6 '
                  'hanya untuk produk yang dapat divektorisasi (misalnya, annual_burned); untuk yang lain: '
                  '<b>Export → Mosaik → Publikasikan mosaik → Bersihkan temp</b>.',
                  'Untuk mengulang langkah, aktifkan <code>FORCE_&lt;LANGKAH&gt; = True</code> di sel '
                  'langkah dan pilih unit di kisi.',
                  'Untuk membuat versi memori Katalog (saat ada data baru di <code>config.py</code>), '
                  'gunakan tombol <b>⤓ Catalog cache (.json)</b> di bilah bawah dan unggah file ke GitHub.'],
        'cols_title': 'Kolom kisi',
        'cols': [['Export', 'unit diekspor dari GEE (temp/)'],
                 ['Mosaic', 'COG yang digabungkan'],
                 ['Public mosaic', 'COG dicerminkan di bucket publik'],
                 ['Vector GCS', 'vektor dalam format ZIP (hanya produk yang dapat divektorisasi)'],
                 ['Vector GEE', 'FeatureCollection di Earth Engine'],
                 ['Public vector', 'ZIP dicerminkan di bucket publik'],
                 ['Clean temp', 'tile sementara dihapus setelah konsolidasi']],
        'links': 'Lencana <b>🔗 OK</b> membuka tautan unduhan; <b>Vector GEE</b> menyalin asset ID.',
        'legend': 'OK = langkah selesai | MISS = langkah tertunda | N/A = tidak berlaku',
        'graphs_title': 'Grafik',
        'graphs': [{'title': 'Navigasi',
                    'lines': ['Negara',
                              '└─ Tema',
                              '   └─ Koleksi',
                              '      └─ Produk',
                              '         └─ Unit (band atau citra)']},
                   {'title': 'Langkah-langkah (produk dapat divektorisasi)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mosaik → COG',
                              ' ↓',
                              '3 Publikasikan mosaik → bucket publik',
                              ' ↓',
                              '4 Vektor GCS → ZIP',
                              ' ↓',
                              '5 Vektor GEE → FeatureCollection',
                              ' ↓',
                              '6 Publikasikan vektor → ZIP publik',
                              ' ↓',
                              '7 Bersihkan temp']},
                   {'title': 'Langkah-langkah (produk lainnya)',
                    'lines': ['1 Export',
                              ' ↓',
                              '2 Mosaik',
                              ' ↓',
                              '3 Publikasikan mosaik',
                              ' ↓',
                              '4 Bersihkan temp']},
                   {'title': 'Memori Katalog',
                    'lines': ['config.py (data mentah)',
                              ' ↓',
                              'Load Data (menemukan unit berdasarkan permintaan)',
                              ' ↓',
                              'catalog_cache.json (memori antar sesi)',
                              ' ↓',
                              'Tombol ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (pembuatan versi)']},
                   {'title': 'Alur data',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mosaik → COG',
                              ' ↓',
                              'Publikasikan → bucket publik',
                              ' ↓',
                              '(jika dapat divektorisasi) Vektorisasi → ZIP + upload GEE']}]},
 'FR': {'name': 'Français',
        'tab_title': 'Guide : Français',
        'welcome_note': "Bienvenue ! Ce guide en français explique l'application Export & Vectorization de "
                        'MapBiomas Fire : comment naviguer (pays → thème → collection → produit), découvrir '
                        'les unités, traiter chaque étape et publier les cartes. Les onglets ci-dessus '
                        "affichent l'interface ; utilisez ce guide chaque fois que nécessaire.",
        'what': "L'application exporte les cartes des zones brûlées/incendies de MapBiomas Fire : de Earth "
                'Engine (GEE) vers GCS, assemble des mosaïques (COG) par unité (bande ou image), vectorise '
                'le cas échéant, publie sur Earth Engine et le bucket public, et supprime les fichiers '
                'temporaires.',
        'howto_title': 'Comment utiliser',
        'steps': ['<b>Naviguez</b> à travers les onglets : pays → thème → collection → produit.',
                  'Cliquez sur <b>Load Data</b> (bouton rouge clignotant) pour découvrir les <b>unités</b> : '
                  'bandes (image multibande) ou images (ImageCollection). La découverte est à la demande — '
                  "le cache n'est pas rempli de données non chargées.",
                  'Dans la grille, cochez les unités souhaitées. Le filtre <code>Unit:</code> (par défaut « '
                  "All units ») restreint par préfixe d'unité.",
                  "Cliquez sur <b>Sync</b> pour vérifier le statut des étapes. L'analyse s'exécute en "
                  'arrière-plan avec un indicateur de progression (le noyau ne se bloque pas).',
                  "Exécutez les étapes dans l'ordre : <b>Export → Mosaïque → Publier mosaïque → Vecteur GCS "
                  '→ Vecteur GEE → Publier vecteur → Nettoyer temp</b>. étapes 4–6 uniquement pour les '
                  'produits vectorisables (ex. : annual_burned) ; pour les autres : <b>Export → Mosaïque → '
                  'Publier mosaïque → Nettoyer temp</b>.',
                  'Pour refaire une étape, activez <code>FORCE_&lt;ÉTAPE&gt; = True</code> dans la cellule '
                  "de l'étape et sélectionnez les unités dans la grille.",
                  "Pour versionner la mémoire du catalogue (lorsqu'il y a de nouvelles données dans "
                  '<code>config.py</code>), utilisez le bouton <b>⤓ Catalog cache (.json)</b> dans la barre '
                  'inférieure et téléchargez le fichier sur GitHub.'],
        'cols_title': 'Colonnes de la grille',
        'cols': [['Export', 'unité exportée depuis GEE (temp/)'],
                 ['Mosaic', 'COG assemblé'],
                 ['Public mosaic', 'COG mis en miroir dans le bucket public'],
                 ['Vector GCS', 'vecteur zippé (uniquement produits vectorisables)'],
                 ['Vector GEE', 'FeatureCollection dans Earth Engine'],
                 ['Public vector', 'ZIP mis en miroir dans le bucket public'],
                 ['Clean temp', 'tuiles temporaires supprimées après consolidation']],
        'links': "Les badges <b>🔗 OK</b> ouvrent le lien de téléchargement ; <b>Vector GEE</b> copie l'asset "
                 'ID.',
        'legend': 'OK = étape terminée | MISS = étape en attente | N/A = non applicable',
        'graphs_title': 'Graphiques',
        'graphs': [{'title': 'Navigation',
                    'lines': ['Pays',
                              '└─ Thème',
                              '   └─ Collection',
                              '      └─ Produit',
                              '         └─ Unités (bandes ou images)']},
                   {'title': 'Étapes (produit vectorisable)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mosaïque → COG',
                              ' ↓',
                              '3 Publier mosaïque → bucket public',
                              ' ↓',
                              '4 Vecteur GCS → ZIP',
                              ' ↓',
                              '5 Vecteur GEE → FeatureCollection',
                              ' ↓',
                              '6 Publier vecteur → ZIP public',
                              ' ↓',
                              '7 Nettoyer temp']},
                   {'title': 'Étapes (autres produits)',
                    'lines': ['1 Export',
                              ' ↓',
                              '2 Mosaïque',
                              ' ↓',
                              '3 Publier mosaïque',
                              ' ↓',
                              '4 Nettoyer temp']},
                   {'title': 'Mémoire du catalogue',
                    'lines': ['config.py (données brutes)',
                              ' ↓',
                              'Load Data (découvre les unités à la demande)',
                              ' ↓',
                              'catalog_cache.json (mémoire entre sessions)',
                              ' ↓',
                              'Bouton ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (versionnage)']},
                   {'title': 'Flux de données',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mosaïque → COG',
                              ' ↓',
                              'Publier → bucket public',
                              ' ↓',
                              '(si vectorisable) Vectorisation → ZIP + upload GEE']}]},
 'NL': {'name': 'Nederlands',
        'tab_title': 'Gids: Nederlands',
        'welcome_note': 'Welkom! Deze Nederlandstalige gids legt de Export & Vectorization app van MapBiomas '
                        'Fire uit: hoe te navigeren (land → thema → collectie → product), de eenheden te '
                        'ontdekken, elke stap te verwerken en de kaarten te publiceren. De tabbladen '
                        'hierboven tonen de interface; gebruik deze gids wanneer je hem nodig hebt.',
        'what': 'De app exporteert kaarten van verbrande gebieden/branden van MapBiomas Fire: van Earth '
                'Engine (GEE) naar GCS, bouwt mozaïeken (COG) per eenheid (band of afbeelding), vectoriseert '
                'indien van toepassing, publiceert naar Earth Engine en de openbare bucket, en verwijdert '
                'tijdelijke bestanden.',
        'howto_title': 'Hoe te gebruiken',
        'steps': ['<b>Navigeer</b> door de tabbladen: land → thema → collectie → product.',
                  'Klik op <b>Load Data</b> (kloppende rode knop) om de <b>eenheden</b> te ontdekken: banden '
                  '(multiband afbeelding) of afbeeldingen (ImageCollection). Ontdekking is on-demand — de '
                  'cache wordt niet gevuld met ongeladen gegevens.',
                  'Vink in het raster de gewenste eenheden aan. Het filter <code>Unit:</code> (standaard '
                  '“All units”) beperkt op voorvoegsel van de eenheid.',
                  'Klik op <b>Sync</b> om de status van de stappen te controleren. De scan draait op de '
                  'achtergrond, met een voortgangsindicator (de kernel blokkeert niet).',
                  'Voer de stappen op volgorde uit: <b>Export → Mozaïek → Publiceer mozaïek → Vector GCS → '
                  'Vector GEE → Publiceer vector → Wis temp</b>. stappen 4–6 alleen voor vectoriseerbare '
                  'producten (bijv. annual_burned); voor de rest: <b>Export → Mozaïek → Publiceer mozaïek → '
                  'Wis temp</b>.',
                  'Om een stap opnieuw te doen, activeer je <code>FORCE_&lt;STAP&gt; = True</code> in de cel '
                  'van de stap en selecteer je de eenheden in het raster.',
                  'Om de geheugen van de catalogus te versioneren (wanneer er nieuwe gegevens in '
                  '<code>config.py</code> zijn), gebruik je de knop <b>⤓ Catalog cache (.json)</b> in de '
                  'onderste balk en upload je het bestand naar GitHub.'],
        'cols_title': 'Rasterkolommen',
        'cols': [['Export', 'eenheid geëxporteerd van GEE (temp/)'],
                 ['Mosaic', 'geassembleerde COG'],
                 ['Public mosaic', 'COG gespiegeld in de openbare bucket'],
                 ['Vector GCS', 'gezipte vector (alleen vectoriseerbare producten)'],
                 ['Vector GEE', 'FeatureCollection in Earth Engine'],
                 ['Public vector', 'ZIP gespiegeld in de openbare bucket'],
                 ['Clean temp', 'tijdelijke tegels verwijderd na consolidatie']],
        'links': 'Badges <b>🔗 OK</b> openen de downloadlink; <b>Vector GEE</b> kopieert de asset ID.',
        'legend': 'OK = stap voltooid | MISS = stap in afwachting | N/A = niet van toepassing',
        'graphs_title': 'Grafieken',
        'graphs': [{'title': 'Navigatie',
                    'lines': ['Land',
                              '└─ Thema',
                              '   └─ Collectie',
                              '      └─ Product',
                              '         └─ Eenheden (banden of afbeeldingen)']},
                   {'title': 'Stappen (vectoriseerbaar product)',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 Mozaïek → COG',
                              ' ↓',
                              '3 Publiceer mozaïek → openbare bucket',
                              ' ↓',
                              '4 Vector GCS → ZIP',
                              ' ↓',
                              '5 Vector GEE → FeatureCollection',
                              ' ↓',
                              '6 Publiceer vector → openbare ZIP',
                              ' ↓',
                              '7 Wis temp']},
                   {'title': 'Stappen (overige producten)',
                    'lines': ['1 Export',
                              ' ↓',
                              '2 Mozaïek',
                              ' ↓',
                              '3 Publiceer mozaïek',
                              ' ↓',
                              '4 Wis temp']},
                   {'title': 'Geheugen van de catalogus',
                    'lines': ['config.py (ruwe data)',
                              ' ↓',
                              'Load Data (ontdekt eenheden on-demand)',
                              ' ↓',
                              'catalog_cache.json (geheugen tussen sessies)',
                              ' ↓',
                              'Knop ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub (versioneren)']},
                   {'title': 'Dataflow',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              'Mozaïek → COG',
                              ' ↓',
                              'Publiceer → openbare bucket',
                              ' ↓',
                              '(indien vectoriseerbaar) Vectorisatie → ZIP + upload GEE']}]},
 'ZH': {'name': '中文',
        'tab_title': '指南：中文',
        'welcome_note': '欢迎！本中文指南说明了 MapBiomas Fire 的 Export & Vectorization 应用程序：如何导航（国家 → 主题 → 集合 → 产品），发现 '
                        '单位，处理每个 步骤，并发布地图。上面的选项卡显示了界面；随时使用此指南。',
        'what': '该应用程序从 MapBiomas Fire 导出烧毁面积/火灾地图：从 Earth Engine (GEE) 到 GCS，按 单位（波段或图像）构建镶嵌图 '
                '(COG)，在适用时进行矢量化，发布到 Earth Engine 和公共存储桶，并删除临时文件。',
        'howto_title': '如何使用',
        'steps': ['通过选项卡<b>导航</b>：国家 → 主题 → 集合 → 产品。',
                  '单击 <b>Load Data</b>（闪烁的红色按钮）以发现 <b>单位</b>：波段（多波段图像）或图像（ImageCollection）。发现是按需进行的 — '
                  '缓存不会被未加载的数据填满。',
                  '在网格中，勾选所需的 单位。<code>Unit:</code> 过滤器（默认为“All units”）按 单位 前缀进行限制。',
                  '单击 <b>Sync</b> 检查 步骤 的状态。扫描在后台运行，带有进度指示器（内核不会阻塞）。',
                  '按顺序执行 步骤：<b>Export → 镶嵌 → 发布镶嵌图 → 矢量 GCS → 矢量 GEE → 发布矢量 → 清理临时文件</b>。步骤 4–6 '
                  '仅适用于可矢量化的产品（例如 annual_burned）；对于其他产品：<b>Export → 镶嵌 → 发布镶嵌图 → 清理临时文件</b>。',
                  '要重做 步骤，请在 步骤 单元格中激活 <code>FORCE_&lt;步骤&gt; = True</code> 并在网格中选择 单位。',
                  '要对 目录 内存进行版本控制（当 <code>config.py</code> 中有新数据时），请使用底部栏中的 <b>⤓ Catalog cache (.json)</b> '
                  '按钮并将文件上传到 GitHub。'],
        'cols_title': '网格列',
        'cols': [['Export', '从 GEE 导出的 单位 (temp/)'],
                 ['Mosaic', '组装的 COG'],
                 ['Public mosaic', '镜像在公共存储桶中的 COG'],
                 ['Vector GCS', '压缩的矢量（仅限可矢量化产品）'],
                 ['Vector GEE', 'Earth Engine 中的 FeatureCollection'],
                 ['Public vector', '镜像在公共存储桶中的 ZIP'],
                 ['Clean temp', '合并后删除的临时切片']],
        'links': '带有 <b>🔗 OK</b> 的徽章可打开下载链接；<b>Vector GEE</b> 复制 asset ID。',
        'legend': 'OK = 步骤 已完成 | MISS = 步骤 待处理 | N/A = 不适用',
        'graphs_title': '图表',
        'graphs': [{'title': '导航：',
                    'lines': ['国家', '└─ 主题', '   └─ 集合', '      └─ 产品', '         └─ 单位（波段或图像）']},
                   {'title': '步骤（可矢量化产品）：',
                    'lines': ['1 Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '2 镶嵌 → COG',
                              ' ↓',
                              '3 发布镶嵌图 → 公共存储桶',
                              ' ↓',
                              '4 矢量 GCS → ZIP',
                              ' ↓',
                              '5 矢量 GEE → FeatureCollection',
                              ' ↓',
                              '6 发布矢量 → 公共 ZIP',
                              ' ↓',
                              '7 清理临时文件']},
                   {'title': '步骤（其他产品）：',
                    'lines': ['1 Export', ' ↓', '2 镶嵌', ' ↓', '3 发布镶嵌图', ' ↓', '4 清理临时文件']},
                   {'title': '目录 内存：',
                    'lines': ['config.py（原始数据）',
                              ' ↓',
                              'Load Data（按需发现 单位）',
                              ' ↓',
                              'catalog_cache.json（会话之间的内存）',
                              ' ↓',
                              '按钮 ⤓ Catalog cache (.json)',
                              ' ↓',
                              'GitHub（版本控制）']},
                   {'title': '数据流：',
                    'lines': ['GEE ImageCollection',
                              ' ↓',
                              'Export → tiles 0/1 (temp/)',
                              ' ↓',
                              '镶嵌 → COG',
                              ' ↓',
                              '发布 → 公共存储桶',
                              ' ↓',
                              '（如果可矢量化）矢量化 → ZIP + upload GEE']}]}}


def _guide_graphs(g, p):
    """Grafos simples da guia: diagramas de texto em <pre>, fundo claro e letra
    escura. Cada idioma pode ter sua propria lista `graphs` (opcional)."""
    graphs = g.get("graphs", [])
    if not graphs:
        return ""
    heading = ""
    gtitle = g.get("graphs_title", "")
    if gtitle:
        heading = f'<h4 style="margin:12px 0 4px 0;">{gtitle}</h4>'
    blocks = []
    for graph in graphs:
        title = graph.get("title", "")
        lines = graph.get("lines", [])
        text = "\n".join(lines)
        blocks.append(
            f'<div style="margin:8px 0 6px 0;">'
            f'<div style="font-weight:700;color:{p["guide_fg"]};font-size:12px;'
            f'margin-bottom:3px;">{title}</div>'
            f'<pre style="margin:0;padding:8px 10px;background:#ffffff;'
            f'border:1px solid #dddddd;border-radius:4px;font-size:12px;'
            f'color:#212529;line-height:1.5;overflow-x:auto;">{text}</pre>'
            f'</div>'
        )
    return heading + "".join(blocks)


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
    graphs_html = _guide_graphs(g, p)
    return (
        f'<div style="font-size:12px;color:{p["guide_fg"]};line-height:1.6;">'
        f'{welcome_html}'
        f'<p><b>{g["name"]}</b> — {g["what"]}</p>'
        f'<h4 style="margin:10px 0 4px 0;">{g["howto_title"]}</h4>'
        f'<ol style="margin:0 0 8px 0;padding-left:20px;">{steps}</ol>'
        f'{graphs_html}'
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
def _discover_units(country, theme, collection, product, logger=None):
    """Descobre bandas/imagens de um produto (Sync).

    So consulta o GEE quando o produto ainda nao esta no catalogo (memo/disco)
    — discovery sob demanda para nao encher o cache sem necessidade.
    """
    try:
        return catalog.inventory_units(country, theme, collection, product,
                                       logger=logger, discover=True)
    except Exception:
        return []


def _cached_units(country, theme, collection, product):
    """Units ja conhecidas na memoria (memo/disco) — sem nenhuma chamada de rede.

    Usado pelo caminho rapido do Load Data, que nunca trava.
    """
    try:
        return catalog.inventory_units(country, theme, collection, product,
                                       logger=None, discover=False)
    except Exception:
        return []


class UnitGridPanel:
    _DATE_W = "230px"
    _CELL_W = "88px"
    _SEL_W  = "72px"
    _GRID_RENDER_CAP = 60   # acima disso, o filtro Unit: inicia no prefixo recente

    def __init__(self, country, theme, collection, log_area=None, on_data_loaded_change=None,
                 on_clear_all=None, on_load_collection=None, on_select_all_collection=None):
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
        self._on_clear_all_cb = on_clear_all
        self._on_load_collection_cb = on_load_collection
        self._on_select_all_collection_cb = on_select_all_collection

        self.grid_container = widgets.VBox([
            _loading_html("Loading units...")
        ])

        self.btn_load = widgets.Button(description="Load Data", button_style="danger",
                                       icon="download", layout=L(width="120px", height="34px"),
                                       tooltip="Load this product: cached units, discover new bands/units and scan status (bounded by SCAN_TIMEOUT)")
        self.btn_load.on_click(self._on_sync)
        self.btn_load_collection = widgets.Button(description="Load Collection", button_style="danger",
                                                  icon="layers", layout=L(width="150px", height="34px"),
                                                  tooltip="Load all products of this collection (sequential queue); green when all are loaded")
        self.btn_load_collection.on_click(self._on_load_collection)
        self.btn_select_pending = widgets.Button(description="Select Pending", button_style="info",
                                                 layout=L(width="150px", height="34px"),
                                                 tooltip="Each click selects the units pending at the next stage (cycle)")
        self.btn_select_pending.on_click(self._on_select_pending)
        self.btn_select_all = widgets.Button(description="Select All", button_style="info",
                                             layout=L(width="120px", height="34px"))
        self.btn_select_all.on_click(self._on_select_all)
        self.btn_select_all_collection = widgets.Button(description="Select All Collection", button_style="info",
                                                        icon="list-check", layout=L(width="170px", height="34px"),
                                                        tooltip="Select all rendered units in every product of this collection")
        self.btn_select_all_collection.on_click(self._on_select_all_collection)
        self.btn_clear = widgets.Button(description="Clear", button_style="warning",
                                        layout=L(width="90px", height="34px"))
        self.btn_clear.on_click(self._on_clear)
        self.btn_clear_all = widgets.Button(description="Clear All", button_style="warning",
                                            icon="trash", layout=L(width="100px", height="34px"),
                                            tooltip="Clear selections in ALL products")
        self.btn_clear_all.on_click(self._on_clear_all)

        self.year_dropdown = widgets.Dropdown(options=["All units"], value="All units",
                                              description="Unit:", layout=L(width="200px"))
        self.year_dropdown.observe(self._on_year_change, names="value")

        self._filter_explicit = False   # usuario escolheu o filtro manualmente
        self._rendering = False         # guarda de reentrancia no render
        self._pending_cycle = []        # ciclo de estagios do Select Pending
        self._pending_cycle_idx = 0
        self.progress_loader = ProgressLoader("Click Load Data / Load Collection to load.")
        spacer = widgets.HBox([], layout=L(flex="1 1 20px", min_width="0px"))
        self.toolbar = widgets.HBox([
            self.btn_load, self.btn_load_collection, self.btn_select_pending,
            self.btn_select_all, self.btn_select_all_collection,
            self.btn_clear, self.btn_clear_all, self.year_dropdown,
            spacer, self.progress_loader.widget,
        ], layout=L(margin="0 0 8px 0", gap="8px", align_items="center", flex_wrap="wrap"))

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
        # Units (bandas/imagens) sao descobertas apenas no Load Data / Load Collection —
        # nenhuma chamada GEE/GCS aqui na abertura da UI.
        self.units = []
        # Keep existing state if available, don't auto-sync
        if not hasattr(self, 'state') or not self.state:
            self.state = {"updated_at": None}
        self._data_loaded = False
        self._filter_explicit = False
        self._pending_cycle = []
        self._pending_cycle_idx = 0
        self.btn_select_pending.description = "Select Pending"
        self._update_load_button_style()
        # Notify parent ProductTabs to update tab style
        self._notify_tab_style()

    def _update_load_button_style(self):
        """Estado do botao Load Data conforme o carregamento do produto."""
        if self._data_loaded:
            self.btn_load.button_style = "success"
            self.btn_load.description = "Load Data"
            self.btn_load.icon = "check"
            # Remove pulsing outline
            if hasattr(self.btn_load, 'remove_class'):
                self.btn_load.remove_class('mfm-btn-unloaded')
        else:
            self.btn_load.button_style = "danger"
            self.btn_load.description = "Load Data"
            self.btn_load.icon = "download"
            # Add pulsing outline
            if hasattr(self.btn_load, 'add_class'):
                self.btn_load.add_class('mfm-btn-unloaded')

    def _update_load_collection_style(self, all_loaded):
        """Load Collection segue o padrao do Load Data: vermelho (base) ate
        todos os produtos da colecao estarem carregados, verde quando sim."""
        if all_loaded:
            self.btn_load_collection.button_style = "success"
            self.btn_load_collection.icon = "check"
        else:
            self.btn_load_collection.button_style = "danger"
            self.btn_load_collection.icon = "layers"

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
        self.year_dropdown.options = options
        # Acima do cap, inicia o filtro no prefixo mais recente para o render
        # ficar leve (mantendo "All units" disponivel no dropdown).
        if len(self._all_units()) > self._GRID_RENDER_CAP and not self._filter_explicit:
            recent = options[1] if len(options) > 1 else "All units"
            self.year_dropdown.value = recent
            self.year_filter = int(recent) if recent != "All units" else None
        elif self.year_dropdown.value not in options:
            self.year_dropdown.value = "All units"
            self.year_filter = None

    def _on_year_change(self, change):
        value = change.get("new")
        self.year_filter = int(value) if value != "All units" else None
        if getattr(self, "_rendering", False):
            return   # mudanca programatica (auto-cap durante render)
        self._filter_explicit = True   # usuario escolheu; nao sobrescrever depois
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
        if self._rendering:
            return
        self._rendering = True
        try:
            self._do_render_grid()
        finally:
            self._rendering = False

    def _do_render_grid(self):
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
        cols = _product_cols()
        sel_header = widgets.HTML(
            f'<div style="width:{self._SEL_W};height:42px;background:{p["grid_header_bg"]};'
            f'text-align:center;font-weight:700;font-size:11px;color:{p["grid_header_fg"]};padding:12px 3px;'
            f'box-sizing:border-box;border-left:1px solid {p["sep"]};'
            f'border-right:1px solid {p["sep"]};">Select</div>')
        unit_header = widgets.HTML(
            f'<div style="width:{self._DATE_W};height:42px;background:{p["grid_header_bg"]};'
            f'font-weight:700;font-size:12px;color:{p["grid_header_fg"]};padding:12px 6px;'
            f'box-sizing:border-box;border-right:1px solid {p["sep"]};">Unit</div>')
        header_row = widgets.HBox(
            [sel_header, unit_header]
            + [_header_cell(self._CELL_W, t, e, kind in _VECTOR_KINDS) for _, t, e, kind in cols],
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
            for key, _t, _e, kind in cols:
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
            row = widgets.HBox([chk_wrapper] + cells, layout=row_layout)
            row.layout.background = bg
            rows.append(row)

        n_all = len(self._all_units())
        n_visible = len(units)
        n_complete = sum(1 for u in self._all_units() if _is_complete(self.state.get(u, {})))

        if self.year_filter is not None:
            cap_note = ('' if self._filter_explicit else
                        ' <span style="color:#e67e22;">(default — &gt;60 units; use Unit: filter for all)</span>')
            label = (f'{n_visible} units with prefix {self.year_filter} in filter{cap_note} &nbsp;|&nbsp; '
                     f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> complete')
        else:
            label = (f'{n_all} units &nbsp;|&nbsp; '
                     f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> complete')

        legend = widgets.HTML(
            f'<div style="font-size:11px;color:{p["legend_fg"]};margin:6px 0 0 10px;padding:6px 10px;'
            f'background:{p["legend_bg"]};border-radius:4px;">{label}</div>'
        )
        if vectorizable:
            step_hint = ('Export=Step 1 &nbsp;|&nbsp; Mosaic=Step 2 &nbsp;|&nbsp; '
                         'Public mosaic=Step 3 &nbsp;|&nbsp; Clean temp=Step 4 &nbsp;|&nbsp; '
                         'Vector GCS=Step 5 (optional) &nbsp;|&nbsp; Vector GEE=Step 6 (optional) '
                         '&nbsp;|&nbsp; Public vector=Step 7 (optional)')
        else:
            step_hint = ('Export=Step 1 &nbsp;|&nbsp; Mosaic=Step 2 &nbsp;|&nbsp; '
                         'Public mosaic=Step 3 &nbsp;|&nbsp; Clean temp=Step 4 (last)')
        hint = widgets.HTML(
            f'<div style="font-size:11px;color:{p["hint_fg"]};margin:4px 0 0 10px;padding:6px 10px;'
            f'background:{p["hint_bg"]};border:1px solid {p["hint_border"]};border-radius:4px;line-height:1.5;">'
            f'<strong>MISS &rarr; OK:</strong> {step_hint}'
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

    def _on_sync(self, _, label=None):
        """Sync: descobre units + varre GCS/GEE em uma thread de fundo e espera
        na thread principal com timeout (SCAN_TIMEOUT). Render sempre na main
        thread — nunca fica 'carregando' para sempre."""
        if self.is_refreshing:
            return
        if not self.product:
            return
        config.set_country(self.country, verbose=False)
        config.set_theme(self.theme)
        config.set_collection(self.collection)
        config.set_product(self.product)
        self.is_refreshing = True
        set_button_busy(self.btn_load, True, "Loading...")
        self.progress_loader.start(label or "Loading data...")
        self._log("Checking files in GCS and assets in GEE...", "info")

        # Render provisorio (main thread) com units da memoria + estado persistido.
        if not self.units:
            self.units = _cached_units(self.country, self.theme, self.collection, self.product)
        try:
            persisted = load_state(self.country, self.theme, self.collection, self.product)
            if persisted and len(persisted) > 1:
                self.state = persisted
            self._render_grid()
            self._restore_selected(self._get_selected_keys())
        except Exception as e:
            self._log(f"Pre-render failed: {e}", "warning")

        # Scan completo (discovery de bandas se faltar + varredura GCS/GEE) em
        # thread de fundo, com timeout — nao trava para sempre.
        def _sync_work():
            units = self.units
            if not units:
                units = _discover_units(self.country, self.theme, self.collection,
                                        self.product, logger=self._log)
            fresh = build_state(country=self.country, theme=self.theme,
                                collection=self.collection, product=self.product,
                                logger=self._log, on_stage=self._on_stage)
            return units, fresh

        timeout = getattr(config, "SCAN_TIMEOUT", 180)
        units = None
        fresh = None
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            fut = ex.submit(_sync_work)
            units, fresh = fut.result(timeout=timeout)
        except TimeoutError:
            self._log(f"[WARN] Sync timed out after {timeout}s — showing current "
                      "status. The scan may still finish and update the grid.", "warning")
        except Exception as e:
            self._log(f"Sync error: {e}", "error")
        finally:
            ex.shutdown(wait=False)

        if units is not None:
            self.units = units
        if fresh is not None:
            self.state = fresh
            if not self.units:
                self.units = [u for u in fresh if u != "updated_at"]
        self._render_grid()
        self._restore_selected(self._get_selected_keys())
        n_ok = sum(1 for u in self._all_units() if _is_complete(self.state.get(u, {})))
        if fresh is not None:
            msg = f"Sync complete: {n_ok}/{len(self._all_units())} units complete."
            self.progress_loader.stop(msg)
            self._log(msg, "success")
        else:
            msg = "Sync incomplete — showing current status."
            self.progress_loader.stop("Timed out — showing current status.")
            self._log(msg, "warning")
        self._data_loaded = True
        self._update_load_button_style()
        self._notify_tab_style()
        # estado mudou: reinicia o ciclo do Select Pending
        self._pending_cycle = []
        self._pending_cycle_idx = 0
        self.btn_select_pending.description = "Select Pending"
        self._finish_sync()

    def _on_stage(self, stage):
        self.progress_loader.set_status(stage)

    def _finish_sync(self):
        self.is_refreshing = False
        set_button_busy(self.btn_load, False)
        self._update_load_button_style()

    def _on_select_pending(self, _):
        """Select Pending dinamico: a cada clique seleciona (e substitui) apenas
        as unidades pendentes no proximo estagio do ciclo (decrescente — mais
        avancado primeiro). So sugere estagios cujas etapas anteriores ja foram
        processadas (primeira etapa incompleta)."""
        present = []
        for key, _t, _n, _b in _product_cols():
            if any(_unit_pending_key(self.state.get(u, {})) == key
                   for u in self._all_units()):
                present.append(key)
        if not present:
            self._log("Select Pending: no pending units (all complete or empty).", "info")
            self._pending_cycle = []
            self.btn_select_pending.description = "Select Pending"
            return
        present_desc = list(reversed(present))
        if set(self._pending_cycle) != set(present_desc):
            self._pending_cycle = present_desc
            self._pending_cycle_idx = 0
        target = self._pending_cycle[self._pending_cycle_idx % len(self._pending_cycle)]
        self._pending_cycle_idx = (self._pending_cycle_idx + 1) % len(self._pending_cycle)
        # Substitui a selecao: so as unidades pendentes no estagio-alvo.
        for u, chk in self.chk_dict.items():
            chk.value = _unit_pending_key(self.state.get(u, {})) == target
        self.btn_select_pending.description = f"Select Pending \u00b7 {_step_title(target)}"
        self._log(f"Select Pending \u00b7 {_step_title(target)}: selected pending units "
                  "at this stage (individual checks still editable).", "info")

    def _on_load_collection(self, _=None):
        """Load Collection: delega ao ProductTabs (fila simples dos produtos)."""
        if self._on_load_collection_cb:
            self._on_load_collection_cb()
        else:
            self._log("Load Collection not available in this context.", "warning")

    def _on_select_all_collection(self, _=None):
        """Select All Collection: delega ao ProductTabs (todos os produtos)."""
        if self._on_select_all_collection_cb:
            self._on_select_all_collection_cb()
        else:
            self._on_select_all(None)

    def _on_select_all(self, _):
        for chk in self.chk_dict.values():
            chk.value = True

    def _on_clear(self, _):
        for chk in self.chk_dict.values():
            chk.value = False

    def _on_clear_all(self, _):
        """Limpa as selecoes deste painel e de TODOS os outros paineis de produto."""
        if self._on_clear_all_cb:
            self._on_clear_all_cb()
        else:
            self._on_clear(None)

    def get_selected_units(self):
        return [k for k, chk in self.chk_dict.items() if chk.value]

    def get_selected_items(self):
        """Selecao com contexto completo para pipelines multi-painel."""
        return [
            {
                "country": self.country,
                "theme": self.theme,
                "collection": self.collection,
                "product": self.product,
                "unit": k,
            }
            for k, chk in self.chk_dict.items() if chk.value
        ]

    def context_key(self):
        return (self.country, self.theme, self.collection, self.product)

    def matches_context(self, item=None, country=None, theme=None, collection=None, product=None):
        if item is not None:
            country = item.get("country")
            theme = item.get("theme")
            collection = item.get("collection")
            product = item.get("product")
        return (country, theme, collection, product) == self.context_key()

    def sync_context(self):
        """Sincroniza os seletores globais do config com o contexto deste painel."""
        if not self.product:
            return
        config.set_country(self.country, verbose=False)
        config.set_theme(self.theme)
        config.set_collection(self.collection)
        config.set_product(self.product)

    def _get_selected_keys(self):
        return [k for k, chk in self.chk_dict.items() if chk.value]

    def _restore_selected(self, keys):
        for k in keys:
            if k in self.chk_dict:
                self.chk_dict[k].value = True

    def sync(self):
        self._on_sync(None)


def _iter_unit_panels(obj):
    """Percorre a arvore de abas (pais -> tema -> colecao -> produto) e
    rende todos os UnitGridPanel criados, de todos os paises/produtos."""
    panels = getattr(obj, "__dict__", {}).get("_panels")
    if isinstance(panels, dict) and panels:
        for child in panels.values():
            yield from _iter_unit_panels(child)
        return
    if isinstance(obj, UnitGridPanel):
        yield obj


# ---------------------------------------------------------------------------
# Navegacao: pais -> tema -> colecao -> produto (abas) -> grid de unidades
# ---------------------------------------------------------------------------
class ProductTabs:
    """Abas de produtos visiveis (barra com quebra de linha) + um grid
    independente por produto. A cor de fundo da guia indica o estado de
    carregado (verde) / nao carregado (cinza) do produto."""

    def __init__(self, country, theme, collection, log_area=None, on_data_loaded_change=None,
                 on_clear_all=None):
        self.country = country
        self.theme = theme
        self.collection = collection
        self.log_area = log_area
        self._on_data_loaded_change = on_data_loaded_change
        self._on_clear_all_cb = on_clear_all
        products = config.list_products(country, theme, collection)
        self.products = [p["product"] for p in products if p.get("visible", True)]
        self._panels = {}
        self._active_panel = None
        self._active_product_name = None
        self._loaded_products = set()
        self._loading_collection = False
        self._bar, self._tab_btns = _wrap_tab_bar(self.products, self._activate_product)
        self._panels_box = widgets.VBox([])
        self.container = widgets.VBox([_STATUS_CSS, self._bar, self._panels_box])
        if self.products:
            self._activate_product(0)

    def _activate_product(self, index):
        if index < 0 or index >= len(self.products):
            return
        product = self.products[index]
        self._active_product_name = product
        if product not in self._panels:
            panel = UnitGridPanel(
                self.country, self.theme, self.collection, self.log_area,
                on_data_loaded_change=self._on_panel_data_loaded,
                on_clear_all=self._on_clear_all_cb,
                on_load_collection=self._on_load_collection,
                on_select_all_collection=self._on_select_all_collection,
            )
            panel._activate_product(product)
            self._panels[product] = panel
        else:
            # Painel em cache: ressincroniza os seletores globais do config
            # para este produto (evita export com asset/banda de outro painel).
            panel = self._panels[product]
            panel.sync_context()
        self._active_panel = panel
        self._panels_box.children = [panel.container]
        # Repinta todas as guias (ativo + carregado) e o Load Collection.
        for p in self.products:
            self._update_tab_style(p)
        self._sync_load_collection_styles()

    def _all_loaded(self):
        return bool(self.products) and all(p in self._loaded_products for p in self.products)

    def _sync_load_collection_styles(self):
        """Load Collection verde quando TODOS os produtos da colecao carregaram."""
        all_loaded = self._all_loaded()
        for panel in self._panels.values():
            panel._update_load_collection_style(all_loaded)

    def sync_context(self):
        """Propaga a sincronizacao de contexto ate o painel ativo."""
        panel = self.__dict__.get("_active_panel")
        if panel is not None:
            panel.sync_context()

    def _on_panel_data_loaded(self, product, loaded):
        """Callback when a panel's data loaded state changes."""
        if loaded:
            self._loaded_products.add(product)
        self._update_tab_style(product)
        self._sync_load_collection_styles()

    def _update_tab_style(self, product):
        """Cor de fundo da guia = estado de carregado; borda = ativo."""
        if product not in self.products:
            return
        idx = self.products.index(product)
        if idx < 0 or idx >= len(self._tab_btns):
            return
        btn = self._tab_btns[idx]
        loaded = product in self._loaded_products
        btn.style.button_color = "#d4edda" if loaded else "#f8f9fa"   # verde/cinza
        active = (product == self._active_product_name)
        if hasattr(btn, "add_class") and hasattr(btn, "remove_class"):
            if active:
                btn.add_class("mfm-tab-active")
            else:
                btn.remove_class("mfm-tab-active")

    def _on_load_collection(self):
        """Load Collection: fila simples — load+scan de todos os produtos."""
        n = len(self.products)
        if n == 0 or self._loading_collection:
            return
        self._loading_collection = True
        for panel in self._panels.values():
            panel.btn_load_collection.disabled = True
        try:
            for i, product in enumerate(self.products, start=1):
                self._activate_product(self.products.index(product))
                panel = self._panels[product]
                try:
                    panel._on_sync(None, label=f"Collection {i}/{n}: {product}")
                except Exception as e:
                    panel._log(f"Load collection error ({product}): {e}", "error")
        finally:
            self._loading_collection = False
            for panel in self._panels.values():
                panel.btn_load_collection.disabled = False
            self._sync_load_collection_styles()
            if self._active_panel is not None:
                self._active_panel.progress_loader.stop("Collection loaded.")

    def _on_select_all_collection(self):
        """Seleciona as units renderizadas de TODOS os produtos da colecao."""
        total = 0
        for panel in self._panels.values():
            for chk in panel.chk_dict.values():
                chk.value = True
                total += 1
        if self._active_panel is not None:
            self._active_panel._log(
                f"Select All Collection: {total} unit(s) selected across the collection.", "info")

    def __getattr__(self, name):
        if name == "_active_panel":
            raise AttributeError(name)
        panel = self.__dict__.get("_active_panel")
        if panel is None:
            raise AttributeError(name)
        return getattr(panel, name)


class CollectionTabs:
    """Abas de colecao dentro de um tema."""

    def __init__(self, country, theme, log_area=None, on_clear_all=None):
        self.country = country
        self.theme = theme
        self.log_area = log_area
        self._on_clear_all_cb = on_clear_all
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
            pp = ProductTabs(self.country, self.theme, coll, self.log_area,
                             on_clear_all=self._on_clear_all_cb)
            self._panels[coll] = pp
            self._placeholders[idx].children = [pp.container]
        else:
            pp = self._panels[coll]
        self._active_panel = pp
        pp.sync_context()

    def sync_context(self):
        """Propaga a sincronizacao de contexto ate o painel ativo."""
        pp = self.__dict__.get("_active_panel")
        if pp is not None:
            pp.sync_context()

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class ThemeTabs:
    """Abas de tema dentro de um pais."""

    def __init__(self, country, log_area=None, on_clear_all=None):
        self.country = country
        self.log_area = log_area
        self._on_clear_all_cb = on_clear_all
        self._build_tabs()

    def _build_tabs(self):
        available = [t for t, colls in config.OBJ.get(self.country, {}).items()
                     if any([p for p in prods if p.get("visible", True)] for prods in colls.values())]
        allowed = getattr(config, "THEMES", None)
        themes = [t for t in allowed if t in available] if allowed else available
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

    def _rebuild_tabs(self):
        """Rebuild tabs with current self.themes (used after filtering)."""
        self._placeholders = [widgets.VBox([]) for _ in self.themes]
        self.tab.children = self._placeholders
        for i, t in enumerate(self.themes):
            self.tab.set_title(i, t)
        self._panels = {}
        self._active_panel = None
        if self.themes:
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
            ct = CollectionTabs(self.country, theme, self.log_area, on_clear_all=self._on_clear_all_cb)
            self._panels[theme] = ct
            self._placeholders[idx].children = [ct.tab]
        else:
            ct = self._panels[theme]
        self._active_panel = ct
        ct.sync_context()

    def sync_context(self):
        """Propaga a sincronizacao de contexto ate o painel ativo."""
        ct = self.__dict__.get("_active_panel")
        if ct is not None:
            ct.sync_context()

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class CountryTabs:
    """Abas de pais -> tema -> colecao -> produto -> unidades."""

    def __init__(self, countries, log_area=None, on_clear_all=None):
        self.countries = list(countries)
        self.log_area = log_area
        self._on_clear_all_cb = on_clear_all
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
            tt = ThemeTabs(code, self.log_area, on_clear_all=self._on_clear_all_cb)
            self._panels[code] = tt
            self._placeholders[idx].children = [tt.tab]
        else:
            tt = self._panels[code]
        self._active_panel = tt
        tt.sync_context()

    def sync_context(self):
        """Propaga a sincronizacao de contexto ate o painel ativo."""
        tt = self.__dict__.get("_active_panel")
        if tt is not None:
            tt.sync_context()

    def __getattr__(self, name):
        return getattr(self._active_panel, name)


class FireMonitorApp:
    """App em guias: Interface (navegacao) + guias em 7 idiomas."""

    def __init__(self, countries):
        self.header = widgets.HTML()
        self.log_area = LogDrawer()
        self.interface = CountryTabs(countries, self.log_area,
                                     on_clear_all=self.clear_all_selections)
        self.guide_widgets = [widgets.HTML() for _ in LANG_ORDER]

        self.tab = widgets.Tab(children=[self.interface.tab] + self.guide_widgets)
        self.tab.set_title(0, "Interface")
        for i, lang in enumerate(LANG_ORDER, start=1):
            self.tab.set_title(i, GUIDES[lang]["tab_title"])

        # Theme multi-select (applies to active country)
        self.theme_select = widgets.SelectMultiple(
            options=[],
            value=[],
            description="Themes:",
            layout=L(width="200px", height="120px"),
            tooltip="Select one or more themes to display"
        )
        # self.theme_select.observe(self._on_theme_change, names="value")

        # Filter panel (next to log drawer)
        # self.filter_panel = self._build_filter_panel()

        # Bottom row: log + filters side by side
        # self.bottom_row = widgets.HBox([
        #     self.log_area.container,
        #     self.filter_panel,
        # ], layout=L(gap="12px", align_items="flex-start"))
        self.bottom_row = widgets.HBox([self.log_area.container])

        self.container = widgets.VBox([self.header, self.tab, self.bottom_row])
        self._render()

    # def _build_filter_panel(self):
    #     """Build the filter configuration panel."""
    #     p = _palette()
    #
    #     # Preset selector
    #     self.preset_dropdown = widgets.Dropdown(
    #         options=["fire_monitor", "all", "lulc_only", "custom"],
    #         value="fire_monitor",
    #         description="Preset:",
    #         layout=L(width="200px")
    #     )
    #     self.preset_dropdown.observe(self._on_preset_change, names="value")
    #
    #     # Theme multi-select (for quick access)
    #     self.filter_theme_select = widgets.SelectMultiple(
    #         options=[],
    #         value=[],
    #         description="Themes:",
    #         layout=L(width="200px", height="100px"),
    #         tooltip="Filter themes (applies to all countries)"
    #     )
    #     self.filter_theme_select.observe(self._on_filter_theme_change, names="value")
    #
    #     # Country multi-select
    #     self.filter_country_select = widgets.SelectMultiple(
    #         options=self.interface.countries,
    #         value=self.interface.countries,
    #         description="Countries:",
    #         layout=L(width="200px", height="100px"),
    #         tooltip="Filter countries"
    #     )
    #     self.filter_country_select.observe(self._on_filter_country_change, names="value")
    #
    #     # Save/Load buttons
    #     self.btn_save_filters = widgets.Button(
    #         description="Save Filters", button_style="success", icon="save",
    #         layout=L(width="130px", height="34px"),
    #         tooltip="Save current filters to monitor_filters.json"
    #     )
    #     self.btn_save_filters.on_click(self._on_save_filters)
    #
    #     self.btn_load_filters = widgets.Button(
    #         description="Load Filters", button_style="info", icon="folder-open",
    #         layout=L(width="130px", height="34px"),
    #         tooltip="Load filters from monitor_filters.json"
    #     )
    #     self.btn_load_filters.on_click(self._on_load_filters)
    #
    #     self.btn_sync_github = widgets.Button(
    #         description="Sync to GitHub", button_style="success", icon="cloud-upload",
    #         layout=L(width="130px", height="34px"),
    #         tooltip="Commit and push monitor_filters.json (load data cache) to GitHub"
    #     )
    #     self.btn_sync_github.on_click(self._on_sync_github)
    #
    #     # Load data cache status
    #     self.cache_status = widgets.HTML(
    #         value='<div style="padding:8px;color:#6c757d;font-size:11px;">Load data cache: empty</div>'
    #     )
    #
    #     panel = widgets.VBox([
    #         widgets.HTML(f'<div style="font-weight:bold;color:{p["title"]};margin-bottom:8px;">Filters & Cache</div>'),
    #         widgets.HTML('<div style="font-size:11px;color:#6c757d;margin-bottom:4px;">Preset</div>'),
    #         self.preset_dropdown,
    #         widgets.HTML('<div style="font-size:11px;color:#6c757d;margin:8px 0 4px;">Countries</div>'),
    #         self.filter_country_select,
    #         widgets.HTML('<div style="font-size:11px;color:#6c757d;margin:8px 0 4px;">Themes</div>'),
    #         self.filter_theme_select,
    #         widgets.HTML('<div style="margin-top:12px;"></div>'),
    #         widgets.HBox([self.btn_save_filters, self.btn_load_filters, self.btn_sync_github], layout=L(gap="8px")),
    #         widgets.HTML('<div style="margin-top:12px;"></div>'),
    #         widgets.HTML('<div style="font-size:11px;color:#6c757d;">Load Data Cache</div>'),
    #         self.cache_status,
    #     ], layout=L(
    #         width="280px",
    #         padding="12px",
    #         border=f"1px solid {p['panel_border']}",
    #         border_radius="5px",
    #         background=p["panel_bg"]
    #     ))
    #     return panel

    # def _on_preset_change(self, change):
    #     preset = change.get("new")
    #     if preset == "fire_monitor":
    #         self.filter_theme_select.value = ("fire",)
    #         self.filter_country_select.value = tuple(self.interface.countries)
    #     elif preset == "all":
    #         self.filter_theme_select.value = ()
    #         self.filter_country_select.value = tuple(self.interface.countries)
    #     elif preset == "lulc_only":
    #         self.filter_theme_select.value = ("lulc", "lulc_10m")
    #         self.filter_country_select.value = tuple(self.interface.countries)
    #     # For "custom", don't auto-change
    #     self._apply_filters()
    #
    # def _on_filter_theme_change(self, change):
    #     if self.preset_dropdown.value != "custom":
    #         self.preset_dropdown.value = "custom"
    #     self._apply_filters()
    #
    # def _on_filter_country_change(self, change):
    #     if self.preset_dropdown.value != "custom":
    #         self.preset_dropdown.value = "custom"
    #     self._apply_filters()

    # def _apply_filters(self):
    #     """Apply filters to the interface."""
    #     themes = list(self.filter_theme_select.value) if self.filter_theme_select.value else []
    #     countries = list(self.filter_country_select.value) if self.filter_country_select.value else self.interface.countries
    #
    #     # Update the interface's theme tabs for each country
    #     for country in self.interface.countries:
    #         if country in self.interface._panels:
    #             tt = self.interface._panels[country]
    #             # Rebuild theme tabs with filtered themes
    #             filtered_themes = [t for t in tt.themes if not themes or t in themes]
    #             if countries and country not in countries:
    #                 # Country filtered out - hide all themes
    #                 tt.themes = []
    #             else:
    #                 tt.themes = filtered_themes
    #             # Rebuild tabs
    #             tt._rebuild_tabs()
    #
    #     # Update theme options in filter panel
    #     active_country = self.interface._active_code
    #     if active_country in self.interface._panels:
    #         tt = self.interface._panels[active_country]
    #         self.theme_select.options = tt.themes
    #         self.theme_select.value = tuple(tt.themes)
    #         self.filter_theme_select.options = tt.themes
    #         self.filter_theme_select.value = tuple([t for t in tt.themes if not themes or t in themes])
    #
    # def _on_theme_change(self, change):
    #     """Handle theme selection from header toolbar."""
    #     selected = list(change.get("new", []))
    #     if selected:
    #         self.filter_theme_select.value = tuple(selected)
    #         self._apply_filters()

    # def _on_save_filters(self, _):
    #     """Save current filters to monitor_filters.json."""
    #     filters = config.load_filters()
    #     filters["preset"] = self.preset_dropdown.value
    #     filters["include_countries"] = list(self.filter_country_select.value)
    #     filters["include_themes"] = list(self.filter_theme_select.value)
    #     # Keep exclude lists empty for now
    #     filters["exclude_countries"] = []
    #     filters["exclude_themes"] = []
    #     config.save_filters(filters)
    #     self._log("Filters saved to monitor_filters.json", "success")
    #
    # def _on_load_filters(self, _):
    #     """Load filters from monitor_filters.json."""
    #     filters = config.load_filters()
    #     self.preset_dropdown.value = filters.get("preset", "fire_monitor")
    #     self.filter_country_select.value = tuple(filters.get("include_countries", self.interface.countries))
    #     self.filter_theme_select.value = tuple(filters.get("include_themes", []))
    #     self._apply_filters()
    #     self._update_cache_status()
    #     self._log("Filters loaded from monitor_filters.json", "success")
    #
    # def _on_sync_github(self, _):
    #     """Commit and push monitor_filters.json to GitHub."""
    #     self.btn_sync_github.disabled = True
    #     self.btn_sync_github.description = "Syncing..."
    #     self.btn_sync_github.icon = "spinner"
    #     try:
    #         ok = config.sync_filters_to_github(repo_path=".", logger=self._log)
    #         if ok:
    #             self._log("Cache synced to GitHub", "success")
    #         else:
    #             self._log("GitHub sync failed", "error")
    #     except Exception as e:
    #         self._log(f"GitHub sync error: {e}", "error")
    #     finally:
    #         self.btn_sync_github.disabled = False
    #         self.btn_sync_github.description = "Sync to GitHub"
    #         self.btn_sync_github.icon = "cloud-upload"
    #
    # def _update_cache_status(self):
    #     """Update the load data cache status display."""
    #     cache = config.get_load_data_cache()
    #     if cache:
    #         total = sum(len(v.get("units", [])) for v in cache.values())
    #         self.cache_status.value = f'<div style="padding:8px;color:#28a745;font-size:11px;">Load data cache: {len(cache)} products, {total} units</div>'
    #     else:
    #         self.cache_status.value = '<div style="padding:8px;color:#6c757d;font-size:11px;">Load data cache: empty</div>'

    def _log(self, message, type="info"):
        if self.log_area:
            self.log_area.append(message)

    def _render(self):
        p = _palette()
        self.header.value = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;width:100%;'
            f'padding:10px 14px;background:{p["header_bg"]};border:1px solid {p["header_border"]};'
            f'border-radius:5px;margin-bottom:8px;">'
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-weight:bold;font-size:17px;color:{p["title"]};">Export & Vectorization</span>'
            f'<span style="color:{p["subtitle"]};font-size:12px;">MapBiomas Fire Monitor</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<div style="color:{p["subtitle"]};font-size:12px;">{config.flag(config.COUNTRY)} {config.COUNTRY.title()}</div>'
            f'</div>'
            f'</div>'
        )
        self.container.children = [self.header, self.tab, self.bottom_row]
        for i, lang in enumerate(LANG_ORDER):
            self.guide_widgets[i].value = (
                f'<div style="padding:12px;background:{p["guide_bg"]};border:1px solid {p["guide_border"]};'
                f'border-radius:5px;">{_guide_html(lang)}</div>'
            )
        # Initialize theme select options
        active_country = self.interface._active_code
        if active_country in self.interface._panels:
            tt = self.interface._panels[active_country]
            self.theme_select.options = tt.themes
            self.theme_select.value = tuple(tt.themes)
            # self.filter_theme_select.options = tt.themes
            # self.filter_theme_select.value = tuple(tt.themes)
        # self._update_cache_status()

    def get_selected_items(self):
        """Agrega a selecao de TODOS os paineis (todos os paises/produtos),
        cada unidade com seu contexto completo."""
        items = []
        seen = set()
        for panel in _iter_unit_panels(self.interface):
            for item in panel.get_selected_items():
                key = (item["country"], item["theme"], item["collection"],
                       item["product"], item["unit"])
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)
        return items

    def clear_all_selections(self):
        """Limpa as selecoes de TODOS os paineis (todos os paises/produtos)."""
        count = 0
        for panel in _iter_unit_panels(self.interface):
            panel._on_clear(None)
            count += 1
        self._log(f"Cleared selections in {count} product(s).", "info")

    def sync_contexts(self, items):
        """Sincroniza apenas os paineis afetados pelos contextos processados."""
        targets = {(i.get("country"), i.get("theme"),
                    i.get("collection"), i.get("product")) for i in items}
        synced = False
        for panel in _iter_unit_panels(self.interface):
            if panel.context_key() in targets:
                try:
                    panel.sync()
                    synced = True
                except Exception as e:
                    self._log(f"Sync failed for {panel.context_key()}: {e}", "error")
        if not synced:
            self.sync()

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
