import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output

from state import get_state, build_state
from export import start_export
from mosaic import assemble_mosaic
from vectorize import vectorize_month, upload_to_gee


_STATUS_CSS = """<style>
.mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
.mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
.mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
.mfm-grid { width:100%; border-collapse:collapse; font-size:12px; }
.mfm-grid th { background:#343a40; color:#fff; padding:6px 8px; text-align:center; font-weight:600; position:sticky; top:0; z-index:2; }
.mfm-grid td { padding:4px 8px; text-align:center; border-bottom:1px solid #dee2e6; }
.mfm-grid tr:hover { background:#f1f3f5; }
.mfm-grid .col-date { text-align:left; font-family:monospace; }
.mfm-grid .col-sel { width:40px; }
</style>"""


def _status_badge(ok, label_ok="OK", label_miss="MISS"):
    if ok:
        return f'<span style="color:#155724;font-weight:700;font-size:10px;">{label_ok}</span>'
    return f'<span style="color:#adb5bd;font-size:10px;">{label_miss}</span>'


def _build_grid(state):
    months = sorted(
        [k for k in state.keys() if k != "updated_at"],
        reverse=True
    )

    now = datetime.datetime.now()
    current_ym = f"{now.year}_{now.month:02d}"

    rows_html = []
    for m in months:
        info = state.get(m, {})
        exp = info.get("exported", False)
        mos = info.get("mosaiced", False)
        vgc = info.get("vectorized_gcs", False)
        vge = info.get("vectorized_gee", False)

        exp_badge = _status_badge(exp)
        mos_badge = _status_badge(mos)
        vgc_badge = _status_badge(vgc)
        vge_badge = _status_badge(vge)

        show_sel = not (exp and mos and vgc and vge)

        rows_html.append(
            f'<tr>'
            f'<td class="col-date">{m}</td>'
            f'<td>{exp_badge}</td>'
            f'<td>{mos_badge}</td>'
            f'<td>{vgc_badge}</td>'
            f'<td>{vge_badge}</td>'
            f'<td class="col-sel">{"☐" if show_sel else "✓"}</td>'
            f'</tr>'
        )

    grid_html = f'''
    <div style="max-height:500px; overflow-y:auto; overflow-x:hidden; border:1px solid #ddd; background:#fff;">
        <table class="mfm-grid">
            <thead>
                <tr>
                    <th style="width:100px;">Data</th>
                    <th style="width:80px;">Export</th>
                    <th style="width:80px;">Mosaico</th>
                    <th style="width:80px;">Vetor GCS</th>
                    <th style="width:80px;">Vetor GEE</th>
                    <th style="width:50px;">Sel</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows_html)}
            </tbody>
        </table>
    </div>
    <p style="font-size:10px;color:#666;margin-top:4px;">
        <span style="color:#155724;font-weight:700">OK</span> = completo
        &nbsp;|&nbsp;
        <span style="color:#adb5bd">MISS</span> = pendente
        &nbsp;|&nbsp;
        {len(months)} meses encontrados
    </p>
    '''
    return widgets.HTML(grid_html)


class MonitorUI:
    def __init__(self):
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.is_processing = False
        self.log_area = widgets.Output()

        self.grid_container = widgets.VBox([
            widgets.HTML("<i>Clique em Sincronizar para carregar o estado...</i>")
        ])

        self.btn_sync = widgets.Button(
            description="Sincronizar",
            button_style="success",
            icon="refresh",
            layout=widgets.Layout(width="180px")
        )
        self.btn_sync.on_click(self._on_sync)

        self.btn_select_pending = widgets.Button(
            description="Selecionar Pendentes",
            button_style="info",
            layout=widgets.Layout(width="180px")
        )
        self.btn_select_pending.on_click(self._on_select_pending)

        self.btn_clear = widgets.Button(
            description="Limpar",
            button_style="warning",
            layout=widgets.Layout(width="80px")
        )
        self.btn_clear.on_click(self._on_clear)

        self.btn_process = widgets.Button(
            description="Processar Selecionados",
            button_style="danger",
            layout=widgets.Layout(width="220px")
        )
        self.btn_process.on_click(self._on_process)

        self.loader = widgets.HTML(
            value='<span id="mon-loader" style="display:none;margin-left:10px;color:#3498db;">⏳ Processando...</span>'
        )

        header = widgets.HTML(f'''
        <div style="display:flex;align-items:center;justify-content:space-between;width:100%;padding:8px 12px;background:#fff;border-bottom:2px solid #333;margin-bottom:10px;">
            <div>
                <span style="font-weight:bold;font-size:16px;color:#333;">Export & Vectorization</span>
                <span style="color:#888;font-size:11px;margin-left:12px;">Monitor do Fogo — Brasil</span>
            </div>
            <div style="padding:3px 12px;background:#fff1f0;border:1px solid #ffa39e;border-radius:4px;">
                <span style="color:#cf1322;font-size:10px;font-weight:bold;">MapBiomas Fire Monitor</span>
            </div>
        </div>
        ''')

        footer = widgets.VBox([
            widgets.HBox([
                self.btn_select_pending,
                self.btn_clear,
                self.btn_sync,
                self.loader,
            ], layout=widgets.Layout(margin="10px 0", gap="10px", align_items="center")),
            widgets.HBox([self.btn_process], layout=widgets.Layout(margin="5px 0")),
        ])

        self.container = widgets.VBox([
            widgets.HTML(_STATUS_CSS),
            header,
            self.grid_container,
            footer,
            self.log_area,
        ], layout=widgets.Layout(
            border="1px solid #ccc",
            padding="10px",
            border_radius="5px",
            margin="10px 0"
        ))

    def display(self):
        display(self.container)

    def _log(self, message, type="info"):
        colors = {"info": "#3498db", "success": "#27ae60", "error": "#d32f2f", "warning": "#e67e22"}
        color = colors.get(type, "#333")
        with self.log_area:
            display(widgets.HTML(
                f'<span style="color:{color};font-size:11px;">[{type.upper()}] {message}</span>'
            ))

    def _on_sync(self, _):
        if self.is_refreshing:
            return
        self.is_refreshing = True
        self.btn_sync.disabled = True
        self.btn_sync.description = "Sincronizando..."
        self._log("Iniciando sincronizacao (GCS + GEE)...", "info")
        try:
            self.state = build_state(logger=self._log)
            self._render_grid()
            self._log("Sincronizacao concluida.", "success")
        except Exception as e:
            self._log(f"Erro na sincronizacao: {e}", "error")
        finally:
            self.is_refreshing = False
            self.btn_sync.disabled = False
            self.btn_sync.description = "Sincronizar"

    def _render_grid(self):
        self.grid_container.children = [_build_grid(self.state)]

    def _on_select_pending(self, _):
        self._render_grid()
        self._log("Grid atualizado. Meses pendentes podem ser processados.", "info")

    def _on_clear(self, _):
        self._render_grid()
        self._log("Selecao limpa.", "info")

    def _on_process(self, _):
        if self.is_processing:
            self._log("Ja existe um processamento em andamento.", "warning")
            return
        self.is_processing = True
        self.btn_process.disabled = True
        self.btn_process.description = "Processando..."
        self._log("Iniciando processamento dos meses pendentes...", "info")

        try:
            months = sorted(
                [k for k in self.state.keys() if k != "updated_at"],
                reverse=True
            )
            for m in months:
                info = self.state.get(m, {})
                if info.get("exported") and info.get("mosaiced") and info.get("vectorized_gcs") and info.get("vectorized_gee"):
                    continue

                parts = m.split("_")
                y, mo = int(parts[0]), int(parts[1])
                self._log(f"--- Processando {m} ---", "info")

                if not info.get("exported"):
                    self._log(f"[EXPORT] {m}", "info")
                    start_export(y, mo, logger=self._log)
                    self.state = build_state(logger=None)
                    self._render_grid()

                if not info.get("mosaiced"):
                    if info.get("exported"):
                        self._log(f"[MOSAIC] {m}", "info")
                        assemble_mosaic(y, mo, logger=self._log)
                        self.state = build_state(logger=None)
                        self._render_grid()

                info2 = self.state.get(m, {})
                if not info2.get("vectorized_gcs"):
                    if info2.get("mosaiced"):
                        self._log(f"[VECTORIZE] {m}", "info")
                        vectorize_month(y, mo, logger=self._log)
                        self.state = build_state(logger=None)
                        self._render_grid()

                info3 = self.state.get(m, {})
                if not info3.get("vectorized_gee"):
                    if info3.get("vectorized_gcs"):
                        self._log(f"[UPLOAD GEE] {m}", "info")
                        upload_to_gee(y, mo, logger=self._log)
                        self.state = build_state(logger=None)
                        self._render_grid()

            self._log("Processamento concluido.", "success")
        except Exception as e:
            self._log(f"Erro no processamento: {e}", "error")
        finally:
            self.is_processing = False
            self.btn_process.disabled = False
            self.btn_process.description = "Processar Selecionados"


def run_ui():
    ui = MonitorUI()
    ui.display()

    state = get_state()
    if state and len(state) > 1:
        ui.state = state
        ui._render_grid()
    else:
        ui._log("Cache vazio. Clique em Sincronizar para carregar o estado.", "info")

    return ui
