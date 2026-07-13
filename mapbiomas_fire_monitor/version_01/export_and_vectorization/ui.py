import ipywidgets as widgets
from IPython.display import display, clear_output

from .state import get_state, build_state

L = widgets.Layout

_STATUS_CSS = widgets.HTML("""<style>
.mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
.mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
.mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
</style>""")


def _badge(ok, label_ok="OK", label_miss="MISS"):
    if ok:
        return f'<span style="color:#155724;font-weight:700;font-size:10px;">{label_ok}</span>'
    return f'<span style="color:#adb5bd;font-size:10px;">{label_miss}</span>'


class MonitorUI:
    _DATE_W = "100px"
    _CELL_W = "80px"
    _SEL_W  = "42px"

    def __init__(self):
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.log_area = widgets.Output()

        self.grid_container = widgets.VBox([
            widgets.HTML("<i>Clique em Sincronizar para carregar o estado...</i>")
        ])

        self.btn_sync = widgets.Button(
            description="Sincronizar", button_style="success", icon="refresh",
            layout=L(width="180px")
        )
        self.btn_sync.on_click(self._on_sync)

        self.btn_select_pending = widgets.Button(
            description="Selecionar Pendentes", button_style="info",
            layout=L(width="200px")
        )
        self.btn_select_pending.on_click(self._on_select_pending)

        self.btn_clear = widgets.Button(
            description="Limpar", button_style="warning",
            layout=L(width="80px")
        )
        self.btn_clear.on_click(self._on_clear)

        self.loader = widgets.HTML(
            value='<span id="mon-loader" style="display:none;margin-left:10px;color:#3498db;">Sincronizando...</span>'
        )

        header = widgets.HTML(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;width:100%;padding:8px 12px;background:#fff;border-bottom:2px solid #333;margin-bottom:10px;">
            <div>
                <span style="font-weight:bold;font-size:16px;color:#333;">Export &amp; Vectorization</span>
                <span style="color:#888;font-size:11px;margin-left:12px;">Monitor do Fogo &mdash; Brasil</span>
            </div>
            <div style="padding:3px 12px;background:#fff1f0;border:1px solid #ffa39e;border-radius:4px;">
                <span style="color:#cf1322;font-size:10px;font-weight:bold;">MapBiomas Fire Monitor</span>
            </div>
        </div>
        """)

        footer = widgets.HBox([
            self.btn_select_pending, self.btn_clear, self.btn_sync, self.loader,
        ], layout=L(margin="10px 0", gap="10px", align_items="center"))

        self.container = widgets.VBox([
            _STATUS_CSS,
            header,
            self.grid_container,
            footer,
            self.log_area,
        ], layout=L(
            border="1px solid #ccc", padding="10px",
            border_radius="5px", margin="10px 0"
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
        self.loader.value = self.loader.value.replace("display:none", "display:flex")
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
            self.loader.value = self.loader.value.replace("display:flex", "display:none")

    def _render_grid(self):
        self.chk_dict = {}

        header_row = widgets.HBox([
            widgets.HTML(f'<div style="width:{self._DATE_W};font-weight:700;font-size:11px;color:#fff;">Data</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:11px;color:#fff;">Export</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:11px;color:#fff;">Mosaico</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:11px;color:#fff;">Vetor GCS</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:11px;color:#fff;">Vetor GEE</div>'),
            widgets.HTML(f'<div style="width:{self._SEL_W};text-align:center;font-weight:700;font-size:11px;color:#fff;">Sel</div>'),
        ], layout=L(
            background="#343a40", padding="5px 8px", min_height="30px",
            align_items="center", overflow="visible"
        ))

        rows = [header_row]

        months = sorted(
            [k for k in self.state.keys() if k != "updated_at"],
            reverse=True
        )

        row_layout = L(
            align_items="center", min_height="32px",
            border_bottom="1px solid #eee", padding="2px 8px",
            overflow="visible", width="100%"
        )

        for m in months:
            info = self.state.get(m, {})
            exp_ok = info.get("exported", False)
            mos_ok = info.get("mosaiced", False)
            vgc_ok = info.get("vectorized_gcs", False)
            vge_ok = info.get("vectorized_gee", False)

            all_ok = exp_ok and mos_ok and vgc_ok and vge_ok

            date_cell = widgets.HTML(
                f'<div style="width:{self._DATE_W};font-family:monospace;font-size:11px;">{m}</div>'
            )

            exp_cell = widgets.HTML(
                f'<div style="width:{self._CELL_W};text-align:center;">{_badge(exp_ok)}</div>'
            )
            mos_cell = widgets.HTML(
                f'<div style="width:{self._CELL_W};text-align:center;">{_badge(mos_ok)}</div>'
            )
            vgc_cell = widgets.HTML(
                f'<div style="width:{self._CELL_W};text-align:center;">{_badge(vgc_ok)}</div>'
            )
            vge_cell = widgets.HTML(
                f'<div style="width:{self._CELL_W};text-align:center;">{_badge(vge_ok)}</div>'
            )

            chk = widgets.Checkbox(value=False, indent=False, layout=L(width="18px", height="18px", margin="0 auto"))
            if all_ok:
                chk.disabled = True

            chk_wrapper = widgets.HBox([chk], layout=L(width=self._SEL_W, justify_content="center"))

            self.chk_dict[m] = chk

            rows.append(widgets.HBox(
                [date_cell, exp_cell, mos_cell, vgc_cell, vge_cell, chk_wrapper],
                layout=row_layout
            ))

        legend = widgets.HTML(
            '<p style="font-size:10px;color:#666;margin:4px 0 0 8px;">'
            '<span style="color:#155724;font-weight:700">OK</span> = completo'
            ' &nbsp;|&nbsp; '
            '<span style="color:#adb5bd">MISS</span> = pendente'
            f' &nbsp;|&nbsp; {len(months)} meses'
            '</p>'
        )

        self.grid_container.children = [
            widgets.VBox(rows, layout=L(
                max_height="450px", width="100%",
                overflow_y="auto", overflow_x="hidden",
                padding="0", border="1px solid #ddd",
                background_color="#fff"
            )),
            legend,
        ]

    def _on_select_pending(self, _):
        for key, chk in self.chk_dict.items():
            if not chk.disabled:
                chk.value = True

    def _on_clear(self, _):
        for chk in self.chk_dict.values():
            chk.value = False

    def get_selected_months(self):
        result = []
        for key, chk in self.chk_dict.items():
            if chk.value and not chk.disabled:
                parts = key.split("_")
                if len(parts) >= 2:
                    result.append((int(parts[0]), int(parts[1])))
        return result

    def sync(self):
        self.state = build_state(logger=self._log)
        self._render_grid()


def run_ui():
    ui = MonitorUI()
    ui.display()
    ui._log("Clique em Sincronizar para carregar o estado.", "info")
    return ui
