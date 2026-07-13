import ipywidgets as widgets
from IPython.display import display, clear_output

from .state import list_months_in_collection, build_state

L = widgets.Layout

_STATUS_CSS = widgets.HTML("""<style>
.mfm-ok   { background:#d4edda !important; border:1px solid #c3e6cb !important; }
.mfm-run  { background:#fff3cd !important; border:1px solid #ffeaa8 !important; }
.mfm-null { background:#f8f9fa !important; border:1px solid #dee2e6 !important; }
</style>""")


def _badge(ok, label_ok="OK", label_miss="MISS"):
    if ok:
        return (
            '<span style="background:#28a745;color:#fff;padding:2px 7px;'
            'border-radius:3px;font-size:11px;font-weight:700;">'
            f'{label_ok}</span>'
        )
    return (
        '<span style="background:#e9ecef;color:#6c757d;padding:2px 7px;'
        'border-radius:3px;font-size:11px;">'
        f'{label_miss}</span>'
    )


class MonitorUI:
    _DATE_W = "110px"
    _CELL_W = "85px"
    _SEL_W  = "44px"

    def __init__(self):
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.log_area = widgets.Output()

        self.grid_container = widgets.VBox([
            widgets.HTML(
                '<div style="padding:20px;text-align:center;color:#6c757d;font-size:13px;">'
                '<i>Carregando meses disponiveis na colecao...</i></div>'
            )
        ])

        self.btn_sync = widgets.Button(
            description="Sincronizar", button_style="success", icon="refresh",
            layout=L(width="180px", height="34px")
        )
        self.btn_sync.on_click(self._on_sync)

        self.btn_select_pending = widgets.Button(
            description="Selecionar Pendentes", button_style="info",
            layout=L(width="200px", height="34px")
        )
        self.btn_select_pending.on_click(self._on_select_pending)

        self.btn_clear = widgets.Button(
            description="Limpar", button_style="warning",
            layout=L(width="80px", height="34px")
        )
        self.btn_clear.on_click(self._on_clear)

        self.loader = widgets.HTML(
            value='<span id="mon-loader" style="display:none;margin-left:10px;color:#3498db;font-size:13px;">Sincronizando...</span>'
        )

        header = widgets.HTML("""
        <div style="display:flex;align-items:center;justify-content:space-between;width:100%;padding:10px 14px;background:#fff;border-bottom:2px solid #333;margin-bottom:10px;">
            <div>
                <span style="font-weight:bold;font-size:17px;color:#333;">Export &amp; Vectorization</span>
                <span style="color:#6c757d;font-size:12px;margin-left:14px;">Monitor do Fogo &mdash; Brasil</span>
            </div>
            <div style="padding:4px 14px;background:#fff1f0;border:1px solid #ffa39e;border-radius:4px;">
                <span style="color:#cf1322;font-size:11px;font-weight:bold;">MapBiomas Fire Monitor</span>
            </div>
        </div>
        """)

        instructions = widgets.HTML("""
        <div style="padding:6px 10px;margin-bottom:8px;background:#e8f4fd;border:1px solid #bee5eb;border-radius:4px;font-size:12px;color:#0c5460;line-height:1.6;">
            <strong>Como usar:</strong>
            a) Clique em <strong>Sincronizar</strong> para verificar o status de cada mes no GCS e GEE.
            b) Marque os checkboxes dos meses que deseja processar.
            c) Execute as celulas abaixo em ordem: <em>Export</em> &rarr; <em>Mosaico</em> &rarr; <em>Vetorizacao</em> &rarr; <em>Upload GEE</em>.
            <br>
            <span style="color:#28a745;font-weight:700;">OK</span> = etapa concluida &nbsp;|&nbsp;
            <span style="color:#6c757d;">MISS</span> = etapa pendente
        </div>
        """)

        footer = widgets.HBox([
            self.btn_select_pending, self.btn_clear, self.btn_sync, self.loader,
        ], layout=L(margin="10px 0 6px 0", gap="10px", align_items="center"))

        self.container = widgets.VBox([
            _STATUS_CSS,
            header,
            instructions,
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
                f'<span style="color:{color};font-size:12px;">[{type.upper()}] {message}</span>'
            ))

    def _on_sync(self, _):
        if self.is_refreshing:
            return
        self.is_refreshing = True
        self.btn_sync.disabled = True
        self.btn_sync.description = "Sincronizando..."
        self.loader.value = self.loader.value.replace("display:none", "display:flex")
        self._log("Verificando arquivos no GCS e assets no GEE...", "info")
        try:
            self.state = build_state(logger=self._log)
            self._render_grid()
            completed = sum(
                1 for k, v in self.state.items()
                if k != "updated_at" and v.get("exported") and v.get("mosaiced")
                and v.get("vectorized_gcs") and v.get("vectorized_gee")
            )
            total = len([k for k in self.state if k != "updated_at"])
            self._log(f"Sincronizacao concluida: {completed}/{total} meses completos.", "success")
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
            widgets.HTML(f'<div style="width:{self._DATE_W};font-weight:700;font-size:12px;color:#fff;">Data</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:12px;color:#fff;">Export</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:12px;color:#fff;">Mosaico</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:12px;color:#fff;">Vetor GCS</div>'),
            widgets.HTML(f'<div style="width:{self._CELL_W};text-align:center;font-weight:700;font-size:12px;color:#fff;">Vetor GEE</div>'),
            widgets.HTML(f'<div style="width:{self._SEL_W};text-align:center;font-weight:700;font-size:12px;color:#fff;">Sel</div>'),
        ], layout=L(
            background="#343a40", padding="6px 10px", min_height="34px",
            align_items="center", overflow="visible"
        ))

        rows = [header_row]

        months = sorted(
            [k for k in self.state.keys() if k != "updated_at"],
            reverse=True
        )

        row_layout = L(
            align_items="center", min_height="38px",
            border_bottom="1px solid #dee2e6", padding="3px 10px",
            overflow="visible", width="100%"
        )

        for i, m in enumerate(months):
            info = self.state.get(m, {})
            exp_ok = info.get("exported", False)
            mos_ok = info.get("mosaiced", False)
            vgc_ok = info.get("vectorized_gcs", False)
            vge_ok = info.get("vectorized_gee", False)

            all_ok = exp_ok and mos_ok and vgc_ok and vge_ok

            bg = "#fcfcfc" if i % 2 == 0 else "#fff"

            date_cell = widgets.HTML(
                f'<div style="width:{self._DATE_W};font-family:monospace;font-size:13px;color:#212529;font-weight:600;">{m}</div>'
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

            chk = widgets.Checkbox(value=False, indent=False, layout=L(width="20px", height="20px", margin="0 auto"))
            if all_ok:
                chk.disabled = True

            chk_wrapper = widgets.HBox([chk], layout=L(width=self._SEL_W, justify_content="center"))

            self.chk_dict[m] = chk

            row = widgets.HBox(
                [date_cell, exp_cell, mos_cell, vgc_cell, vge_cell, chk_wrapper],
                layout=row_layout
            )
            row.layout.background = bg
            rows.append(row)

        n_complete = sum(
            1 for k, v in self.state.items()
            if k != "updated_at" and v.get("exported") and v.get("mosaiced")
            and v.get("vectorized_gcs") and v.get("vectorized_gee")
        )

        legend = widgets.HTML(
            f'<div style="font-size:11px;color:#6c757d;margin:6px 0 0 10px;padding:6px 10px;'
            f'background:#f8f9fa;border-radius:4px;">'
            f'{len(months)} meses na colecao &nbsp;|&nbsp; '
            f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> completos &nbsp;|&nbsp; '
            f'<span style="color:#6c757d;">{len(months) - n_complete}</span> pendentes'
            f'</div>'
        )

        self.grid_container.children = [
            widgets.VBox(rows, layout=L(
                max_height="460px", width="100%",
                overflow_y="auto", overflow_x="hidden",
                padding="0", border="1px solid #ced4da",
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

    months = list_months_in_collection()
    if months:
        for m in months:
            ui.state[m] = {"exported": False, "mosaiced": False, "vectorized_gcs": False, "vectorized_gee": False}
        ui._render_grid()
        ui._log(f"{len(months)} meses na colecao. Sincronizando automaticamente...", "info")
    else:
        ui._log("Nao foi possivel consultar a colecao. Verifique a autenticacao GEE.", "warning")

    ui._on_sync(None)

    return ui
