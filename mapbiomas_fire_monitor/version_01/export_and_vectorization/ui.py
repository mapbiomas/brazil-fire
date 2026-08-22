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


def _palette(theme):
    if theme == "dark":
        return {
            "panel_bg": "#1e1e1e",
            "panel_border": "#444444",
            "header_bg": "#2d2d2d",
            "header_border": "#555555",
            "title": "#e8e8e8",
            "subtitle": "#b0b0b0",
            "grid_header_bg": "#3a3a3a",
            "grid_header_fg": "#ffffff",
            "row_a": "#262626",
            "row_b": "#202020",
            "date_bg": "#333333",
            "date_fg": "#e0e0e0",
            "legend_bg": "#2a2a2a",
            "legend_fg": "#b8b8b8",
            "hint_bg": "#3a3320",
            "hint_border": "#665c3a",
            "hint_fg": "#d0c9b0",
            "inst_bg": "#12354a",
            "inst_border": "#1d4e6b",
            "inst_fg": "#a8d4ea",
            "border": "#444444",
        }
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
        "border": "#cccccc",
    }


def _badge(ok):
    if ok:
        return (
            '<span style="background:#28a745;color:#fff;padding:2px 7px;'
            'border-radius:3px;font-size:11px;font-weight:700;">OK</span>'
        )
    return (
        '<span style="background:#e9ecef;color:#6c757d;padding:2px 7px;'
        'border-radius:3px;font-size:11px;">MISS</span>'
    )


def _badge_link(url):
    return (
        f'<a href="{url}" target="_blank" rel="noopener" title="Baixar" '
        f'style="background:#28a745;color:#fff;padding:2px 7px;border-radius:3px;'
        f'font-size:11px;font-weight:700;text-decoration:none;cursor:pointer;display:inline-block;">OK</a>'
    )


def _badge_copy(asset_id):
    js = asset_id.replace("'", "\\'")
    return (
        f'<a href="#" title="Copiar asset ID" '
        f'onclick="navigator.clipboard.writeText(\'{js}\');return false;" '
        f'style="background:#28a745;color:#fff;padding:2px 7px;border-radius:3px;'
        f'font-size:11px;font-weight:700;text-decoration:none;cursor:pointer;display:inline-block;">OK</a>'
    )


# (chave do estado, titulo da coluna, numero da etapa, tipo de badge)
_COLS = [
    ("exported",         "Export",          1, "badge"),
    ("mosaiced",         "Mosaico",         2, "link_mosaic"),
    ("vectorized_gcs",   "Vetor GCS",       3, "link_vector"),
    ("vectorized_gee",   "Vetor GEE",       4, "copy_asset"),
    ("published_mosaic", "Publico mosaico", 5, "link_pub_mosaic"),
    ("published_vector", "Publico vetor",   6, "link_pub_vector"),
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


class MonitorUI:
    _DATE_W = "100px"
    _CELL_W = "76px"
    _SEL_W  = "56px"

    def __init__(self):
        self.state = {"updated_at": None}
        self.chk_dict = {}
        self.is_refreshing = False
        self.theme = "light"
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

        self.btn_select_all = widgets.Button(
            description="Selecionar Todos", button_style="info",
            layout=L(width="150px", height="34px")
        )
        self.btn_select_all.on_click(self._on_select_all)

        self.theme_btn = widgets.Button(
            description="🌙", layout=L(width="48px", height="34px")
        )
        self.theme_btn.on_click(self._on_theme_toggle)

        self.force_chk = widgets.Checkbox(
            description="Forçar reprocessamento", indent=False,
            layout=L(width="200px")
        )

        self.year_filter = None
        self.year_dropdown = widgets.Dropdown(
            options=["Todos os anos"], value="Todos os anos",
            description="Ano:", layout=L(width="240px")
        )
        self.year_dropdown.observe(self._on_year_change, names="value")

        self.loader = widgets.HTML(
            value='<span id="mon-loader" style="display:none;margin-left:10px;color:#3498db;font-size:13px;">Sincronizando...</span>'
        )

        self.header = widgets.HTML()
        self.instructions = widgets.HTML()

        footer = widgets.HBox([
            self.btn_select_pending, self.btn_select_all, self.btn_clear,
            self.btn_sync, self.loader, self.theme_btn, self.force_chk,
        ], layout=L(margin="10px 0 6px 0", gap="10px", align_items="center"))

        self.container = widgets.VBox([
            _STATUS_CSS,
            self.header,
            self.instructions,
            widgets.HBox([self.year_dropdown], layout=L(margin="0 0 8px 10px")),
            self.grid_container,
            footer,
            self.log_area,
        ])

        self._apply_theme()

    # ---- tema ----
    def _on_theme_toggle(self, _):
        self.theme = "dark" if self.theme == "light" else "light"
        self._apply_theme()

    def _apply_theme(self):
        p = _palette(self.theme)
        self.header.value = (
            f'<div style="display:flex;align-items:center;justify-content:space-between;width:100%;'
            f'padding:10px 14px;background:{p["header_bg"]};border-bottom:2px solid {p["header_border"]};margin-bottom:10px;">'
            f'<div>'
            f'<span style="font-weight:bold;font-size:17px;color:{p["title"]};">Export &amp; Vectorization</span>'
            f'<span style="color:{p["subtitle"]};font-size:12px;margin-left:14px;">Monitor do Fogo &mdash; '
            f'{config.flag(config.COUNTRY)} {config.COUNTRY.title()}</span>'
            f'</div>'
            f'<div style="padding:4px 14px;background:#fff1f0;border:1px solid #ffa39e;border-radius:4px;">'
            f'<span style="color:#cf1322;font-size:11px;font-weight:bold;">MapBiomas Fire Monitor</span>'
            f'</div>'
            f'</div>'
        )
        self.instructions.value = (
            f'<div style="padding:6px 10px;margin-bottom:8px;background:{p["inst_bg"]};'
            f'border:1px solid {p["inst_border"]};border-radius:4px;font-size:12px;'
            f'color:{p["inst_fg"]};line-height:1.6;">'
            f'<strong>Como usar:</strong> '
            f'a) Escolha o pais na aba acima (cada aba tem sua propria grid). '
            f'b) Clique em <strong>Sincronizar</strong> para ver o status. '
            f'c) Marque os meses e execute as celulas em ordem (Etapa 1..7): '
            f'<em>Export</em> &rarr; <em>Mosaico</em> &rarr; <em>Vetorizacao</em> &rarr; '
            f'<em>Upload GEE</em> &rarr; <em>Publicar mosaico</em> &rarr; <em>Publicar vetor</em> &rarr; <em>Limpar temp</em>.'
            f'<br>'
            f'<span style="color:#28a745;font-weight:700;">OK</span> = concluida &nbsp;|&nbsp; '
            f'<span style="color:#6c757d;">MISS</span> = pendente &nbsp;|&nbsp; '
            f'<span style="text-decoration:underline;">OK com link</span> = baixa o dado'
            f'</div>'
        )
        self.theme_btn.description = "☀️" if self.theme == "dark" else "🌙"
        self.container.layout = L(
            border=f"1px solid {p['panel_border']}", padding="10px",
            border_radius="5px", margin="10px 0", background=p["panel_bg"]
        )
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
            self._log(f"{len(months)} meses na colecao. Sincronizando automaticamente...", "info")
        else:
            self._log("Nao foi possivel consultar a colecao. Verifique a autenticacao GEE.", "warning")
        self._on_sync(None)

    def _on_sync(self, _):
        if self.is_refreshing:
            return
        self.is_refreshing = True
        self.btn_sync.disabled = True
        self.btn_sync.description = "Sincronizando..."
        self.loader.value = self.loader.value.replace("display:none", "display:flex")
        self._log("Verificando arquivos no GCS e assets no GEE...", "info")
        try:
            selected = self._get_selected_keys()
            self.state = build_state(logger=self._log)
            self._render_grid()
            self._restore_selected(selected)
            completed = sum(1 for k, v in self.state.items() if k != "updated_at" and _is_complete(v))
            total = len([k for k in self.state if k != "updated_at"])
            self._log(f"Sincronizacao concluida: {completed}/{total} meses completos.", "success")
        except Exception as e:
            self._log(f"Erro na sincronizacao: {e}", "error")
        finally:
            self.is_refreshing = False
            self.btn_sync.disabled = False
            self.btn_sync.description = "Sincronizar"
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
        options = ["Todos os anos"] + [str(y) for y in years]
        if self.year_dropdown.value not in options:
            self.year_dropdown.value = "Todos os anos"
        self.year_dropdown.options = options

    def _on_year_change(self, change):
        value = change.get("new")
        self.year_filter = int(value) if value != "Todos os anos" else None
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
        p = _palette(self.theme)

        def _header_cell(width, title, etapa):
            return widgets.HTML(
                f'<div style="width:{width};text-align:center;font-weight:700;font-size:10px;'
                f'color:{p["grid_header_fg"]};line-height:1.25;">'
                f'{title}<br><span style="font-size:9px;font-weight:400;opacity:.85;">Etapa {etapa}</span>'
                f'</div>'
            )

        header_row = widgets.HBox(
            [widgets.HTML(f'<div style="width:{self._DATE_W};font-weight:700;font-size:12px;color:{p["grid_header_fg"]};">Data</div>')]
            + [_header_cell(self._CELL_W, t, e) for _, t, e, _ in _COLS]
            + [widgets.HTML(f'<div style="width:{self._SEL_W};text-align:center;font-weight:700;font-size:11px;color:{p["grid_header_fg"]};">Sel</div>')],
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
                f'background:{p["date_bg"]};padding:2px 6px;border-radius:3px;">{m}</div>'
            )

            cells = [date_cell]
            for key, _t, _e, kind in _COLS:
                ok = info.get(key, False)
                cells.append(widgets.HTML(
                    f'<div style="width:{self._CELL_W};text-align:center;">{self._col_content(kind, ok, y, mm)}</div>'
                ))

            chk = widgets.Checkbox(value=False, indent=False, layout=L(width="20px", height="20px"))
            if _is_complete(info) and not self.force_chk.value:
                chk.disabled = True

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
                f'{n_visible} meses de {self.year_filter} no filtro &nbsp;|&nbsp; '
                f'<span style="color:#28a745;font-weight:700;">{n_visible_complete}</span> completos &nbsp;|&nbsp; '
                f'<span style="color:#6c757d;">{n_visible - n_visible_complete}</span> pendentes'
            )
        else:
            label = (
                f'{n_all} meses na colecao &nbsp;|&nbsp; '
                f'<span style="color:#28a745;font-weight:700;">{n_complete}</span> completos &nbsp;|&nbsp; '
                f'<span style="color:#6c757d;">{n_all - n_complete}</span> pendentes'
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
            f'<b>Export</b>=célula Etapa 1 &nbsp;|&nbsp; '
            f'<b>Mosaico</b>=célula Etapa 2 &nbsp;|&nbsp; '
            f'<b>Vetor GCS</b>=célula Etapa 3 &nbsp;|&nbsp; '
            f'<b>Vetor GEE</b>=célula Etapa 4 &nbsp;|&nbsp; '
            f'<b>Publico mosaico</b>=célula Etapa 5 &nbsp;|&nbsp; '
            f'<b>Publico vetor</b>=célula Etapa 6 &nbsp;|&nbsp; '
            f'<b>Clean temp</b>=célula Etapa 7 (após os dois publicados)'
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
            if not chk.disabled:
                chk.value = True

    def _on_select_all(self, _):
        for chk in self.chk_dict.values():
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

    def _get_selected_keys(self):
        return [k for k, chk in self.chk_dict.items() if chk.value and not chk.disabled]

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
    """Aba por pais. Cada aba tem seu proprio MonitorUI (build lazy).

    Delega get_selected_months/_log/sync para o painel da aba ativa, entao
    as celulas de processamento continuam usando `ui` sem mudanca.
    """

    def __init__(self, countries):
        self.countries = list(countries)
        if not self.countries:
            raise ValueError("Nenhum pais configurado para as abas.")
        for c in self.countries:
            if c not in config.COUNTRIES:
                raise ValueError(f"Pais '{c}' nao existe em config.COUNTRIES.")

        self._panels = {}
        self._active_code = self.countries[0]
        self._active_panel = None

        self._placeholders = [widgets.VBox([]) for _ in self.countries]
        self.tab = widgets.Tab(children=self._placeholders)
        for i, c in enumerate(self.countries):
            self.tab.set_title(i, f"{config.flag(c)} {c.title()}")

        self.tab.observe(self._on_tab_change, names="selected_index")
        self._activate(0)

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


def run_ui(countries=None):
    countries = countries or config.COUNTRIES_AVAILABLE
    tabs = CountryTabs(countries)
    tabs.display()
    return tabs
