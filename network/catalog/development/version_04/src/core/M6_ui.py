import os
import csv
import io
import time
import base64
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from M0_auth_config import CONFIG, _get_fs, gcs_classifications_path
from M5_workplan import gcs_full, consolidated_stats_path, region_path
from M6_publisher import discover_classified_groups, gee_asset_exists, stats_row_exists, load_gee_assets, load_stats_done
from M_cache import CacheManager
from M_ui_components import THEME, make_empty_state, flash_output, make_select_all_none, make_refresh_button, make_sync_button
from M_lang import L as Lang

L = widgets.Layout

# Cache módulo: evita re-scan GEE/GCS entre refreshes no mesmo kernel
_M6_DISCOVERY_CACHE = None
_M6_UI_INSTANCE = None


def invalidate_discovery_cache():
    """Force the next refresh to rescan GCS for classified groups."""
    global _M6_DISCOVERY_CACHE
    _M6_DISCOVERY_CACHE = None


class M6WorkplanUI:
    def __init__(self):
        global _M6_UI_INSTANCE
        _M6_UI_INSTANCE = self
        self.fs = _get_fs()
        self._groups = []
        self._mosaics = set()
        self._gee_assets = set()
        self._stats_done = set()
        self._stats_data = []
        self.lc_base = gcs_classifications_path()
        self._thumbnails = {}
        self._publish_checks = {}

        self.tabs = widgets.Tab()
        self.w_guide = widgets.HTML()
        self.tab_to_publish = widgets.VBox()
        self.tab_finished = widgets.VBox()
        self.tab_analytics = widgets.VBox()
        self.tab_coverage = widgets.VBox()

        self.btn_sync, _ = make_sync_button(Lang.SYNC_DATA, "refresh", lambda: self._refresh_all(force=True), width='180px', button_style='success')

        self._build_guide()

    def _build_guide(self):
        self.w_guide.value = Lang.GUIDE_M6_HTML.format(
            tab_publish=Lang.TAB_PUBLISH,
            tab_done=Lang.TAB_DONE,
            tab_analytics=Lang.TAB_ANALYTICS,
            tab_coverage=Lang.TAB_M6_COVERAGE,
        )

    def _discover_all(self, force=False):
        global _M6_DISCOVERY_CACHE
        if _M6_DISCOVERY_CACHE is not None and not force:
            ch = _M6_DISCOVERY_CACHE
            self._groups = ch['groups']
            self._mosaics = ch['mosaics']
            self._gee_assets = ch['gee_assets']
            self._stats_done = ch['stats_done']
            self._thumbnails = ch['thumbnails']
            return

        raw = discover_classified_groups(fs=self.fs)
        self._groups = sorted(raw)
        self._mosaics = set()
        self._gee_assets = set()
        self._stats_done = set()

        # 1. Mosaic: GCS exists check (barato)
        for group in self._groups:
            m, r, p, c = group
            if self.fs.exists(gcs_full(region_path(m, r, p, c))):
                self._mosaics.add(group)

        # 2. GEE: 1 chamada listAssets por (modelo, campanha)
        by_model_camp = {}
        for g in self._groups:
            by_model_camp.setdefault((g[0], g[3]), []).append(g)
        for (model_id, campaign), model_groups in by_model_camp.items():
            assets = load_gee_assets(model_id, campaign=campaign)
            for m, r, p, c in model_groups:
                if f"{r}_{p}" in assets:
                    self._gee_assets.add((m, r, p, c))

        # 3. Stats: 1 download CSV por campanha
        stats_done = load_stats_done(self._groups, self.fs)
        for g in self._groups:
            if g in stats_done:
                self._stats_done.add(g)

        # 4. Thumbnails
        self._load_thumbnails()

        # Salva cache
        _M6_DISCOVERY_CACHE = {
            'groups': list(self._groups),
            'mosaics': set(self._mosaics),
            'gee_assets': set(self._gee_assets),
            'stats_done': set(self._stats_done),
            'thumbnails': dict(self._thumbnails),
        }

    def _load_stats(self):
        self._stats_data = []
        lib_class = gcs_full(self.lc_base)
        patterns = [
            f"{lib_class}/*/consolidated_stats.csv",
            f"{lib_class}/consolidated_stats.csv",
        ]
        from M0_auth_config import _gcs_download
        for pat in patterns:
            for fp in sorted(self.fs.glob(pat)):
                try:
                    local_csv = os.path.join('/tmp', 'm6_consolidated.csv')
                    _gcs_download(fp, local_csv)
                    with open(local_csv, 'r') as f:
                        self._stats_data.extend(list(csv.DictReader(f)))
                    os.remove(local_csv)
                except Exception:
                    continue

    def _load_thumbnails(self):
        from M6_publisher import generate_region_thumbnail
        for r in sorted(set(g[1] for g in self._groups)):
            if r not in self._thumbnails:
                b64 = generate_region_thumbnail(r, size=128)
                if b64:
                    self._thumbnails[r] = b64

    def _refresh_all(self, force=False):
        if force:
            CacheManager.build_cache_from_classifications()
        self._discover_all(force=force)
        self._load_stats()
        self._render_to_publish()
        self._render_finished()
        self._render_analytics()
        self._render_coverage()

    def _render_to_publish(self):
        from M_ui_components import make_card_body, build_thumbnail_column
        pending = [g for g in self._groups if not (g in self._mosaics and g in self._gee_assets and g in self._stats_done)]
        if not pending:
            self.tab_to_publish.children = [make_empty_state(Lang.NO_TASKS_PUBLISH)]
            return

        models = sorted(set(g[0] for g in self._groups))
        regions = sorted(set(g[1] for g in self._groups))
        periods = sorted(set(g[2] for g in self._groups), reverse=True)
        campaigns = sorted(set(g[3] for g in self._groups if g[3]))

        f_model = widgets.Dropdown(options=[Lang.ALL] + models, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_MODEL, layout=L(width='180px'))
        f_region = widgets.Dropdown(options=[Lang.ALL_F] + regions, value=Lang.ALL_F, description=Lang.ANALYTICS_FILTER_REGION, layout=L(width='180px'))
        f_period = widgets.Dropdown(options=[Lang.ALL] + periods, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_PERIOD, layout=L(width='180px'))
        f_campaign = widgets.Dropdown(options=[Lang.ALL] + campaigns, value=Lang.ALL, description='Campaign', layout=L(width='160px'))
        filters_hbox = widgets.HBox([f_model, f_region, f_period, f_campaign])

        cards_container = widgets.VBox()

        def _render_cards(filtered=None):
            if filtered is None:
                fm, fr, fp, fc = f_model.value, f_region.value, f_period.value, f_campaign.value
                filtered = [g for g in pending
                            if (fm == Lang.ALL or g[0] == fm)
                            and (fr == Lang.ALL_F or g[1] == fr)
                            and (fp == Lang.ALL or g[2] == fp)
                            and (fc == Lang.ALL or g[3] == fc)]
            self._publish_checks.clear()
            cards = []
            for g in sorted(filtered):
                m, r, p, c = g
                thumb_b64 = self._thumbnails.get(r, '')
                left_col = build_thumbnail_column(thumb_b64)
                cb = widgets.Checkbox(description='', indent=False, layout=L(width='24px', margin='2px 6px 0 0'))
                self._publish_checks[g] = cb
                c_label = f"<span style='color:#7f8c8d;font-size:12px;'>[{c}]</span>" if c else ""
                def _badge(ok, label):
                    color = '#22c55e' if ok else '#ef4444'
                    return f"<span style='display:inline-block;padding:0 6px;border-radius:8px;font-size:10px;font-weight:600;color:white;background:{color};margin:0 2px;'>{label}</span>"
                badges = widgets.HTML(
                    _badge(g in self._mosaics, Lang.M6_BADGE_MOSAIC) +
                    _badge(g in self._stats_done, Lang.M6_BADGE_STATS) +
                    _badge(g in self._gee_assets, Lang.M6_BADGE_GEE),
                    layout=L(margin='0 0 0 30px'))
                right_col = widgets.VBox([
                    widgets.HBox([cb, widgets.HTML(f"<b style='font-size:15px;color:#0f172a;'>{m}</b> {c_label}")], layout=L(align_items='center')),
                    widgets.HTML(f"<span style='color:#475569;margin-left:30px;'>{Lang.DROP_REGION} <b>{r}</b></span>"),
                    widgets.HTML(f"<span style='color:#475569;margin-left:30px;'>{Lang.M6_LABEL_PERIOD} <b>{p}</b></span>"),
                    badges,
                ], layout=L(flex='1', margin='0 0 0 12px'))
                cards.append(make_card_body(left_col, right_col))
            cards_container.children = cards or [make_empty_state(Lang.M6_NO_MATCHING)]

        for f in [f_model, f_region, f_period, f_campaign]:
            f.observe(lambda _: _render_cards(), names='value')

        _render_cards(filtered=pending)

        btn_all_pub, btn_none_pub, hbox_pub = make_select_all_none(
            on_all=lambda _: self._toggle_publish(True),
            on_none=lambda _: self._toggle_publish(False))
        self.tab_to_publish.children = [
            widgets.HTML(f"<b>{Lang.M6_GROUPS_PENDING.format(n=len(pending))}</b>"),
            filters_hbox,
            hbox_pub,
            cards_container,
        ]

    def _toggle_publish(self, value):
        for cb in self._publish_checks.values():
            cb.value = value

    def _render_finished(self):
        done = sorted([g for g in self._groups if g in self._mosaics and g in self._gee_assets and g in self._stats_done])
        if not done:
            self.tab_finished.children = [make_empty_state(Lang.NO_TASKS_DONE)]
            return

        models = sorted(set(g[0] for g in self._groups))
        regions = sorted(set(g[1] for g in self._groups))
        periods = sorted(set(g[2] for g in self._groups), reverse=True)
        campaigns = sorted(set(g[3] for g in self._groups if g[3]))

        f_model = widgets.Dropdown(options=[Lang.ALL] + models, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_MODEL, layout=L(width='180px'))
        f_region = widgets.Dropdown(options=[Lang.ALL_F] + regions, value=Lang.ALL_F, description=Lang.ANALYTICS_FILTER_REGION, layout=L(width='180px'))
        f_period = widgets.Dropdown(options=[Lang.ALL] + periods, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_PERIOD, layout=L(width='180px'))
        f_campaign = widgets.Dropdown(options=[Lang.ALL] + campaigns, value=Lang.ALL, description='Campaign', layout=L(width='160px'))
        filters_hbox = widgets.HBox([f_model, f_region, f_period, f_campaign])

        rows_container = widgets.VBox()

        def _render_rows(filtered=None):
            if filtered is None:
                fm, fr, fp, fc = f_model.value, f_region.value, f_period.value, f_campaign.value
                filtered = [g for g in done
                            if (fm == Lang.ALL or g[0] == fm)
                            and (fr == Lang.ALL_F or g[1] == fr)
                            and (fp == Lang.ALL or g[2] == fp)
                            and (fc == Lang.ALL or g[3] == fc)]
            rows = []
            for m, r, p, c in filtered:
                c_label = f"<span style='color:#7f8c8d;'> [{c}]</span>" if c else ""
                mosaic_path = gcs_full(region_path(m, r, p, c))
                mosaic_url = f"https://storage.googleapis.com/{mosaic_path}"
                rows.append(widgets.HBox([
                    widgets.HTML(f"<b>{m}</b>{c_label}", layout=L(width='220px')),
                    widgets.HTML(r, layout=L(width='150px')),
                    widgets.HTML(p, layout=L(width='120px')),
                    widgets.HTML(f"<a href='{mosaic_url}' target='_blank' style='color:{THEME['SUCCESS']};font-weight:700;font-size:12px;text-decoration:underline;'>{Lang.M6_DOWNLOAD_MOSAIC}</a>"),
                ], layout=L(margin='2px 0', padding='4px', border='1px solid #eee')))
            rows_container.children = rows or [make_empty_state(Lang.M6_NO_MATCHING)]

        for f in [f_model, f_region, f_period, f_campaign]:
            f.observe(lambda _: _render_rows(), names='value')

        _render_rows(filtered=done)

        self.tab_finished.children = [
            widgets.HTML(f"<b>{Lang.M6_PUBLISHED_GROUPS.format(n=len(done))}</b>"),
            filters_hbox,
            rows_container,
        ]

    def _render_analytics(self):
        if not self._stats_data:
            self.tab_analytics.children = [make_empty_state(Lang.M6_NO_STATS)]
            return

        models = sorted(set(r['model_id'] for r in self._stats_data))
        regions = sorted(set(r['region'] for r in self._stats_data))
        periods = sorted(set(r['period'] for r in self._stats_data), reverse=True)

        f_model = widgets.Dropdown(options=[Lang.ALL] + models, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_MODEL, layout=L(width='200px'))
        f_region = widgets.Dropdown(options=[Lang.ALL_F] + regions, value=Lang.ALL_F, description=Lang.ANALYTICS_FILTER_REGION, layout=L(width='200px'))
        f_period = widgets.Dropdown(options=[Lang.ALL] + periods, value=Lang.ALL, description=Lang.ANALYTICS_FILTER_PERIOD, layout=L(width='200px'))
        filters = widgets.HBox([f_model, f_region, f_period])

        w_table = widgets.HTML()
        btn_download = widgets.Button(description=Lang.DOWNLOAD_TABLE, icon='download', layout=L(width='180px'))
        w_download = widgets.HTML()

        def _update_table(_=None):
            data = self._stats_data
            if f_model.value != Lang.ALL:
                data = [r for r in data if r['model_id'] == f_model.value]
            if f_region.value != Lang.ALL_F:
                data = [r for r in data if r['region'] == f_region.value]
            if f_period.value != Lang.ALL:
                data = [r for r in data if r['period'] == f_period.value]

            if not data:
                w_table.value = f"<p style='color:gray;'>{Lang.M6_NO_MATCHING}</p>"
                return

            h = '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
            h += '<tr style="background:#2c3e50;color:white;">'
            for col in [Lang.MODEL, Lang.REGION, Lang.M6_LABEL_PERIOD.rstrip(':'), Lang.M6_COL_HA, Lang.M6_COL_PCT, Lang.CONFIDENCE, Lang.M6_COL_TILES]:
                h += f'<th style="padding:6px 10px;text-align:left;">{col}</th>'
            h += '</tr>'
            for row in data:
                h += '<tr style="border-bottom:1px solid #eee;">'
                h += f'<td style="padding:4px 10px;">{row["model_id"]}</td>'
                h += f'<td style="padding:4px 10px;">{row["region"]}</td>'
                h += f'<td style="padding:4px 10px;">{row["period"]}</td>'
                h += f'<td style="padding:4px 10px;">{row.get("burned_area_ha", "0")}</td>'
                h += f'<td style="padding:4px 10px;">{row.get("burned_percentage", "0")}%</td>'
                h += f'<td style="padding:4px 10px;">{row.get("mean_confidence", "0")}</td>'
                h += f'<td style="padding:4px 10px;">{row.get("tiles_total", "0")}</td>'
                h += '</tr>'
            h += '</table>'
            h += f'<p style="color:gray;font-size:12px;">{Lang.M6_N_RECORDS.format(n=len(data))}</p>'
            w_table.value = h
            btn_download._data = data

        def _on_download(_):
            data = getattr(btn_download, '_data', [])
            if not data:
                return
            now = time.strftime('%Y-%m-%d')
            fname = f"{CONFIG.get('country', 'fire')}_fire_stats_{f_model.value}_{f_region.value}_{f_period.value}_{now}.csv"
            fname = fname.replace(' ', '_').replace('/', '-')

            output = io.StringIO()
            fieldnames = ['model_id', 'region', 'period', 'burned_area_ha', 'burned_percentage', 'mean_confidence', 'tiles_total', 'total_pixels', 'burned_pixels']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow({k: row.get(k, '') for k in fieldnames})
            csv_bytes = output.getvalue().encode('utf-8')
            b64 = base64.b64encode(csv_bytes).decode('utf-8')
            w_download.value = f'<a href="data:text/csv;base64,{b64}" download="{fname}">{Lang.M6_DOWNLOAD_LINK.format(fname=fname)}</a>'

        for f in [f_model, f_region, f_period]:
            f.observe(_update_table, names='value')
        btn_download.on_click(_on_download)

        _update_table()
        self.tab_analytics.children = [filters, w_table, btn_download, w_download]

    def _render_coverage(self):
        all_rows = sorted(set((g[0], g[3]) for g in self._groups))
        all_regions = sorted(set(g[1] for g in self._groups))

        if not all_rows:
            self.tab_coverage.children = [make_empty_state(Lang.M6_NO_CLASSIFIED_GROUPS)]
            return

        lines = ["<table style='border-collapse:collapse;font-size:13px;'>"]
        header = "<tr><th style='padding:6px 10px;text-align:left;'>" + Lang.MODEL + "</th>"
        for r in all_regions:
            header += f"<th style='padding:6px 10px;'>{r}</th>"
        header += "</tr>"
        lines.append(header)

        for m, c in all_rows:
            label = f"{m} [{c}]" if c else m
            line = f"<tr style='border-bottom:1px solid #eee;'><td style='padding:4px 10px;'><b>{label}</b></td>"
            for r in all_regions:
                periods = sorted([p for _m, _r, p, _c in self._groups if _m == m and _r == r and _c == c])
                periods_done = sorted([p for _m, _r, p, _c in self._mosaics if _m == m and _r == r and _c == c])
                if not periods:
                    cell = "<span style='color:#ccc;'>-</span>"
                else:
                    n = len(periods)
                    n_done = len(periods_done)
                    if n_done == n:
                        cell = f"<span style='color:{THEME['SUCCESS']};'>{n}/{n}</span>"
                    elif n_done > 0:
                        cell = f"<span style='color:{THEME['WARNING']};'>{n_done}/{n}</span>"
                    else:
                        cell = f"<span style='color:{THEME['INFO']};'>0/{n}</span>"
                line += f"<td style='padding:4px 10px;text-align:center;'>{cell}</td>"
            line += "</tr>"
            lines.append(line)

        lines.append("</table>")
        legend = widgets.HTML(f"""
            <div style='margin-top:15px;font-size:12px;'>
                <span style='color:{THEME['SUCCESS']};'>\u25cf {Lang.M6_LEGEND_PUBLISHED}</span> &nbsp;
                <span style='color:{THEME['WARNING']};'>\u25cf {Lang.M6_LEGEND_PARTIAL}</span> &nbsp;
                <span style='color:{THEME['INFO']};'>\u25cf {Lang.M6_LEGEND_CLASSIFIED_ONLY}</span> &nbsp;
                <span style='color:#ccc;'>\u25cf {Lang.M6_LEGEND_NO_DATA}</span>
            </div>
        """)
        self.tab_coverage.children = [widgets.HTML("".join(lines)), legend]

    def display(self):
        self._build_guide()
        self._refresh_all(force=True)

        self.tabs.children = [
            self.w_guide, self.tab_to_publish, self.tab_finished,
            self.tab_analytics, self.tab_coverage
        ]
        self.tabs.set_title(0, Lang.TAB_GUIDE)
        self.tabs.set_title(1, Lang.TAB_PUBLISH)
        self.tabs.set_title(2, Lang.TAB_DONE)
        self.tabs.set_title(3, Lang.TAB_ANALYTICS)
        self.tabs.set_title(4, Lang.TAB_M6_COVERAGE)

        header = widgets.HBox([
            widgets.HTML(f"<b style='color:#2c3e50; font-size:14px;'>{Lang.M6_HEADER_TITLE}</b>"),
            self.btn_sync
        ], layout=L(margin='0 0 15px 0', align_items='center', padding='10px',
                    border='1px solid #e0e0e0', background='#fcfcfc'))

        display(widgets.VBox([header, self.tabs]))

    def refresh(self):
        self._refresh_all()


def run_m6_ui():
    ui = M6WorkplanUI()
    ui.display()
    return ui
