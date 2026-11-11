print("\n>>> M1_export_ui initializing (v7.0 Tabs) <<<")
import ee
import traceback
import ipywidgets as widgets

from M0_auth_config import (
    CONFIG, GLOBAL_OPTS, mosaic_name_for_period, is_edit_mode,
    get_sensor_start_year, resolve_years, period_cog_path, period_chunk_path,
    get_asset_mosaic_collection,
)
from M_periods import list_periods, parse_period, period_state_key
import M0_auth_config as config_module
from M_cache import CacheManager
from M_ui_components import PipelineStepUI, cell_log
from M_lang import L as Lang
from M_mosaics import methods_for


def resolve_output_state(row_type, has_chunks, has_cog, has_asset, is_active):
    """Return (state, exists) for one M1 grid cell.

    GCS rows are satisfied by either the exported chunks or the assembled
    COG. ASSET rows only depend on the GEE asset. An active export task wins
    over any existing output.
    """
    if row_type == 'ASSET':
        exists = has_asset
    else:
        exists = bool(has_chunks or has_cog)
    if is_active:
        return 'RUN', exists
    return ('OK' if exists else 'MISS'), exists


def output_link_url(
    row_type, has_chunks, has_cog, has_asset,
    sensor, period, mosaic, period_key, band, mosaic_name,
):
    """Return the clickable target for an OK cell, or None."""
    if row_type == 'ASSET':
        if not has_asset:
            return None
        collection_id = get_asset_mosaic_collection(
            sensor=sensor, period=period, band=band,
            mosaic=mosaic, period_key=period_key,
        )
        return f"https://code.earthengine.google.com/?asset={collection_id}/{mosaic_name}"

    if has_cog:
        cog_dir = period_cog_path(period_key, mosaic=mosaic, sensor=sensor)
        return f"https://storage.googleapis.com/{CONFIG['bucket']}/{cog_dir}/{mosaic_name}_cog.tif"

    if has_chunks:
        chunk_dir = period_chunk_path(period_key, mosaic=mosaic, sensor=sensor)
        return f"https://console.cloud.google.com/storage/browser/{CONFIG['bucket']}/{chunk_dir}"

    return None


def make_link_cell(chk, url, width='auto'):
    """Checkbox + clickable OK label, matching the M2 COG cells."""
    status_html = widgets.HTML(
        f'<a href="{url}" target="_blank" '
        'style="font-size:10px;font-weight:700;color:#155724;text-decoration:underline;">OK ↗</a>',
        layout=widgets.Layout(width='32px'),
    )
    cell = widgets.HBox(
        [chk, status_html],
        layout=widgets.Layout(
            width=width,
            min_height='34px',
            justify_content='center',
            align_items='center',
            padding='0',
            overflow='hidden',
            margin='1px',
        ),
    )
    cell.add_class('mfm-ok')
    return cell


class ExportDispatcherUI(PipelineStepUI):
    def __init__(self, years=None):
        super().__init__(
            title=Lang.M1_TITLE_FULL, 
            description=Lang.M1_DESCRIPTION
        )
        self.chk_dict = {}
        self.requested_years = years
        self.btn_refresh = None
        self.is_refreshing = False
        
        try:
            self.main_area.children = [widgets.HTML(f"<i>{Lang.LOADING_INTERFACE}</i>")]
        except Exception as e:
            print(f"CRITICAL ERROR IN INTERFACE: {e}")
            traceback.print_exc()

    def _init_data(self):
        import datetime
        self.now = datetime.datetime.now()
        self.years = resolve_years(self.requested_years)
        self.bands = CONFIG['bands_all']
        self.update_status(Lang.LOADING_CACHE)
        self.state = CacheManager.load() or {}
        # Chunks/COGs written outside this session live only in GCS. Rebuild
        # the cache when it is older than the TTL so the grid reflects the
        # current GCS state without requiring a manual Sync.
        if self._cache_is_stale():
            try:
                self.state = CacheManager.build_cache_from_gcs(
                    logger=self.update_status, years=self.years,
                )
            except Exception as e:
                print(f"[WARN] M1: could not sync GCS cache: {e}")

    _CACHE_TTL_SECONDS = 300

    def _cache_is_stale(self):
        import datetime
        raw = (self.state or {}).get('updated_at')
        if not raw:
            return True
        try:
            stamp = datetime.datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        except ValueError:
            return True
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=datetime.timezone.utc)
        age = (datetime.datetime.now(datetime.timezone.utc) - stamp).total_seconds()
        return age > self._CACHE_TTL_SECONDS

    def _get_active_tasks(self):
        try:
            tasks = ee.data.getTaskList() 
            active = [t.get('description', '') for t in tasks[:1000] if t.get('state') in ['RUNNING', 'PENDING', 'READY']]
            return active
        except Exception:
            return []

    def _on_select_all(self, _):
        for chk in self.chk_dict.values():
            if not chk.disabled and not chk._meta.get('exists'): chk.value = True

    def _on_select_none(self, _):
        for chk in self.chk_dict.values():
            if not chk.disabled: chk.value = False

    def _on_reexport_toggle(self, change):
        config_module.set_edit_mode(change['new'])
        for chk in self.chk_dict.values():
            chk.disabled = chk._meta.get('active', False) or (
                chk._meta.get('exists', False) and not change['new']
            )

    def _on_select_row(self, btn):
        prefix = f"{btn._sensor}_{btn._period}_{btn._mosaic}_{btn._date}_{btn._type}_"
        row_chks = [chk for key, chk in self.chk_dict.items() if key.startswith(prefix) and not chk.disabled]
        if not row_chks: return
        any_on = any(chk.value for chk in row_chks)
        target_val = not any_on
        for chk in row_chks: chk.value = target_val

    def _build_mosaic_grid(self, sensor, period, mosaic_method, active_tasks):
        L = widgets.Layout
        
        hdr = [
            widgets.HTML(f'<div style="width:100px;">{Lang.DATE}</div>'),
            widgets.HTML(f'<div style="width:60px;">{Lang.COL_TYPE}</div>'),
            widgets.HTML(f'<div style="width:40px; text-align:center; font-weight:bold;">{Lang.BTN_SELECT_ROW}</div>')
        ]
        for b in self.bands:
            h_widget = widgets.HTML(f'<div style="text-align:center; font-weight:bold; font-size:11px;">{b}</div>')
            h_widget.layout.flex = '1'
            hdr.append(h_widget)

        matrix_row_layout = L(
            align_items='center', 
            min_height='35px', 
            border_bottom='1px solid #eee',
            padding='2px 0',
            overflow='visible',
            width='100%'
        )
        
        matrix_rows = [widgets.HBox(hdr, layout=L(
            border_bottom='2px solid #343a40', 
            padding='5px 0',
            min_height='35px',
            overflow='visible'
        ))]
        
        periods = list_periods(
            self.years, period,
            include_current=(period == 'yearly'),
            now=self.now,
        )
        cogs_list = {
            name.lower()
            for name in self.state.get(period_state_key('cogs', period), [])
        }
        for period_info in periods:
            if period_info.year < get_sensor_start_year(sensor):
                continue
            date_str = period_info.key
            y = period_info.year
            m = period_info.index if period_info.periodicity == 'monthly' else None

            for row_type in ['GCS', 'ASSET']:
                cells = [
                    widgets.HTML(f'<div style="width:100px;font-family:monospace;">{date_str if row_type=="GCS" else ""}</div>'),
                    widgets.HTML(f'<div style="width:60px;font-weight:bold;color:#666;">{row_type}</div>'),
                ]
                
                btn_s = widgets.Button(description=Lang.BTN_SELECT_ROW, layout=L(width='40px', height='28px', padding='0'))
                btn_s._sensor, btn_s._period, btn_s._mosaic, btn_s._date, btn_s._type = sensor, period, mosaic_method, date_str, row_type
                btn_s.on_click(self._on_select_row)
                cells.append(btn_s)

                for b in self.bands:
                    m_name = mosaic_name_for_period(date_str, band=b, mosaic=mosaic_method, sensor=sensor)
                    m_base_name = mosaic_name_for_period(date_str, mosaic=mosaic_method, sensor=sensor)

                    chunks_dict = self.state.get('gcs_chunks', {})
                    has_chunks = b.lower() in {
                        chunk.lower() for chunk in chunks_dict.get(m_base_name.lower(), [])
                    }

                    assets_key = period_state_key('assets', period)
                    assets_list = self.state.get(assets_key, [])
                    has_asset = m_name.lower() in {asset.lower() for asset in assets_list}

                    has_cog = m_name.lower() in cogs_list

                    m_name_lower = m_name.lower()
                    # Busca precisa usando a flag pessoal: FLAG_TIPO_NOME
                    task_id = f"{GLOBAL_OPTS['PERSONAL_TASK_FLAG']}_{row_type}_{m_name_lower}".lower()
                    is_active = any(task_id in t.lower() for t in active_tasks)

                    chk = widgets.Checkbox(value=False, indent=False, layout=L(width='18px', height='18px', margin='0'))
                    chk._meta = {'sensor': sensor, 'period': period, 'mosaic': mosaic_method, 'year': y, 'month': m, 'period_key': date_str, 'band': b, 'type': row_type}

                    status, exists = resolve_output_state(
                        row_type, has_chunks, has_cog, has_asset, is_active,
                    )
                    chk._meta['exists'] = exists
                    chk._meta['active'] = is_active
                    css = {'RUN': 'mfm-run', 'OK': 'mfm-ok', 'MISS': 'mfm-null'}[status]

                    chk.disabled = is_active or (exists and not is_edit_mode())

                    link = None
                    if status == 'OK':
                        link = output_link_url(
                            row_type, has_chunks, has_cog, has_asset,
                            sensor=sensor, period=period, mosaic=mosaic_method,
                            period_key=date_str, band=b, mosaic_name=m_name,
                        )
                    if link:
                        cell = make_link_cell(chk, link, width='auto')
                    else:
                        cell = PipelineStepUI.make_status_cell(chk, status, css, width='auto')
                    cell.layout.flex = '1'
                    self.chk_dict[f"{sensor}_{period}_{mosaic_method}_{date_str}_{row_type}_{b}"] = chk
                    cells.append(cell)

                matrix_rows.append(widgets.HBox(cells, layout=matrix_row_layout))
        
        return widgets.VBox(matrix_rows, layout=L(
            max_height='450px', 
            width='100%',
            overflow_y='auto', 
            overflow_x='hidden',
            padding='10px', 
            border='1px solid #ddd',
            background_color='#fff'
        ))

    def _build_ui(self):
        self._init_data()
        self.header_title.value = ""
        self.header_desc.value = ""
        self.header_box.children = [self.header_title]
        
        L = widgets.Layout
        self.chk_dict = {}

        header_html = f'''
        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 5px 10px; background: #fff; border-bottom: 2px solid #333; margin-bottom: 10px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-weight: bold; font-size: 16px; color: #333;">{Lang.M1_HEADER_TITLE}</span>
                <span style="color: #888; font-size: 11px; font-style: italic;">{Lang.M1_HEADER_SUBTITLE}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; padding: 3px 12px; background: #fff1f0; border: 1px solid #ffa39e; border-radius: 4px;">
                <span style="color: #cf1322; font-size: 10px; font-weight: bold; text-transform: uppercase;">{Lang.LABEL_PROJECT}</span>
                <span style="color: #cf1322; font-weight: bold; font-size: 12px;">MapBiomas Fire Monitor</span>
            </div>
        </div>
        '''
        
        self.btn_refresh = widgets.Button(description=Lang.SYNC_DATA, button_style='success', icon='refresh', layout=L(width='180px'))
        self.btn_refresh.on_click(lambda _: self._refresh_cache())
        
        btn_all = widgets.Button(description=Lang.SELECT_PENDING, button_style='info', layout=L(width='180px'))
        btn_none = widgets.Button(description=Lang.CLEAR_SELECTION, button_style='warning', layout=L(width='150px'))
        btn_all.on_click(self._on_select_all)
        btn_none.on_click(self._on_select_none)

        self.chk_reexport = widgets.Checkbox(
            value=is_edit_mode(),
            description='Allow re-export of existing outputs',
            indent=False
        )
        self.chk_reexport.observe(self._on_reexport_toggle, names='value')
        
        footer = widgets.VBox([
            widgets.HBox([btn_all, btn_none, self.chk_reexport, self.btn_refresh, self.loader_html], layout=L(margin='15px 0', gap='10px', align_items='center'))
        ])

        sensors = [s.upper() for s in (GLOBAL_OPTS['SENSOR'] if isinstance(GLOBAL_OPTS['SENSOR'], list) else [GLOBAL_OPTS['SENSOR']])]
        periods = GLOBAL_OPTS['PERIODICITY'] if isinstance(GLOBAL_OPTS['PERIODICITY'], list) else [GLOBAL_OPTS['PERIODICITY']]
        
        self.update_status(Lang.SYNCING_TASKS)
        active_tasks = self._get_active_tasks()
        
        self.sensor_tabs = widgets.Tab()
        self.sensor_children = []
        self.tab_map = {}

        for i, s in enumerate(sensors):
            period_tabs = widgets.Tab()
            period_children = []
            for j, p in enumerate(periods):
                methods = methods_for(CONFIG['mosaic_methods'], sensor=s.lower(), period=p)
                method_tabs = widgets.Tab()
                method_children = []
                for k, m in enumerate(methods):
                    placeholder = widgets.VBox([widgets.HTML(f"<i>{Lang.CLICK_TO_LOAD.format(sensor=s, period=p, mosaic=m)}</i>")], layout=L(padding='20px'))
                    method_children.append(placeholder)
                    self.tab_map[(i, j, k)] = (s.lower(), p, m)
                
                method_tabs.children = method_children
                for k, m in enumerate(methods):
                    method_tabs.set_title(k, m.upper())
                
                method_tabs.observe(lambda change, si=i, pi=j: self._on_method_change(change, si, pi), names='selected_index')
                period_children.append(method_tabs)
            
            period_tabs.children = period_children
            for j, p in enumerate(periods):
                period_tabs.set_title(j, p.capitalize())
            
            period_tabs.observe(lambda change, si=i: self._on_period_change(change, si), names='selected_index')
            self.sensor_children.append(period_tabs)
        
        # Add Guide tab at index 0
        guide_widget = widgets.HTML(Lang.GUIDE_M1_HTML)
        self.sensor_tabs.children = [guide_widget] + self.sensor_children
        self.sensor_tabs.set_title(0, Lang.TAB_GUIDE)
        for i, s in enumerate(sensors):
            self.sensor_tabs.set_title(i + 1, s)

        self.sensor_tabs.observe(self._on_sensor_change, names='selected_index')

        self._load_tab(0, 0, 0, active_tasks)
        self.sensor_tabs.selected_index = 1  # show first sensor, not guide

        if hasattr(self, '_last_sensor_idx'):
            try:
                vis_idx = self._last_sensor_idx
                self.sensor_tabs.selected_index = vis_idx
                if vis_idx > 0:
                    real_idx = vis_idx - 1
                    s_tab = self.sensor_children[real_idx]
                    s_tab.selected_index = self._last_period_idx
                    m_tab = s_tab.children[self._last_period_idx]
                    m_tab.selected_index = self._last_method_idx
                    self._load_tab(real_idx, self._last_period_idx, self._last_method_idx, active_tasks)
            except Exception:
                pass

        self.clear_main()
        self.main_area.children = [PipelineStepUI.get_status_css(), widgets.HTML(header_html), self.sensor_tabs, footer]

    def _on_sensor_change(self, change):
        vis_idx = change['new']
        if vis_idx == 0:  # Guide tab
            return
        real_idx = vis_idx - 1
        p_tabs = self.sensor_children[real_idx]
        p_idx = p_tabs.selected_index
        m_idx = p_tabs.children[p_idx].selected_index
        self._load_tab(real_idx, p_idx, m_idx)

    def _on_period_change(self, change, s_idx):
        p_idx = change['new']
        p_tabs = self.sensor_children[s_idx]
        m_idx = p_tabs.children[p_idx].selected_index
        self._load_tab(s_idx, p_idx, m_idx)

    def _on_method_change(self, change, s_idx, p_idx):
        m_idx = change['new']
        self._load_tab(s_idx, p_idx, m_idx)

    def _load_tab(self, s_idx, p_idx, m_idx, active_tasks=None):
        if active_tasks is None:
            active_tasks = self._get_active_tasks()
            
        sensor, period, method = self.tab_map[(s_idx, p_idx, m_idx)]
        p_tabs = self.sensor_children[s_idx]
        method_tabs = p_tabs.children[p_idx]
        
        target_container = method_tabs.children[m_idx]
        if not isinstance(target_container.children[0], widgets.HTML) or "<i>" not in target_container.children[0].value:
            return

        grid = self._build_mosaic_grid(sensor, period, method, active_tasks)
        
        new_children = list(method_tabs.children)
        new_children[m_idx] = grid
        method_tabs.children = new_children

    def _refresh_cache(self):
        if self.is_refreshing: return
        try:
            self.is_refreshing = True
            if hasattr(self, 'sensor_tabs'):
                self._last_sensor_idx = self.sensor_tabs.selected_index  # visual index
                if self._last_sensor_idx > 0:
                    real_idx = self._last_sensor_idx - 1
                    s_tab = self.sensor_children[real_idx]
                    self._last_period_idx = s_tab.selected_index
                    m_tab = s_tab.children[self._last_period_idx]
                    self._last_method_idx = m_tab.selected_index
                
            self.show_loader(Lang.SYNCING)
            self.state = CacheManager.build_full_cache(logger=self.update_status, years=self.years)
            self._build_ui()
        finally:
            self.is_refreshing = False
            self.hide_loader()

def run_ui(years=None):
    ui = ExportDispatcherUI(years=years)
    ui.display() 
    ui.show_loader(Lang.LOADING)
    ui._build_ui()
    ui.hide_loader()
    return ui

def start_export(ui_obj, mode=None):
    selected = []
    for chk in ui_obj.chk_dict.values():
        if chk.value:
            meta = chk._meta
            if mode is None or meta['type'] == mode:
                selected.append(meta)
    
    if not selected:
        print(Lang.EXPORT_NONE_SEL)
        cell_log(Lang.EXPORT_NONE_SEL, type='warning')
        return

    selected_sensors = {meta['sensor'].lower() for meta in selected}
    try:
        from M_blocklist import refresh_blocklists
        blocklists = refresh_blocklists(selected_sensors)
        loaded = {sensor: len(ids) for sensor, ids in blocklists.items() if ids}
        if loaded:
            summary = ', '.join(f"{sensor}={count}" for sensor, count in sorted(loaded.items()))
            print(f"[BLOCKLIST] Loaded scene exclusions: {summary}.")
            cell_log(f"Blocklists loaded: {summary}", type='info')
        else:
            print("[BLOCKLIST] No lists available; continuing without scene exclusions.")
    except Exception as e:
        message = f"Blocklist refresh unavailable; continuing without scene exclusions: {e}"
        print(f"[WARN] {message}")
        cell_log(message, type='warning')

    print(f" Starting {len(selected)} exports...")
    cell_log(f"Starting {len(selected)} exports...", type='info')

    import M1_export_logic as logic
    from M0_auth_config import CONFIG
    
    country_geom = ee.FeatureCollection(CONFIG['asset_regions']).geometry().bounds()

    succeeded = 0
    for i, meta in enumerate(selected, 1):
        y, m, p = meta['year'], meta['month'], meta['period']
        period_key = meta.get('period_key') or (f"{y}_{m:02d}" if m else str(y))
        info = parse_period(period_key)
        y = info.year
        m = info.index if info.periodicity == 'monthly' else None
        p = info.periodicity
        band, mosaic_m, sensor = meta['band'], meta['mosaic'], meta['sensor']

        name_base = mosaic_name_for_period(period_key, mosaic=mosaic_m, sensor=sensor, band=None)
        name_full = mosaic_name_for_period(period_key, band=band, mosaic=mosaic_m, sensor=sensor)

        try:
            print(f"  [{i}/{len(selected)}] Sending {sensor} {mosaic_m} ({band}) {period_key} for {meta['type']}...")

            start_date = info.start.strftime('%Y-%m-%d')
            end_date = info.end.strftime('%Y-%m-%d')

            mosaic_obj = logic.get_quality_mosaic(sensor, y, start_date, end_date, country_geom, month=m, method=mosaic_m)

            if meta['type'] == 'ASSET':
                logic.export_to_asset(
                    mosaic_obj, name_full, y, m, p, band=band, mosaic=mosaic_m,
                    sensor=sensor, period_key=period_key,
                )
            else:
                if meta.get('exists'):
                    logic.delete_gcs_band(
                        y, m, p, band, sensor=sensor, mosaic=mosaic_m,
                        period_key=period_key,
                    )
                logic.export_to_gcs(
                    mosaic_obj, name_base, y, m, p, bands=[band], mosaic=mosaic_m,
                    sensor=sensor, period_key=period_key,
                )
            succeeded += 1
            cell_log(f"Export sent: {sensor} {mosaic_m} ({band}) {period_key} [{meta['type']}]", type='success')
        except Exception as e:
            print(f"  [ERR] Failed {sensor} {mosaic_m} ({band}) {period_key}: {e}")
            cell_log(f"Failed {sensor} {mosaic_m} ({band}) {period_key}: {e}", type='error')
            traceback.print_exc()
            
    print(f"\n Completed! {succeeded}/{len(selected)} tasks sent.")
    cell_log(f"Completed! {succeeded}/{len(selected)} tasks sent.", type='success' if succeeded == len(selected) else 'warning')

    if succeeded:
        # New files are still being written asynchronously; drop the cache stamp
        # so M2 rescans GCS when it opens instead of trusting the old state.
        CacheManager.mark_stale()
