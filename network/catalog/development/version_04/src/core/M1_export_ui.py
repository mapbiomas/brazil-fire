print("\n>>> M1_export_ui initializing (v7.0 Tabs) <<<")
import ee
import traceback
import html
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
from M1_update_manifest import (
    add_pending_item, find_item, load_manifest, mark_item_complete,
    pending_item, save_manifest, set_item_status,
)


def resolve_output_state(row_type, has_chunks, has_cog, has_asset, is_active,
                         update_pending=False):
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
    if exists and update_pending:
        return 'UPDATE', True
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


def make_update_cell(chk, callback, width='auto'):
    """Render a direct Update action for an old country-wide output."""
    button = widgets.Button(
        description=Lang.M1_UPDATE_LABEL, icon='refresh', button_style='info',
        tooltip=Lang.M1_UPDATE_TOOLTIP,
        layout=widgets.Layout(width='76px', height='26px', padding='0'),
    )
    button.on_click(lambda _button: callback(chk))
    chk._update_button = button
    cell = widgets.HBox(
        [chk, button],
        layout=widgets.Layout(
            width=width, min_height='34px', justify_content='center',
            align_items='center', padding='0', overflow='hidden', margin='1px',
        ),
    )
    cell.add_class('mfm-update')
    return cell


def make_update_help():
    """Explain why existing outputs need an update and the GCS/M2 follow-up."""
    return widgets.HTML(
        f'<span style="font-size:11px;color:#666;">{Lang.M1_UPDATE_HELP}</span>',
        layout=widgets.Layout(flex='1 1 320px', min_width='260px'),
    )


def make_update_guide():
    """Build the localized M1 update explainer tab."""
    sections = (
        (Lang.M1_UPDATE_GUIDE_NORMAL_TITLE, Lang.M1_UPDATE_GUIDE_NORMAL),
        (Lang.M1_UPDATE_GUIDE_REASON_TITLE, Lang.M1_UPDATE_GUIDE_REASON),
        (Lang.M1_UPDATE_GUIDE_ACTION_TITLE, Lang.M1_UPDATE_GUIDE_ACTION),
    )
    cards = ''.join(
        '<section style="background:#fff;border:1px solid #d6e4ff;'
        'border-left:4px solid #1677ff;border-radius:8px;padding:14px 16px;'
        'margin:12px 0;">'
        f'<h3 style="margin:0 0 7px;color:#0958d9;font-size:15px;">{html.escape(title)}</h3>'
        f'<p style="margin:0;color:#334155;line-height:1.55;">{html.escape(body)}</p>'
        '</section>'
        for title, body in sections
    )
    return widgets.HTML(
        '<div style="max-width:900px;margin:10px auto;padding:12px 20px;'
        'background:#f5f9ff;border:1px solid #d6e4ff;border-radius:12px;">'
        f'<h2 style="margin:0;color:#003eb3;font-size:20px;">'
        f'{html.escape(Lang.M1_UPDATE_GUIDE_TITLE)}</h2>{cards}</div>',
        layout=widgets.Layout(width='100%', padding='8px'),
    )


def make_primary_tabs(guide_widget, update_widget, sensor_children, sensors):
    """Create M1's top-level tabs, keeping the update guide before sensors."""
    tabs = widgets.Tab()
    tabs.children = [guide_widget, update_widget] + list(sensor_children)
    tabs.add_class('m1-primary-tabs')
    tabs.set_title(0, Lang.TAB_GUIDE)
    tabs.set_title(1, Lang.M1_UPDATE_TAB)
    for index, sensor in enumerate(sensors, start=2):
        tabs.set_title(index, sensor)
    return tabs


def update_existing_gcs_output(meta, manifest):
    """Update existing GCS data without scheduling a new M1 export.

    Returns False only when neither the target COG nor its source chunks exist;
    callers must then use the ordinary M1 export path for a genuinely missing
    product. Errors while processing an existing product are propagated.
    """
    from M_gcs import blob_exists, glob

    period_key = meta.get('period_key') or (
        f"{meta['year']}_{meta['month']:02d}" if meta.get('month') else str(meta['year'])
    )
    band = meta['band']
    sensor = meta['sensor']
    mosaic = meta['mosaic']
    name = mosaic_name_for_period(
        period_key, band=band, mosaic=mosaic, sensor=sensor,
    )
    cog_dir = period_cog_path(period_key, mosaic=mosaic, sensor=sensor)
    cog_name = f"{name}_cog.tif"
    cog_uri = f"gs://{CONFIG['bucket']}/{cog_dir}/{cog_name}"
    cog_exists = blob_exists(cog_uri)

    if cog_exists:
        if find_item(manifest, meta) is None:
            add_pending_item(manifest, meta)
        set_item_status(manifest, meta, 'masking_cog')
        save_manifest(manifest)
        try:
            from M2_mosaic_logic import mask_existing_country_cog
            updated = mask_existing_country_cog(
                period_key, band, sensor, mosaic,
                logger=lambda message, _level='info': print(message),
            )
            if not updated:
                raise RuntimeError(f"Existing COG disappeared before masking: {cog_uri}")
            mark_item_complete(manifest, meta)
            save_manifest(manifest)
            return True
        except Exception as e:
            set_item_status(manifest, meta, 'pending', last_error=str(e))
            save_manifest(manifest)
            raise

    chunks_dir = period_chunk_path(period_key, mosaic=mosaic, sensor=sensor)
    chunk_prefix = f"gs://{CONFIG['bucket']}/{chunks_dir}/{name}_*.tif"
    existing_chunks = glob(chunk_prefix)
    if not existing_chunks:
        return False

    if find_item(manifest, meta) is None:
        add_pending_item(manifest, meta)
    # Existing chunks but no COG: M2 assembles those chunks directly.
    set_item_status(manifest, meta, 'chunks_ready')
    save_manifest(manifest)
    from M2_mosaic_logic import assemble_country_mosaic
    result = assemble_country_mosaic(
        year=meta['year'], month=meta.get('month'), period=meta['period'],
        sensor=sensor, mosaic_method=mosaic, bands=[band],
        period_key=period_key,
        logger=lambda message, _level='info': print(message),
    )
    if not result:
        set_item_status(manifest, meta, 'pending', last_error='M2 did not produce the COG.')
        save_manifest(manifest)
        raise RuntimeError(f"M2 could not assemble existing chunks for {period_key}/{band}.")
    mark_item_complete(manifest, meta)
    save_manifest(manifest)
    return True


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
        try:
            self.update_manifest = load_manifest()
            self._reconcile_update_tasks()
        except Exception as e:
            print(f"[WARN] M1 update manifest unavailable: {e}")
            self.update_manifest = {'version': 1, 'items': []}
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
            if not chk.disabled and (
                not chk._meta.get('exists') or chk._meta.get('update_pending')
            ):
                chk.value = True

    def _on_select_none(self, _):
        for chk in self.chk_dict.values():
            if not chk.disabled: chk.value = False

    def _on_reexport_toggle(self, change):
        config_module.set_edit_mode(change['new'])
        for chk in self.chk_dict.values():
            chk.disabled = chk._meta.get('active', False) or (
                chk._meta.get('exists', False)
                and not chk._meta.get('update_pending', False)
                and not change['new']
            )

    def _on_select_row(self, btn):
        prefix = f"{btn._sensor}_{btn._period}_{btn._mosaic}_{btn._date}_{btn._type}_"
        row_chks = [chk for key, chk in self.chk_dict.items() if key.startswith(prefix) and not chk.disabled]
        if not row_chks: return
        any_on = any(chk.value for chk in row_chks)
        target_val = not any_on
        for chk in row_chks: chk.value = target_val

    def _on_update_click(self, chk):
        """Run exactly the selected M1 date/band/target update."""
        try:
            self._reconcile_update_tasks()
            manifest_entry = find_item(self.update_manifest, chk._meta)
            if manifest_entry and manifest_entry.get('status') in ('chunks_ready', 'complete'):
                self._refresh_cache()
                return
        except Exception as e:
            print(f"[WARN] Could not reconcile before M1 update: {e}")
        for candidate in self.chk_dict.values():
            candidate.value = False
        chk.value = True
        chk.disabled = True
        if hasattr(chk, '_update_button'):
            chk._update_button.disabled = True
        if chk._meta['type'] == 'GCS':
            try:
                if update_existing_gcs_output(chk._meta, self.update_manifest):
                    CacheManager.mark_stale()
                    self._refresh_cache()
                    return
                # The cache or manifest may have been stale: if neither COG nor
                # shards exist now, treat this as a new product and export M1.
                chk._meta['update_pending'] = False
                chk._meta['exists'] = False
            except Exception as e:
                print(f"[ERR] Existing GCS output update failed; no new M1 task started: {e}")
                cell_log(f"Update failed: {e}", type='error')
                chk.disabled = False
                if hasattr(chk, '_update_button'):
                    chk._update_button.disabled = False
                return
        submitted = start_export(self, mode=chk._meta['type'])
        if not submitted:
            chk.disabled = False
            if hasattr(chk, '_update_button'):
                chk._update_button.disabled = False

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
                    chk._meta = {
                        'country': CONFIG['country'], 'sensor': sensor,
                        'period': period, 'mosaic': mosaic_method,
                        'year': y, 'month': m, 'period_key': date_str,
                        'band': b, 'type': row_type,
                    }

                    candidate = dict(chk._meta)
                    candidate['exists'] = bool(has_asset if row_type == 'ASSET' else (has_chunks or has_cog))
                    if candidate['exists'] and not is_active:
                        add_pending_item(self.update_manifest, candidate)
                    update_entry = find_item(self.update_manifest, candidate)
                    chunks_ready = bool(
                        update_entry and update_entry.get('status') == 'chunks_ready'
                    )
                    is_update_pending = candidate['exists'] and pending_item(
                        self.update_manifest, candidate,
                    )

                    status, exists = resolve_output_state(
                        row_type, has_chunks, has_cog, has_asset, is_active,
                        update_pending=is_update_pending,
                    )
                    chk._meta['exists'] = exists
                    chk._meta['active'] = is_active
                    chk._meta['update_pending'] = is_update_pending
                    if chunks_ready and status == 'UPDATE':
                        status = 'M2_READY'
                    css = {
                        'RUN': 'mfm-run', 'UPDATE': 'mfm-run',
                        'M2_READY': 'mfm-run', 'OK': 'mfm-ok', 'MISS': 'mfm-null',
                    }[status]

                    chk.disabled = is_active or (
                        exists and not is_edit_mode() and not is_update_pending
                    ) or chunks_ready

                    link = None
                    if status == 'OK':
                        link = output_link_url(
                            row_type, has_chunks, has_cog, has_asset,
                            sensor=sensor, period=period, mosaic=mosaic_method,
                            period_key=date_str, band=b, mosaic_name=m_name,
                        )
                    if status == 'UPDATE':
                        cell = make_update_cell(
                            chk, self._on_update_click, width='auto',
                        )
                    elif status == 'M2_READY':
                        cell = PipelineStepUI.make_status_cell(
                            chk, Lang.M1_M2_READY, css, width='auto',
                        )
                    elif link:
                        cell = make_link_cell(chk, link, width='auto')
                    else:
                        cell = PipelineStepUI.make_status_cell(chk, status, css, width='auto')
                    cell.layout.flex = '1'
                    self.chk_dict[f"{sensor}_{period}_{mosaic_method}_{date_str}_{row_type}_{b}"] = chk
                    cells.append(cell)

                matrix_rows.append(widgets.HBox(cells, layout=matrix_row_layout))
        
        try:
            save_manifest(self.update_manifest)
        except Exception as e:
            print(f"[WARN] Could not persist M1 update manifest: {e}")

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
            widgets.HBox([
                btn_all, btn_none, self.chk_reexport, self.btn_refresh,
                make_update_help(), self.loader_html,
            ], layout=L(margin='15px 0', gap='10px', align_items='center', flex_flow='row wrap'))
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
        
        # Keep static guides separate from lazy-loaded sensor/date grids.
        guide_widget = widgets.HTML(Lang.GUIDE_M1_HTML)
        self.sensor_tabs = make_primary_tabs(
            guide_widget, make_update_guide(), self.sensor_children, sensors,
        )

        self.sensor_tabs.observe(self._on_sensor_change, names='selected_index')

        self._load_tab(0, 0, 0, active_tasks)
        self.sensor_tabs.selected_index = 2  # show first sensor, not a guide

        if hasattr(self, '_last_sensor_idx'):
            try:
                vis_idx = self._last_sensor_idx
                self.sensor_tabs.selected_index = vis_idx
                if vis_idx > 0:
                    real_idx = vis_idx - 2
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
        if vis_idx < 2:  # static guide tabs
            return
        real_idx = vis_idx - 2
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
                if self._last_sensor_idx > 1:
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

    def _reconcile_update_tasks(self):
        """Promote successful staged GCS shards; complete successful EE assets."""
        entries = self.update_manifest.get('items', [])
        changed = False
        for entry in entries:
            if entry.get('status') not in ('submitted', 'exporting', 'submitted_new'):
                continue
            task_ids = entry.get('task_ids') or [entry.get('task_id')]
            task_ids = [task_id for task_id in task_ids if task_id]
            if not task_ids:
                continue
            try:
                states = ee.data.getTaskStatus(*task_ids)
                states = [states] if isinstance(states, dict) else states
                by_id = {str(task.get('id')): task.get('state') for task in states}
                task_states = [by_id.get(str(task_id), 'UNKNOWN') for task_id in task_ids]
            except Exception as e:
                print(f"[WARN] Could not check M1 task status: {e}")
                continue

            if any(state in ('FAILED', 'CANCELLED') for state in task_states):
                if entry.get('type') == 'GCS' and entry.get('staging_prefix'):
                    try:
                        from M_gcs import glob, rm_files
                        rm_files(glob(entry['staging_prefix'] + '*'))
                    except Exception as e:
                        print(f"[WARN] Could not clean failed M1 staging objects: {e}")
                entry['status'] = 'pending'
                entry['last_error'] = f"M1 task state(s): {task_states}"
                entry.pop('task_ids', None)
                entry.pop('staging_prefix', None)
                entry.pop('target_prefix', None)
                changed = True
                continue
            if not all(state == 'COMPLETED' for state in task_states):
                continue

            if entry.get('type') == 'ASSET':
                mark_item_complete(self.update_manifest, entry)
                changed = True
                continue

            if entry.get('status') == 'submitted_new':
                mark_item_complete(self.update_manifest, entry)
                changed = True
                continue

            from M_gcs import copy, glob, rm_files
            stage_prefix = entry.get('staging_prefix')
            target_prefix = entry.get('target_prefix')
            staged = glob(stage_prefix + '*') if stage_prefix and target_prefix else []
            if not staged:
                entry['status'] = 'pending'
                entry['last_error'] = 'Export completed but no staged GCS shards were found.'
                entry.pop('task_ids', None)
                changed = True
                continue
            existing_targets = glob(target_prefix + '*')
            promoted_targets = []
            try:
                for source in staged:
                    staged_name = source.rsplit('/', 1)[-1]
                    prefix_name = stage_prefix.rsplit('/', 1)[-1]
                    if not staged_name.startswith(prefix_name):
                        raise RuntimeError(f"Unexpected staged M1 shard name: {source}")
                    suffix = staged_name[len(prefix_name):]
                    destination = target_prefix + suffix
                    copy(source, destination)
                    promoted_targets.append(destination)
                obsolete_targets = [
                    target for target in existing_targets
                    if target not in promoted_targets
                ]
                if obsolete_targets:
                    rm_files(obsolete_targets)
                rm_files(staged)
            except Exception as e:
                entry['last_error'] = f"Staged shard promotion failed: {e}"
                print(f"[WARN] {entry['last_error']}")
                changed = True
                continue
            entry['status'] = 'chunks_ready'
            entry.pop('task_ids', None)
            entry.pop('staging_prefix', None)
            entry.pop('target_prefix', None)
            changed = True

        if changed:
            save_manifest(self.update_manifest)

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
    
    country_regions = ee.FeatureCollection(CONFIG['asset_regions'])
    country_geometry = country_regions.geometry()
    try:
        manifest = ui_obj.update_manifest
    except AttributeError:
        manifest = load_manifest()

    succeeded = 0
    for i, meta in enumerate(selected, 1):
        y, m, p = meta['year'], meta['month'], meta['period']
        period_key = meta.get('period_key') or (f"{y}_{m:02d}" if m else str(y))
        info = parse_period(period_key)
        y = info.year
        m = info.index if info.periodicity == 'monthly' else None
        p = info.periodicity
        band, mosaic_m, sensor = meta['band'], meta['mosaic'], meta['sensor']

        # Recheck GCS at execution time. A stale cache/manifest must not
        # schedule a new Earth Engine export over an existing COG/chunks.
        if meta['type'] == 'GCS':
            if update_existing_gcs_output(meta, manifest):
                succeeded += 1
                cell_log(
                    f"Updated existing GCS output: {sensor} {period_key} ({band})",
                    type='success',
                )
                continue
            meta['update_pending'] = False
            meta['exists'] = False

        name_base = mosaic_name_for_period(period_key, mosaic=mosaic_m, sensor=sensor, band=None)
        name_full = mosaic_name_for_period(period_key, band=band, mosaic=mosaic_m, sensor=sensor)

        try:
            print(f"  [{i}/{len(selected)}] Sending {sensor} {mosaic_m} ({band}) {period_key} for {meta['type']}...")

            start_date = info.start.strftime('%Y-%m-%d')
            end_date = info.end.strftime('%Y-%m-%d')

            mosaic_obj = logic.get_quality_mosaic(
                sensor, y, start_date, end_date, country_geometry,
                month=m, method=mosaic_m,
            )

            if meta['type'] == 'ASSET':
                task = logic.export_to_asset(
                    mosaic_obj, name_full, y, m, p, band=band, mosaic=mosaic_m,
                    sensor=sensor, period_key=period_key,
                )
                entry, _ = add_pending_item(manifest, meta)
                if not meta.get('update_pending'):
                    entry['before'] = 'not previously exported'
                set_item_status(
                    manifest, meta,
                    'submitted' if meta.get('update_pending') else 'submitted_new',
                    task_id=getattr(task, 'id', None),
                )
            else:
                stage_prefix = None
                target_prefix = None
                if meta.get('update_pending'):
                    import uuid
                    from M0_auth_config import period_chunk_path, CONFIG
                    stage_dir = f"{period_chunk_path(period_key, mosaic=mosaic_m, sensor=sensor)}/.m1_updates/{uuid.uuid4().hex}"
                    stage_prefix = f"{stage_dir}/{name_full}_"
                    target_prefix = f"gs://{CONFIG['bucket']}/{period_chunk_path(period_key, mosaic=mosaic_m, sensor=sensor)}/{name_full}_"
                tasks = logic.export_to_gcs(
                    mosaic_obj, name_base, y, m, p, bands=[band], mosaic=mosaic_m,
                    sensor=sensor, period_key=period_key,
                    staging_prefix=stage_prefix.rsplit('/', 1)[0] if stage_prefix else None,
                )
                entry, _ = add_pending_item(manifest, meta)
                if not meta.get('update_pending'):
                    entry['before'] = 'not previously exported'
                set_item_status(
                    manifest, meta,
                    'exporting' if meta.get('update_pending') else 'submitted_new',
                    task_ids=[getattr(task, 'id', None) for task in tasks],
                    **({
                        'staging_prefix': stage_prefix,
                        'target_prefix': target_prefix,
                    } if meta.get('update_pending') else {}),
                )
            succeeded += 1
            cell_log(f"Export sent: {sensor} {mosaic_m} ({band}) {period_key} [{meta['type']}]", type='success')
        except Exception as e:
            print(f"  [ERR] Failed {sensor} {mosaic_m} ({band}) {period_key}: {e}")
            cell_log(f"Failed {sensor} {mosaic_m} ({band}) {period_key}: {e}", type='error')
            traceback.print_exc()
            
    print(f"\n Completed! {succeeded}/{len(selected)} selected items processed.")
    cell_log(
        f"Completed! {succeeded}/{len(selected)} selected items processed.",
        type='success' if succeeded == len(selected) else 'warning',
    )

    if succeeded:
        # New files are still being written asynchronously; drop the cache stamp
        # so M2 rescans GCS when it opens instead of trusting the old state.
        CacheManager.mark_stale()
    try:
        save_manifest(manifest)
    except Exception as e:
        print(f"[WARN] Could not persist M1 update task state: {e}")
    return succeeded
