print("\n>>> M2_mosaic_ui initializing (v7.0 Tabs) <<<")
import traceback
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor, as_completed

from M0_auth_config import CONFIG, GLOBAL_OPTS, mosaic_name_for_period, is_edit_mode, period_cog_path, get_sensor_start_year, resolve_years
from M_periods import list_periods, period_state_key
from M_cache import CacheManager
from M_ui_components import PipelineStepUI, cell_log, ProgressTracker, make_update_note
from M_lang import L as Lang
from M_mosaics import methods_for
from M1_update_manifest import (
    find_item, load_manifest, mark_item_complete, pending_item,
    save_manifest, set_item_status,
)

M2_MAX_WORKERS = 2

class MosaicAssemblerUI(PipelineStepUI):
    def __init__(self, years=None):
        super().__init__(
            title=Lang.M2_CONSTRUCTOR_TITLE, 
            description=Lang.M2_CONSTRUCTOR_DESC
        )
        self.chk_dict = {}
        self.requested_years = years
        self.btn_refresh = None
        self.is_refreshing = False
        # Persistência de abas
        self.last_tabs = (0, 0, 0)
        
        try:
            self.main_area.children = [widgets.HTML(f"<i>{Lang.LOADING_INTERFACE}</i>")]
        except Exception as e:
            print(f"CRITICAL ERROR IN M2 INTERFACE: {e}")
            traceback.print_exc()

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

    def _init_data(self):
        import datetime
        self.now = datetime.datetime.now()
        self.years = resolve_years(self.requested_years)
        self.bands = CONFIG['bands_all']
        self.state = CacheManager.load() or {}
        try:
            self.update_manifest = load_manifest()
        except Exception as e:
            print(f"[WARN] M2 update manifest unavailable: {e}")
            self.update_manifest = {'version': 1, 'items': []}
        # Chunks/COGs exported by earlier sessions live only in GCS. Rebuild the
        # cache when it is empty (fresh clone) or older than the TTL, so the
        # already-exported mosaics show up as READY/OK without a manual Sync.
        if self._cache_is_stale():
            try:
                self.state = CacheManager.build_cache_from_gcs(logger=self.update_status, years=self.years)
            except Exception as e:
                print(f"[WARN] M2: could not sync GCS cache: {e}")

    def _on_select_all(self, _):
        for chk in self.chk_dict.values():
            if not chk.disabled: chk.value = True

    def _on_select_none(self, _):
        for chk in self.chk_dict.values():
            if not chk.disabled: chk.value = False

    def _build_mosaic_grid(self, sensor, period, mosaic_method):
        L = widgets.Layout
        hdr = [
            widgets.HTML(f'<div style="width:100px;">{Lang.DATE}</div>'),
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
        for period_info in periods:
            if period_info.year < get_sensor_start_year(sensor):
                continue
            date_str = period_info.key
            y = period_info.year
            m = period_info.index if period_info.periodicity == 'monthly' else None

            cells = [widgets.HTML(f'<div style="width:100px;font-family:monospace;">{date_str}</div>')]
                
            btn_s = widgets.Button(description=Lang.BTN_SELECT_ROW, layout=L(width='40px', height='28px', padding='0'))
            btn_s._sensor, btn_s._period, btn_s._mosaic, btn_s._date = sensor, period, mosaic_method, date_str
            btn_s.on_click(self._on_select_row)
            cells.append(btn_s)

            for b in self.bands:
                # Lógica de existência (COG) vs disponibilidade (Chunks)
                # Sincronizado com CacheManager
                m_name = mosaic_name_for_period(date_str, band=b, mosaic=mosaic_method, sensor=sensor)
                
                # 1. Verifica se o COG final existe
                cogs_list = self.state.get(period_state_key('cogs', period), [])
                exists_cog = m_name.lower() in [c.lower() for c in cogs_list]
                
                # 2. Verifica se existem chunks para montar
                # No CacheManager, a chave dos chunks é o nome base sem a banda
                m_base_name = mosaic_name_for_period(date_str, mosaic=mosaic_method, sensor=sensor)
                chunk_bands = self.state.get('gcs_chunks', {}).get(m_base_name.lower(), [])
                has_chunks = b.lower() in {chunk.lower() for chunk in chunk_bands}

                chk = widgets.Checkbox(value=False, indent=False, layout=L(width='18px', height='18px', margin='0'))
                chk._meta = {'sensor': sensor, 'period': period, 'mosaic': mosaic_method, 'year': y, 'month': m, 'period_key': date_str, 'band': b}

                update_entry = find_item(self.update_manifest, {
                    'country': CONFIG['country'], 'sensor': sensor,
                    'period': period, 'mosaic': mosaic_method,
                    'period_key': date_str, 'band': b, 'type': 'GCS',
                })
                country_update = bool(update_entry and pending_item(
                    self.update_manifest, {
                        'country': CONFIG['country'], 'sensor': sensor,
                        'period': period, 'mosaic': mosaic_method,
                        'period_key': date_str, 'band': b, 'type': 'GCS',
                    },
                ))
                chk._meta.update({
                    'country': CONFIG['country'],
                    'country_update': country_update and (exists_cog or has_chunks),
                    'has_cog': exists_cog,
                    'has_chunks': has_chunks,
                })

                if country_update and (exists_cog or has_chunks):
                    # A legacy COG/chunk update is explicitly selectable even
                    # when edit mode is off; the ordinary M2 start cell runs it.
                    chk.disabled = False
                    cell = PipelineStepUI.make_status_cell(
                        chk, Lang.M1_UPDATE_LABEL, 'mfm-update', width='auto',
                    )
                elif exists_cog:
                    if not is_edit_mode(): chk.disabled = True
                    cog_dir = period_cog_path(date_str, mosaic=mosaic_method, sensor=sensor)
                    cog_file = f"{m_name}_cog.tif"
                    gcs_url = f"https://storage.googleapis.com/{CONFIG['bucket']}/{cog_dir}/{cog_file}"
                    status_html = widgets.HTML(
                        f'<a href="{gcs_url}" target="_blank" '
                        f'style="font-size:10px;font-weight:700;color:#155724;text-decoration:underline;">OK ↗</a>',
                        layout=widgets.Layout(width='32px')
                    )
                    cell = widgets.HBox(
                        [chk, status_html],
                        layout=widgets.Layout(
                            width='auto', min_height='34px',
                            justify_content='center', align_items='center',
                            padding='0', overflow='hidden', margin='1px'
                        )
                    )
                    cell.add_class('mfm-ok')
                elif has_chunks:
                    cell = PipelineStepUI.make_status_cell(chk, 'READY', 'mfm-run', width='auto')
                else:
                    cell = PipelineStepUI.make_status_cell(chk, 'MISS', 'mfm-null', width='auto')
                    chk.disabled = True

                cell.layout.flex = '1'
                self.chk_dict[f"{sensor}_{period}_{mosaic_method}_{date_str}_{b}"] = chk
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

    def _on_select_row(self, btn):
        prefix = f"{btn._sensor}_{btn._period}_{btn._mosaic}_{btn._date}_"
        row_chks = [chk for key, chk in self.chk_dict.items() if key.startswith(prefix) and not chk.disabled]
        if not row_chks: return
        any_on = any(chk.value for chk in row_chks)
        target_val = not any_on
        for chk in row_chks: chk.value = target_val
    
    def _build_ui(self):
        self._init_data()
        # Limpa widgets de título da classe base para usar o novo header compacto
        self.header_title.value = ""
        self.header_desc.value = ""
        # Remove loader do topo para não ficar duplicado (já estará no footer)
        self.header_box.children = [self.header_title]
        
        L = widgets.Layout
        self.chk_dict = {}

        # --- HEADER COMPACTO (LINHA ÚNICA) ---
        header_html = f'''
        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 5px 10px; background: #fff; border-bottom: 2px solid #333; margin-bottom: 5px;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <span style="font-weight: bold; font-size: 16px; color: #333;">{Lang.M2_HEADER_TITLE}</span>
                <span style="color: #888; font-size: 11px; font-style: italic;">{Lang.M2_HEADER_SUBTITLE}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px; padding: 3px 12px; background: #fff1f0; border: 1px solid #ffa39e; border-radius: 4px;">
                <span style="color: #cf1322; font-size: 10px; font-weight: bold; text-transform: uppercase;">{Lang.LABEL_PROJECT}</span>
                <span style="color: #cf1322; font-weight: bold; font-size: 12px;">MapBiomas Fire Monitor</span>
            </div>
        </div>
        '''
        
        # --- FOOTER DE CONTROLE ---
        self.btn_refresh = widgets.Button(description=Lang.MOSAIC_SYNC, button_style='success', icon='refresh', layout=L(width='180px'))
        self.btn_refresh.on_click(lambda _: self._refresh_cache())
        
        btn_all = widgets.Button(description=Lang.MOSAIC_SELECT, button_style='info', layout=L(width='180px'))
        btn_none = widgets.Button(description=Lang.MOSAIC_CLEAR, button_style='warning', layout=L(width='150px'))
        btn_all.on_click(self._on_select_all)
        btn_none.on_click(self._on_select_none)
        
        # Loader integrado ao rodapé
        footer = widgets.VBox([
            widgets.HBox([
                btn_all, btn_none, self.btn_refresh,
                make_update_note(Lang.M1_UPDATE_M2_NOTE), self.loader_html,
            ], layout=L(margin='15px 0', gap='10px', align_items='center', flex_flow='row wrap'))
        ])

        # --- SISTEMA DE ABAS (TABS) ---
        sensors = [s.upper() for s in (GLOBAL_OPTS['SENSOR'] if isinstance(GLOBAL_OPTS['SENSOR'], list) else [GLOBAL_OPTS['SENSOR']])]
        periods = GLOBAL_OPTS['PERIODICITY'] if isinstance(GLOBAL_OPTS['PERIODICITY'], list) else [GLOBAL_OPTS['PERIODICITY']]
        
        self.sensor_tabs = widgets.Tab()
        self.sensor_children = []
        self.tab_map = {} # (s_idx, p_idx, m_idx) -> (sensor, period, method)

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
        guide_widget = widgets.HTML(Lang.GUIDE_M2_HTML)
        self.sensor_tabs.children = [guide_widget] + self.sensor_children
        self.sensor_tabs.set_title(0, Lang.TAB_GUIDE)
        for i, s in enumerate(sensors):
            self.sensor_tabs.set_title(i + 1, s)

        self.sensor_tabs.observe(self._on_sensor_change, names='selected_index')

        # Restaurar abas anteriores
        s_idx, p_idx, m_idx = self.last_tabs  # s_idx is already visual (with guide)
        if s_idx == 0:
            s_idx = 1  # fallback to first sensor
        self.sensor_tabs.selected_index = s_idx
        real_idx = s_idx - 1
        self.sensor_children[real_idx].selected_index = p_idx
        self.sensor_children[real_idx].children[p_idx].selected_index = m_idx
        
        # Gatilho inicial
        self._load_tab(real_idx, p_idx, m_idx)

        self.clear_main()
        self.main_area.children = [
            PipelineStepUI.get_status_css(), 
            widgets.HTML(header_html), 
            self.sensor_tabs, 
            footer
        ]

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

    def _load_tab(self, s_idx, p_idx, m_idx):
        sensor, period, method = self.tab_map[(s_idx, p_idx, m_idx)]
        p_tabs = self.sensor_children[s_idx]
        method_tabs = p_tabs.children[p_idx]
        
        target_container = method_tabs.children[m_idx]
        if not isinstance(target_container.children[0], widgets.HTML) or "<i>" not in target_container.children[0].value:
            return

        grid = self._build_mosaic_grid(sensor, period, method)
        
        new_children = list(method_tabs.children)
        new_children[m_idx] = grid
        method_tabs.children = new_children

    def _reload_current_tab(self):
        """Forca a aba ativa a reconstruir o grid com o cache em memoria."""
        vis_idx = self.sensor_tabs.selected_index
        if vis_idx == 0:
            vis_idx = 1
        real_idx = vis_idx - 1
        p_tabs = self.sensor_children[real_idx]
        p_idx = p_tabs.selected_index
        m_idx = p_tabs.children[p_idx].selected_index
        target = p_tabs.children[p_idx].children[m_idx]
        target.children = [widgets.HTML(f"<i>{Lang.LOADING_INTERFACE}</i>")]
        self._load_tab(real_idx, p_idx, m_idx)

    def _refresh_cache(self):
        if self.is_refreshing: return
        try:
            self.is_refreshing = True
            # Salva abas atuais antes do rebuild (visual index with guide)
            vis_idx = self.sensor_tabs.selected_index
            if vis_idx == 0:
                vis_idx = 1  # fallback to first sensor
            real_idx = vis_idx - 1
            p_idx = self.sensor_children[real_idx].selected_index
            m_idx = self.sensor_children[real_idx].children[p_idx].selected_index
            self.last_tabs = (vis_idx, p_idx, m_idx)
            
            self.show_loader(Lang.SYNCING)
            self.state = CacheManager.build_full_cache(logger=self.update_status, years=self.years)
            self._build_ui()
        finally:
            self.is_refreshing = False
            self.hide_loader()

def run_ui(years=None):
    ui = MosaicAssemblerUI(years=years)
    ui.display() 
    ui.show_loader(Lang.LOADING)
    ui._build_ui()
    ui.hide_loader()
    return ui


def process_country_mosaic_item(item, manifest, logger=print, persist_manifest=True):
    """Apply an M2 country-mask update or run the usual chunks assembly."""
    from M2_mosaic_logic import assemble_country_mosaic, mask_existing_country_cog

    period_key = item.get('period_key') or (
        f"{item['year']}_{item['month']:02d}" if item.get('month') else str(item['year'])
    )
    country_update = item.get('country_update', False)
    has_chunks = item.get('has_chunks', False)

    if country_update and item.get('has_cog'):
        if persist_manifest:
            set_item_status(manifest, item, 'masking_cog')
            save_manifest(manifest)
        try:
            updated = mask_existing_country_cog(
                period_key, item['band'], item['sensor'], item['mosaic'],
                logger=logger, update_manifest=persist_manifest,
            )
        except Exception as error:
            if persist_manifest:
                set_item_status(manifest, item, 'pending', last_error=str(error))
                save_manifest(manifest)
            raise
        if updated:
            if persist_manifest:
                mark_item_complete(manifest, item)
                save_manifest(manifest)
            return [f"updated COG {period_key}/{item['band']}"]
        if not has_chunks:
            error = f"COG disappeared before update and no chunks are available: {period_key}/{item['band']}"
            if persist_manifest:
                set_item_status(manifest, item, 'pending', last_error=error)
                save_manifest(manifest)
            raise RuntimeError(error)

    if country_update:
        if not has_chunks:
            error = f"No COG or chunks available for M2 update: {period_key}/{item['band']}"
            if persist_manifest:
                set_item_status(manifest, item, 'pending', last_error=error)
                save_manifest(manifest)
            raise RuntimeError(error)
        if persist_manifest:
            set_item_status(manifest, item, 'chunks_ready')
            save_manifest(manifest)

    result = assemble_country_mosaic(
        year=item['year'], month=item['month'], period=item['period'],
        sensor=item['sensor'], mosaic_method=item['mosaic'],
        bands=[item['band']], period_key=period_key, logger=logger,
        update_manifest=persist_manifest,
    )
    if country_update:
        if not result:
            error = f"M2 did not produce a COG for {period_key}/{item['band']}."
            if persist_manifest:
                set_item_status(manifest, item, 'pending', last_error=error)
                save_manifest(manifest)
            raise RuntimeError(error)
        if persist_manifest:
            set_item_status(
                manifest, item, 'complete',
            )
            save_manifest(manifest)
    return result


def start_mosaic_assembly(ui_obj):
    selected = [chk._meta for chk in ui_obj.chk_dict.values() if chk.value]
    if not selected:
        print("[WARNING] No mosaic selected.")
        cell_log("No mosaic selected.", type='warning')
        return
    
    worker_count = min(M2_MAX_WORKERS, len(selected))
    print(
        f"[INFO] Starting assembly of {len(selected)} COG bands "
        f"with {worker_count} worker(s)..."
    )
    cell_log(
        f"Starting assembly of {len(selected)} COG bands with {worker_count} worker(s)...",
        type='info',
    )

    manifest = getattr(ui_obj, 'update_manifest', {'version': 1, 'items': []})
    manifest_changed = False
    for item in selected:
        if item.get('country_update'):
            set_item_status(
                manifest, item,
                'masking_cog' if item.get('has_cog') else 'chunks_ready',
            )
            manifest_changed = True
    if manifest_changed:
        save_manifest(manifest)

    succeeded = 0
    prog = ProgressTracker(len(selected), "COG Assembly")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_items = {
            executor.submit(
                process_country_mosaic_item, item, manifest,
                logger=print, persist_manifest=False,
            ): (index, item)
            for index, item in enumerate(selected, 1)
        }
        for future in as_completed(future_items):
            index, item = future_items[future]
            period_key = item.get('period_key') or (
                f"{item['year']}_{item['month']:02d}"
                if item['month'] else f"{item['year']}"
            )
            try:
                cell_log(
                    f"[{index}/{len(selected)}] Processing {item['sensor']} "
                    f"{period_key} {item['band']}...",
                    type='info',
                )
                future.result()
                succeeded += 1
                if item.get('country_update'):
                    mark_item_complete(manifest, item)
                    save_manifest(manifest)
                cell_log(
                    f"OK: {item['sensor']} {period_key} {item['band']}",
                    type='success',
                )
            except Exception as e:
                print(f"[ERR] Failed {item['sensor']} {period_key} {item['band']}: {e}")
                if item.get('country_update'):
                    entry = set_item_status(manifest, item, 'pending')
                    entry['last_error'] = str(e)
                    save_manifest(manifest)
                cell_log(
                    f"Failed {item['sensor']} {period_key} {item['band']}: {e}",
                    type='error',
                )
                traceback.print_exc()
            finally:
                prog.step()
    print(f"\n[SUMMARY] Assembly completed: {succeeded}/{len(selected)} succeeded.")
    cell_log(f"Assembly completed: {succeeded}/{len(selected)} | {prog.summary()}", type='success')

    # The COGs were just uploaded: rescan GCS and redraw the active tab so they
    # appear as OK right away.
    if ui_obj is not None:
        try:
            ui_obj.state = CacheManager.build_cache_from_gcs(
                logger=print, years=getattr(ui_obj, 'years', None)
            )
            ui_obj._reload_current_tab()
        except Exception as e:
            print(f"[WARN] M2: could not refresh cache after assembly: {e}")
            # Chunks were removed, so the old cache is no longer trustworthy.
            CacheManager.mark_stale()
