print("\n>>> M2_mosaic_ui initializing (v7.0 Tabs) <<<")
import traceback
import errno
import copy
import threading
import os
import posixpath
import re
import shutil
import sys
import ipywidgets as widgets
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

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

# Provisional cap: full-size COGs compete for disk. Increase only after
# representative end-to-end benchmarks.
M2_MAX_WORKERS = 2
# Avoid assuming that more compression threads necessarily improve throughput.
M2_MAX_GDAL_THREADS = 8
_CGROUP_UNKNOWN = object()


def choose_m2_workers(selected_count, cpu_count, total_ram, available_ram, free_disk):
    """Choose a conservative product concurrency from observed resources.

    RAM and disk are in bytes. Unknown resources fall back to one worker;
    these thresholds do not guarantee enough disk for a particular COG.
    """
    gib = 1024 ** 3
    if selected_count <= 0:
        return 0
    if (cpu_count is not None and cpu_count >= 4
            and total_ram is not None and total_ram >= 32 * gib
            and available_ram is not None and available_ram >= 24 * gib
            and free_disk is not None and free_disk >= 100 * gib):
        return min(selected_count, M2_MAX_WORKERS, 2)
    return 1


def choose_gdal_threads(cpu_count, product_workers, available_ram):
    """Budget CPUs and a conservative RAM reserve across GDAL products."""
    cpu_budget = max(1, (cpu_count or 1) // max(1, product_workers))
    if available_ram is None:
        return 1
    gib = 1024 ** 3
    ram_budget = max(1, int(max(0, available_ram - 2 * gib) // (gib // 2)))
    return max(1, min(M2_MAX_GDAL_THREADS, cpu_budget, ram_budget // max(1, product_workers)))


def _read_resource_file(path):
    try:
        with open(path, encoding='utf-8') as stream:
            return stream.read().strip()
    except (OSError, UnicodeError):
        return None


def _resource_file_absent(path):
    """Distinguish a missing root controller file from an unreadable one."""
    try:
        os.stat(path)
    except FileNotFoundError:
        return True
    except OSError:
        pass
    return False


def _cgroup_limits():
    """Return visible cgroup CPU and memory limits, including parent limits.

    Missing or inaccessible Linux cgroup files leave the corresponding limit
    unknown. Fractional CPU quotas are rounded down, with a minimum of one.
    """
    memberships = _read_resource_file('/proc/self/cgroup')
    mounts = _read_resource_file('/proc/self/mountinfo')
    if memberships is None or mounts is None:
        return (1, _CGROUP_UNKNOWN, None) if sys.platform.startswith('linux') else (None, None, None)
    groups = {}
    for line in memberships.splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3 and parts[2].startswith('/'):
            for controller in parts[1].split(','):
                groups[controller] = posixpath.normpath(parts[2])
        else:
            return (1, _CGROUP_UNKNOWN, None)
    if not groups and sys.platform.startswith('linux'):
        return (1, _CGROUP_UNKNOWN, None)
    cpu_limits, memory_limits, memory_available = [], [], []
    memory_usage_unknown = False
    cpu_unknown = memory_unknown = False
    matched = set()
    for line in mounts.splitlines():
        left, separator, right = line.partition(' - ')
        fields, fs = left.split(), right.split()
        if not separator or len(fields) < 5 or len(fs) < 3 or fs[0] not in ('cgroup', 'cgroup2'):
            continue
        controllers = [''] if fs[0] == 'cgroup2' else fs[2].split(',')
        root, mount = (re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)
                       for value in fields[3:5])
        for controller in controllers:
            if controller not in groups or controller not in ('', 'cpu', 'memory'):
                continue
            group = groups[controller]
            root = posixpath.normpath(root)
            if root != '/' and group != root and not group.startswith(root + '/'):
                continue
            matched.add(controller)
            relative = group[len(root):].lstrip('/') if root != '/' else group.lstrip('/')
            path = posixpath.join(mount, relative)
            while True:
                if controller in ('', 'cpu'):
                    if controller == '':
                        raw_cpu = _read_resource_file(posixpath.join(path, 'cpu.max'))
                        # The actual cgroup2 root normally has no controller
                        # limit files; a mounted subtree root is not exempt.
                        if (raw_cpu is None and root == '/' and path == mount
                                and _resource_file_absent(posixpath.join(path, 'cpu.max'))):
                            raw_cpu = 'max 100000'
                        value = (raw_cpu or '').split()
                        quota, period = (value + [None, None])[:2]
                    else:
                        quota = _read_resource_file(posixpath.join(path, 'cpu.cfs_quota_us'))
                        period = _read_resource_file(posixpath.join(path, 'cpu.cfs_period_us'))
                    try:
                        period = int(period)
                        if period <= 0:
                            raise ValueError('invalid period')
                        if controller == '' and quota == 'max':
                            pass
                        elif controller == 'cpu' and quota == '-1':
                            pass
                        else:
                            quota = int(quota)
                            if quota <= 0:
                                raise ValueError('invalid quota')
                            cpu_limits.append(max(1, quota // period))
                    except (ValueError, TypeError):
                        cpu_unknown = True
                if controller in ('', 'memory'):
                    limit_name = 'memory.max' if controller == '' else 'memory.limit_in_bytes'
                    usage_name = 'memory.current' if controller == '' else 'memory.usage_in_bytes'
                    try:
                        raw_limit = _read_resource_file(posixpath.join(path, limit_name))
                        if (controller == '' and raw_limit is None and root == '/' and path == mount
                                and _resource_file_absent(posixpath.join(path, limit_name))):
                            raw_limit = 'max'
                        if not (controller == '' and raw_limit == 'max'):
                            limit = int(raw_limit)
                            if limit < 0:
                                raise ValueError('invalid memory limit')
                            # cgroup v1 uses enormous sentinel values for unlimited.
                            if controller == '' or limit < (1 << 60):
                                memory_limits.append(limit)
                                try:
                                    usage = int(_read_resource_file(posixpath.join(path, usage_name)))
                                    if usage >= 0:
                                        memory_available.append(max(0, limit - usage))
                                    else:
                                        memory_usage_unknown = True
                                except (ValueError, TypeError):
                                    memory_usage_unknown = True
                    except (ValueError, TypeError):
                        memory_unknown = True
                if path == mount:
                    break
                parent = posixpath.dirname(path)
                if parent == path:
                    break
                path = parent
    if '' in groups and '' not in matched:
        cpu_unknown = memory_unknown = True
    if 'cpu' in groups and 'cpu' not in matched:
        cpu_unknown = True
    if 'memory' in groups and 'memory' not in matched:
        memory_unknown = True
    return (1 if cpu_unknown else min(cpu_limits, default=None),
            _CGROUP_UNKNOWN if memory_unknown else min(memory_limits, default=None),
            None if memory_unknown or memory_usage_unknown else min(memory_available, default=None))


def detect_m2_resources():
    """Read actual runtime resources, not an assumed Colab subscription tier."""
    try:
        cpu = getattr(os, 'process_cpu_count', os.cpu_count)()
    except (OSError, ValueError):
        cpu = None
    total = available = disk = None
    try:
        affinity = len(os.sched_getaffinity(0))
        if affinity:
            cpu = min(cpu, affinity) if cpu else affinity
    except (AttributeError, OSError, ValueError):
        pass
    cpu_limit, memory_limit, memory_free = _cgroup_limits()
    if cpu_limit is not None:
        cpu = min(cpu, cpu_limit) if cpu else cpu_limit
    try:
        import psutil
        memory = psutil.virtual_memory()
        total, available = memory.total, memory.available
        try:
            affinity = getattr(psutil.Process(), 'cpu_affinity', None)
            cpus = affinity() if affinity else []
            if cpus:
                cpu = min(cpu or len(cpus), len(cpus))
        except (psutil.Error, OSError, AttributeError):
            pass
    except (ImportError, OSError, AttributeError):
        pass
    if memory_limit is _CGROUP_UNKNOWN:
        total = available = None
    elif memory_limit is not None:
        total = min(total, memory_limit) if total is not None else memory_limit
        # A limit without usage cannot safely establish available memory.
        available = min(available, memory_free) if available is not None and memory_free is not None else None
    try:
        from M0_auth_config import get_temp_dir
        disk = shutil.disk_usage(get_temp_dir()).free
    except (OSError, ValueError):
        pass
    return cpu, total, available, disk

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
                    'type': 'GCS',
                    'country_update': country_update and (exists_cog or has_chunks),
                    'has_cog': exists_cog,
                    'has_chunks': has_chunks,
                })

                if country_update and exists_cog:
                    # A legacy COG update is explicitly selectable even
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


def process_country_mosaic_item(item, manifest, logger=print, persist_manifest=True,
                                gdal_threads=2):
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
                logger=logger, update_manifest=persist_manifest, gdal_threads=gdal_threads,
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
        update_manifest=persist_manifest, gdal_threads=gdal_threads,
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


def _is_disk_full(error):
    """Recognize OS and GDAL/subprocess disk-full errors, including causes."""
    import M2_mosaic_logic
    estimate_error = getattr(M2_mosaic_logic, 'COGSpaceEstimateExceeded', ())
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # A conservative preflight estimate is not evidence of actual ENOSPC.
        if isinstance(error, estimate_error):
            error = error.__cause__ or error.__context__
            continue
        if getattr(error, 'errno', None) == errno.ENOSPC:
            return True
        for value in (str(error), getattr(error, 'stderr', None), getattr(error, 'output', None)):
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            if isinstance(value, str) and ('no space left on device' in value.lower()
                                           or 'enospc' in value.lower()):
                return True
        error = error.__cause__ or error.__context__
    return False


def start_mosaic_assembly(ui_obj):
    selected = [chk._meta for chk in ui_obj.chk_dict.values() if chk.value]
    if not selected:
        print("[WARNING] No mosaic selected.")
        cell_log("No mosaic selected.", type='warning')
        return

    unique = []
    seen = set()
    for item in selected:
        period_key = item.get('period_key') or (
            f"{item['year']}_{item['month']:02d}" if item.get('month') else str(item['year'])
        )
        destination = (str(item['sensor']).lower(), str(item['mosaic']).lower(),
                       period_key, str(item['band']).lower())
        if destination in seen:
            print(f"[WARN] Duplicate destination ignored: {destination}")
            continue
        seen.add(destination)
        unique.append(item)
    selected = unique
    
    resources = detect_m2_resources()
    worker_count = choose_m2_workers(len(selected), *resources)
    cpu, total, available, free_disk = resources
    gdal_threads = choose_gdal_threads(cpu, worker_count, available)
    gib = 1024 ** 3
    def format_gib(value):
        return f"{value / gib:.1f} GiB" if value is not None else 'unknown'
    print(f"[M2 resources] CPUs={cpu or 'unknown'} | RAM total={format_gib(total)} "
          f"available={format_gib(available)} | temp free={format_gib(free_disk)} | "
          f"product workers={worker_count} (GDAL: up to {gdal_threads} threads per product)")
    print(
        f"[INFO] Starting assembly of {len(selected)} COG bands "
        f"with {worker_count} worker(s)..."
    )
    cell_log(
        f"Starting assembly of {len(selected)} COG bands with {worker_count} worker(s)...",
        type='info',
    )

    manifest = getattr(ui_obj, 'update_manifest', {'version': 1, 'items': []})
    succeeded = failed = unconfirmed = 0
    disk_full = threading.Event()
    scheduling_lock = threading.Lock()
    persistence_failed = False
    prog = ProgressTracker(len(selected), "COG Assembly")
    def persist_change(change):
        nonlocal persistence_failed
        if persistence_failed:
            return False
        try:
            candidate = copy.deepcopy(manifest)
            change(candidate)
            save_manifest(candidate)
        except Exception as error:
            persistence_failed = True
            message = f"Manifest persistence failed; no further writes or submissions: {error}"
            print(f"[ERR] {message}")
            cell_log(message, type='error')
            return False
        change(manifest)
        return True

    def run_item(index, item):
        period_key = item.get('period_key') or (
            f"{item['year']}_{item['month']:02d}"
            if item.get('month') else f"{item['year']}"
        )
        cell_log(
            f"[{index}/{len(selected)}] Processing {item['sensor']} "
            f"{period_key} {item['band']}...", type='info',
        )
        try:
            return process_country_mosaic_item(
                item, manifest, logger=print, persist_manifest=False, gdal_threads=gdal_threads,
            )
        except Exception as error:
            if _is_disk_full(error):
                disk_full.set()
            raise

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_items = {}
        next_index = 0
        while future_items or (next_index < len(selected) and not disk_full.is_set()
                               and not persistence_failed):
            while (not disk_full.is_set() and not persistence_failed
                   and next_index < len(selected) and len(future_items) < worker_count):
                item = selected[next_index]
                original_entry = find_item(manifest, item) if item.get('country_update') else None
                previous = ({key: original_entry[key] for key in ('status', 'last_error')
                             if key in original_entry} if original_entry is not None else None)
                if item.get('country_update'):
                    status = 'masking_cog' if item.get('has_cog') else 'chunks_ready'
                    if not persist_change(lambda data: set_item_status(data, item, status)):
                        break
                with scheduling_lock:
                    stopped = disk_full.is_set()
                    if not stopped:
                        next_index += 1
                        future_items[executor.submit(run_item, next_index, item)] = (next_index, item)
                if stopped:
                    if item.get('country_update'):
                        def restore_unstarted(data):
                            entry = find_item(data, item)
                            if previous is None:
                                if entry is not None:
                                    data['items'].remove(entry)
                            elif entry is not None:
                                for key in ('status', 'last_error'):
                                    entry.pop(key, None)
                                entry.update(previous)
                        if not persist_change(restore_unstarted):
                            unconfirmed += 1
                    break
            if not future_items:
                break
            done, _ = wait(future_items, return_when=FIRST_COMPLETED)
            for future in done:
                index, item = future_items.pop(future)
                period_key = item.get('period_key') or (
                    f"{item['year']}_{item['month']:02d}"
                    if item['month'] else f"{item['year']}"
                )
                try:
                    result = future.result()
                    if not result:
                        raise RuntimeError('M2 did not produce an output COG.')
                except Exception as e:
                    failed += 1
                    print(f"[ERR] Failed {item['sensor']} {period_key} {item['band']}: {e}")
                    if item.get('country_update'):
                        def record_failure(data):
                            entry = set_item_status(data, item, 'pending')
                            entry['last_error'] = str(e)
                        if not persist_change(record_failure):
                            unconfirmed += 1
                    cell_log(
                        f"Failed {item['sensor']} {period_key} {item['band']}: {e}",
                        type='error',
                    )
                    traceback.print_exc()
                else:
                    succeeded += 1
                    confirmed = (not item.get('country_update')
                                 or persist_change(lambda data: mark_item_complete(data, item)))
                    if not confirmed:
                        unconfirmed += 1
                    cell_log(
                        f"OK: {item['sensor']} {period_key} {item['band']}"
                        + (" (manifest unconfirmed)" if not confirmed else ""),
                        type='success' if confirmed else 'warning',
                    )
                finally:
                    prog.step()
    skipped = len(selected) - next_index
    summary = (f"Assembly completed: {succeeded} succeeded, {failed} failed, "
               f"{skipped} skipped, {unconfirmed} manifest unconfirmed"
               + (" (disk full; not started)" if disk_full.is_set() else "")
               + (" (manifest persistence failed)" if persistence_failed else ""))
    print(f"\n[SUMMARY] {summary}.")
    cell_log(f"{summary} | {prog.summary()}",
             type='warning' if failed or skipped or persistence_failed else 'success')

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
