import os
import base64
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from M0_auth_config import CONFIG, GLOBAL_OPTS, _get_fs, gcs_models_path, resolve_years
from M5_workplan import load_workplan, save_workplan, make_job_id, new_job, gcs_full, classified_tiles_dir, \
    tarea_path, save_tarea, delete_tarea, list_tareas, \
    save_pending_job_to_gcs, delete_pending_job_gcs, \
    load_pending_from_gcs, sync_gcs_to_local_workplan
from M4_data_extractor import list_campaigns_gcs
from M_ui_components import THEME, inline_confirm, make_spinner, make_empty_state, build_thumbnail_column, make_task_badges, make_card_body, flash_output, make_select_all_none, make_refresh_button, make_sync_button
from M_lang import L as Lang
from M_regions import REGION_NAME_PROPERTY
from M_periods import list_periods

L = widgets.Layout

class M5WorkplanUI:
    def __init__(self, years=None, periodicity_active=None):
        self.years = [2025, 2026] if years is None else resolve_years(years)
        configured_periods = [p.lower() for p in GLOBAL_OPTS.get('PERIODICITY', ['monthly'])]
        if periodicity_active and [p.lower() for p in periodicity_active] != configured_periods:
            print(f"M5 periodicity follows M0 configuration: {configured_periods}")
        self.periodicity_active = configured_periods
        self.plan = load_workplan()

        self._thumb_cache = {}
        self._grid_count_cache = {}
        self._processing_state = {}
        self._card_checkboxes = {}
        self._live_status_out = widgets.Output()
        self._model_meta_cache = {}

        self.w_model_box = widgets.VBox()
        self.w_region_box = widgets.VBox()
        self.w_period_box = widgets.VBox()

        self.chk_models = []
        self.chk_regions = []
        self.chk_periods = []

        self.btn_add_container, self.btn_add, _ = make_refresh_button('plus', self._on_add_click, description=Lang.ADD_BATCH, width='200px')
        self.btn_add.button_style = 'primary'
        
        self.btn_sync, _ = make_sync_button(Lang.SYNC_DATA, "refresh", self._sync_data, width='180px', button_style='success')

        self.w_task_name = widgets.Text(
            value='',
            placeholder=Lang.TASK_NAME_PLACEHOLDER,
            description=Lang.DROP_TASK_NAME,
            disabled=False,
            layout=L(width='600px')
        )

        self.w_campaign = widgets.Dropdown(
            options=[Lang.LOADING_CAMPAIGNS],
            value=Lang.LOADING_CAMPAIGNS,
            description=Lang.CAMPAIGN_LABEL,
            layout=L(width='400px')
        )
        self.w_campaign.observe(self._on_campaign_change, names='value')

        self.f_pend_search = widgets.Text(
            description=Lang.FILTER_LABEL, placeholder=Lang.FILTER_PLACEHOLDER,
            layout=L(width='100%'))
        self.f_pend_search.observe(lambda _: self._render_pending(), names='value')

        self.w_guide = widgets.HTML()

        self.w_pend_rows = widgets.VBox()
        self.tab_pending = widgets.VBox()

        self.w_mapa_rows = widgets.VBox()
        self.tab_mapa = widgets.VBox()
        self.f_mapa_model = widgets.Dropdown(description=Lang.DROP_MODEL, options=[Lang.ALL], layout=L(width='250px'))
        self.f_mapa_region = widgets.Dropdown(description=Lang.DROP_REGION, options=[Lang.ALL_F], layout=L(width='250px'))
        self.f_mapa_year = widgets.Dropdown(description=Lang.DROP_YEAR, options=[Lang.ALL], layout=L(width='200px'))
        self.btn_mapa_refresh_container, self.btn_mapa_refresh, _ = make_refresh_button('refresh', self._render_mapa, description=Lang.REFRESH_MAP, width='150px')
        for f in [self.f_mapa_model, self.f_mapa_region, self.f_mapa_year]:
            f.observe(lambda _: self._render_mapa(), names='value')

        self.out_msg = widgets.Output()

        self.tabs = widgets.Tab()
        self._build_guide()
        self._populate_dropdowns()

    # --- PERIODOS ---

    def _get_all_periods(self):
        """Retorna dict {year: [period_label, ...]} com todos periodos disponiveis."""
        import datetime
        now = datetime.datetime.now()
        periods_by_year = {}
        for periodicity in self.periodicity_active:
            for period in list_periods(
                self.years, periodicity, include_current=False, now=now
            ):
                label = '' if period.periodicity == 'yearly' else period.key.split('_', 1)[1]
                periods_by_year.setdefault(period.year, []).append(label)
        return periods_by_year

    # --- THUMBNAIL (GEE) ---

    def _generate_thumb(self, model, size=128, regions=None):
        """Gera thumbnail GEE das cells de um modelo. Cache em memoria."""
        reg_suffix = '_'.join(sorted(regions or []))
        cache_key = f"{model}_{size}_{reg_suffix}"
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key]
        try:
            import ee
            import requests
            if not getattr(ee.data, '_credentials', None):
                from M0_auth_config import authenticate
                authenticate()
            grid = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")

            if regions is not None:
                all_regions = ee.FeatureCollection(CONFIG['asset_regions']).filter(ee.Filter.inList(REGION_NAME_PROPERTY, regions))
            else:
                all_regions = ee.FeatureCollection(CONFIG['asset_regions'])

            region_lines = all_regions.style(**{'color': '2980b9', 'width': 1, 'fillColor': '00000000'})
            grid_lines = grid.style(**{'color': 'e0e0e0', 'width': 0.3, 'fillColor': '00000000'})

            if regions:
                sel_fill = all_regions.style(**{'color': '2980b9', 'width': 1, 'fillColor': '2980b920'})
                layers = [region_lines, grid_lines, sel_fill]
            else:
                layers = [region_lines, grid_lines]

            overlay = ee.ImageCollection(layers).mosaic()
            bounds = all_regions.geometry().bounds(1, 'EPSG:4326')
            url = overlay.getThumbURL({'region': bounds, 'dimensions': size, 'format': 'png'})
            resp = requests.get(url, timeout=60)
            b64 = base64.b64encode(resp.content).decode('ascii')
            self._thumb_cache[cache_key] = b64
            return b64
        except Exception:
            return None

    # --- TIMELINE (HTML) ---

    def _build_year_line(self, year, months, jobs_for_model_region):
        """Retorna HTML de uma linha de timeline: label ano + 12 bolinhas mes."""
        dots = []
        for m in months:
            period = f"{year}_{m}" if m else str(year)
            status = 'none'
            for j in jobs_for_model_region:
                if j['period'] == period:
                    if j['status'] in ('COMPLETED', 'FINISHED'):
                        status = 'done'
                    elif j['status'] in ('PENDING', 'RUNNING'):
                        status = 'running'
                    break
            colors = {'none': '#ddd', 'running': THEME['WARNING'], 'done': THEME['SUCCESS']}
            title = period if m else str(year)
            dots.append(f'<span title="{title}" style="display:inline-block;width:14px;height:14px;'
                        f'border-radius:50%;background:{colors[status]};margin:0 2px;'
                        f'border:1px solid #999;"></span>')
        label = str(year) if months and months != [''] else ''
        return f'<div style="display:flex;align-items:center;margin:2px 0;">' \
               f'<span style="width:50px;font-size:11px;color:#555;">{label}</span>' \
               f'<div style="display:flex;flex-wrap:wrap;">{"".join(dots)}</div></div>'

    # --- GUIA ---

    def _build_guide(self):
        html = Lang.GUIDE_M5_HTML.format(
            tab_register=Lang.TAB_REGISTER,
            tab_pending=Lang.TAB_PENDING,
            tab_map=Lang.TAB_MAP
        )
        self.w_guide = widgets.HTML(value=html)

    # --- CHECKBOX GRID ---

    def _create_checkbox_grid(self, options, description, single_select=False, bg_color='#fafafa', columns=None):
        title = widgets.HTML(f"<div style='margin-bottom:5px; color:#2c3e50;'><b>{description}</b></div>")
        checkboxes = []
        for opt in options:
            chk = widgets.Checkbox(value=False, description=str(opt), indent=False, style={'description_width': 'initial'}, layout=L(width='auto', margin='0 5px 5px 0'))
            checkboxes.append(chk)
        if single_select:
            def on_change(change, current_chk):
                if change['new']:
                    for c in checkboxes:
                        if c != current_chk:
                            c.value = False
            for chk in checkboxes:
                chk.observe(lambda change, c=chk: on_change(change, c), names='value')
        gtc = f"repeat({columns}, 1fr)" if columns else "repeat(auto-fill, minmax(280px, 1fr))"
        grid = widgets.GridBox(checkboxes, layout=L(grid_template_columns=gtc, width='100%'))
        container = widgets.VBox([title, grid], layout=L(margin='0 0 15px 0', padding='10px', border='1px solid #ccc', border_radius='5px', background_color=bg_color))
        return container, checkboxes

    def _populate_dropdowns(self):
        load_error = None
        try:
            from M4_model_trainer import list_trained_models
            models = list_trained_models(force_refresh=True)
        except Exception as e:
            load_error = str(e)
            print(f"[M5] Could not list trained models: {e}")
            models = []

        models = [m for m in (models or []) if isinstance(m, str)]
        if models:
            box, self.chk_models = self._create_checkbox_grid(models, Lang.SECTION_MODEL, single_select=True, bg_color='#e8f4fd')
            self.w_model_box.children = box.children
        else:
            self.chk_models = []
            path = f"gs://{CONFIG['bucket']}/{gcs_models_path()}"
            detail = (
                f"No trained models found in <code>{path}</code> "
                f"(campaign=<b>{CONFIG.get('campaign', '?')}</b>)."
                "<br>Train a model in M4 with this same campaign, or check the GCS path."
            )
            if load_error:
                detail += f"<br><span style='color:#b91c1c;'>Error: {load_error}</span>"
            self.w_model_box.children = (widgets.HTML(
                f"<div style='padding:12px;border:1px solid #fde68a;background:#fffbeb;border-radius:6px;font-size:12px;color:#92400e;'>{detail}</div>"
            ),)
        regions = [CONFIG.get('country', 'country').title()]
        err_banner = None
        try:
            import ee
            if not getattr(ee.data, '_credentials', None):
                from M0_auth_config import authenticate
                authenticate()
            asset = CONFIG.get('asset_regions')
            if not asset:
                err_banner = Lang.ERR_CONFIG_ASSET_REGIONS
            else:
                try:
                    ee.data.getAsset(asset)
                except Exception:
                    err_banner = f'<b>{Lang.ERR_ASSET_NOT_FOUND}</b><br><code>{asset}</code>'

                if not err_banner:
                    fc = ee.FeatureCollection(asset)
                    names = fc.aggregate_array(REGION_NAME_PROPERTY).distinct().getInfo()
                    if names:
                        regions = sorted([n for n in names if n])
                    else:
                        size = fc.size().getInfo()
                        if size == 0:
                            err_banner = f'<b>{Lang.ERR_ASSET_EMPTY}</b><br><code>{asset}</code><br><small>0 features</small>'
                        else:
                            err_banner = (f'<b>{Lang.ERR_ASSET_COLUMN.format(prop=REGION_NAME_PROPERTY)}:</b><br>'
                                          f'<code>{asset}</code><br>'
                                          f'<small>Asset tem {size} features, mas a propriedade <code>'
                                          f'{REGION_NAME_PROPERTY}</code> nao retornou valores.</small>')
        except Exception as e:
            err_banner = f'<b>{Lang.ERR_GEE_ACCESS}</b><br><span style="color:{THEME["ERROR"]};">{e}</span>'

        if err_banner:
                from IPython.display import display, HTML
                display(HTML(f"""
                <div style='background:#fff3cd; border:1px solid #ffc107; border-radius:5px; padding:12px; margin:8px 0; font-family:monospace; font-size:13px;'>
                    <b style='color:#856404;'>{Lang.WARN_REGION_FALLBACK}</b><br><br>
                    {err_banner}<br><br>
                    <b>{Lang.COLUMN_USED}</b> <code>{REGION_NAME_PROPERTY}</code><br><br>
                    <small style='color:#666;'>— Fallback: {regions}</small>
                </div>
                """))
        box, self.chk_regions = self._create_checkbox_grid(regions, Lang.SECTION_REGIONS, bg_color='#fdf7e8')
        self.w_region_box.children = box.children
        import datetime
        now = datetime.datetime.now()
        periods = []
        for periodicity in self.periodicity_active:
            periods.extend(
                period.key
                for period in list_periods(
                    self.years, periodicity, include_current=False, now=now
                )
            )
        periods = sorted(set(periods), reverse=True)
        box, self.chk_periods = self._create_checkbox_grid(periods, Lang.SECTION_PERIODS, bg_color='#ebf5eb', columns=4)
        btn_all_per, btn_none_per, hbox_per = make_select_all_none(
            on_all=lambda _: self._toggle_periods(True),
            on_none=lambda _: self._toggle_periods(False))
        self.w_period_box.children = (*box.children, hbox_per)

        # Campaign dropdown
        try:
            campaigns = list_campaigns_gcs()
            current = CONFIG.get('campaign', 'MONITOR_01')
            val = current if current in campaigns else (campaigns[0] if campaigns else '')
            self.w_campaign.options = campaigns
            self.w_campaign.value = val
        except Exception:
            self.w_campaign.options = ['monitor_01']
            self.w_campaign.value = 'monitor_01'

    # --- REGISTRAR ---

    def _toggle_periods(self, value):
        for c in self.chk_periods:
            c.value = value

    def _on_add_click(self):
        self.plan = load_workplan()
        model = next((c.description for c in self.chk_models if c.value), None)
        regions = [c.description for c in self.chk_regions if c.value]
        periods = [c.description for c in self.chk_periods if c.value]
        task_name = self.w_task_name.value.strip()
        if not task_name:
            with self.out_msg:
                clear_output()
                display(HTML(f"<b style='color:{THEME['ERROR']};'>{Lang.ERR_NO_TASK_NAME}</b>"))
            return
        if not model or not regions or not periods:
            with self.out_msg:
                clear_output()
                display(HTML(f"<b style='color:{THEME['ERROR']};'>{Lang.ERR_NO_SELECTION}</b>"))
            return
        from M0_auth_config import _gcs_models_base
        fs = _get_fs()
        model_dir = f"{CONFIG['bucket']}/{_gcs_models_base()}/{model}"
        if not fs.exists(model_dir):
            with self.out_msg:
                clear_output()
                display(HTML(f"<b style='color:{THEME['ERROR']};'>{model}: model not found in GCS. It may have been deleted. Click 'Sync Data' to refresh the model list.</b>"))
            return
        added = 0
        skipped = 0
        gcs_ok = 0
        gcs_fail = 0
        with self.out_msg:
            clear_output(wait=True)
            display(HTML(f"<i>⏳ {Lang.SAVING_TO_GCS}</i>"))
        fs = _get_fs()
        campaign = CONFIG.get('campaign', 'MONITOR_01')
        for r in regions:
            for period in periods:
                job_id = make_job_id(model, r, period, campaign)
                if any(job['id'] == job_id for job in self.plan):
                    skipped += 1
                    continue
                job = new_job(model, r, period, task_name=task_name)
                ok = save_pending_job_to_gcs(job, fs=fs)
                if ok:
                    job['_saved'] = True
                    gcs_ok += 1
                else:
                    gcs_fail += 1
                self.plan.append(job)
                added += 1
        saved_ok = save_workplan(self.plan)
        for c in self.chk_regions + self.chk_periods:
            c.value = False
        with self.out_msg:
            clear_output()
            if added == 0 and skipped == 0:
                display(HTML(f"<b style='color:{THEME['ERROR']};'>{Lang.MSG_NO_TASKS_ADDED}</b>"))
            elif added > 0:
                msg = f"<b style='color:{THEME['SUCCESS']};'>{Lang.MSG_TASKS_ADDED.format(n=added)}</b>"
                failures = []
                if not saved_ok:
                    failures.append(Lang.FAIL_SAVE_LOCAL)
                if gcs_fail > 0:
                    failures.append(Lang.FAIL_GCS_UPLOAD.format(n=gcs_fail))
                if failures:
                    msg += f"<br><span style='color:{THEME['ERROR']};'>⚠ {'; '.join(failures)}.</span>"
                if gcs_ok > 0:
                    msg += f"<br><span style='color:{THEME['SUCCESS']};'>{Lang.MSG_GCS_OK.format(n=gcs_ok)}</span>"
                if skipped > 0:
                    msg += f"<br><span style='color:{THEME['WARNING']};'>{Lang.MSG_SKIPPED_QUEUED.format(n=skipped)}</span>"
                display(HTML(msg))
            else:
                display(HTML(f"<b style='color:{THEME['WARNING']};'>{Lang.WARN_ALREADY_QUEUED.format(n=skipped)}</b>"))
        self._refresh_ui()

    def _on_campaign_change(self, change):
        campaign = change['new']
        if campaign and campaign != Lang.LOADING_CAMPAIGNS:
            CONFIG['campaign'] = campaign
            self._refresh_ui()

    # --- DELECAO ---

    def _delete_tiles(self, tile_fqpaths, fs):
        from M_gcs import exists, rm
        for fp in tile_fqpaths:
            try:
                if exists(fp):
                    rm(fp)
            except Exception:
                pass
        return len(tile_fqpaths)

    def _delete_job_tiles_region(self, job):
        from M5_workplan import region_path
        from M_gcs import exists, rm
        fs = _get_fs()
        r, p, m = job['region'], job['period'], job['model']
        c = job.get('campaign', '')
        reg_full = gcs_full(region_path(m, r, p, c))
        try:
            if exists(reg_full):
                rm(reg_full)
        except Exception:
            pass
        t_dir = gcs_full(classified_tiles_dir(m, c))
        tile_list = []
        try:
            tile_list = [t for t in fs.glob(f"{t_dir}/tile_{r}_{p}.tif") if not t.endswith('.aux.xml')]
        except Exception:
            pass
        try:
            tile_list += [t for t in fs.glob(f"{t_dir}/tile_{r}_*.tif") if not t.endswith('.aux.xml')]
        except Exception:
            pass
        tile_list = list(set(tile_list))
        n = self._delete_tiles(tile_list, fs)
        return n

    def _delete_job_complete(self, job):
        from M5_workplan import region_path, stats_dir
        from M_gcs import exists, rm
        r, p, m = job['region'], job['period'], job['model']
        c = job.get('campaign', '')
        n_tiles = self._delete_job_tiles_region(job)
        s_dir = gcs_full(stats_dir(m, c))
        for csv_name in ['stats_tile.csv', 'stats_region.csv']:
            csv_path = f"{s_dir}/{csv_name}"
            try:
                if exists(csv_path):
                    rm(csv_path)
            except Exception:
                pass
        return n_tiles

    def _delete_model_region(self, model, region):
        """Elimina todos jobs + tiles + mosaico + stats de (model, region)."""
        from M_gcs import exists, rm
        self.plan = load_workplan()
        jobs = [j for j in self.plan if j['model'] == model and j['region'] == region]
        total_tiles = 0
        for job in jobs:
            if job['status'] in ('COMPLETED', 'FINISHED'):
                total_tiles += self._delete_job_tiles_region(job)
        from M5_workplan import stats_dir
        campaigns = set(j.get('campaign', '') for j in jobs)
        for c in campaigns:
            s_dir = gcs_full(stats_dir(model, c))
            for csv_name in ['stats_tile.csv', 'stats_region.csv']:
                try:
                    p = f"{s_dir}/{csv_name}"
                    if exists(p):
                        rm(p)
                except Exception:
                    pass
        self.plan = [j for j in self.plan if not (j['model'] == model and j['region'] == region)]
        save_workplan(self.plan)
        return total_tiles, len(jobs)

    def _delete_model_all(self, model):
        """Elimina todo de um modelo (todas regioes)."""
        self.plan = load_workplan()
        regions = set(j['region'] for j in self.plan if j['model'] == model)
        total = 0
        total_jobs = 0
        for region in regions:
            n_tiles, n_jobs = self._delete_model_region(model, region)
            total += n_tiles
            total_jobs += n_jobs
        return total, total_jobs

    def _remove_from_plan(self, job_id, job=None):
        self.plan = load_workplan()
        if job and job.get('_saved'):
            delete_pending_job_gcs(job['model'], job['region'], job['period'])
        self.plan = [j for j in self.plan if j['id'] != job_id]
        save_workplan(self.plan)
        self._refresh_ui()

    # --- TAREAS ---

    def _tarea_section(self):
        """Retorna widget com a secao de tareas GCS para Pendientes."""
        fs = _get_fs()
        tareas = list_tareas(fs=fs)
        if not tareas:
            return widgets.HTML("")

        cards = []
        header = widgets.HTML(f"<div style='margin:10px 0 5px; padding:8px; background:#fef9e7; border:1px solid #f9e79f; border-radius:4px;'><b>{Lang.TAREAS_HEADER.format(n=len(tareas))}</b></div>")
        cards.append(header)

        for t in tareas:
            model = t.get('model', '?')
            regions = t.get('regions', [])
            periods = t.get('periods', [])
            years = sorted(set(p.split('_')[0] for p in periods))
            label = f" {Lang.TAREA_SUMMARY.format(m=model, r=', '.join(regions), p=', '.join(years))}"

            cargar_container, btn_cargar, _ = make_refresh_button('plus', _make_cargar(model, regions, periods), description=Lang.LOAD_TO_QUEUE, width='150px')
            btn_cargar.button_style = 'success'
            def _make_cargar(m, regs, pers):
                def _h():
                    campaign = CONFIG.get('campaign', 'MONITOR_01')
                    self.plan = load_workplan()
                    added = 0
                    skipped = 0
                    gcs_ok = 0
                    gcs_fail = 0
                    with self.out_msg:
                        clear_output(wait=True)
                        display(HTML(f"<i>{Lang.SAVING_TO_GCS}</i>"))
                    fs = _get_fs()
                    for r in regs:
                        for p in pers:
                            jid = make_job_id(m, r, p, campaign)
                            if any(job['id'] == jid for job in self.plan):
                                skipped += 1
                                continue
                            job = new_job(m, r, p)
                            ok = save_pending_job_to_gcs(job, fs=fs)
                            if ok:
                                job['_saved'] = True
                                gcs_ok += 1
                            else:
                                gcs_fail += 1
                            self.plan.append(job)
                            added += 1
                    saved_ok = save_workplan(self.plan)
                    with self.out_msg:
                        clear_output()
                        msg = f"<span style='color:{THEME['SUCCESS']};'>{Lang.MSG_LOADED.format(loaded=added, skipped=skipped)}</span>"
                        failures = []
                        if not saved_ok:
                            failures.append(Lang.FAIL_SAVE_LOCAL)
                        if gcs_fail > 0:
                            failures.append(Lang.FAIL_GCS_UPLOAD.format(n=gcs_fail))
                        if failures:
                            msg += f"<br><span style='color:{THEME['ERROR']};'> {', '.join(failures)}.</span>"
                        display(HTML(msg))
                    self._refresh_ui()
                return _h

            row = widgets.HBox([
                widgets.HTML(label, layout=L(width='auto')),
                cargar_container
            ], layout=L(align_items='center', margin='2px 0', padding='5px', border='1px solid #eee'))
            cards.append(row)

        return widgets.VBox(cards, layout=L(margin='0 0 15px 0'))

    # --- GRID COUNTS ---

    def _get_grid_count(self, region_name):
        """Retorna numero de celulas cim-world em uma regiao. Cache em memoria."""
        if region_name in self._grid_count_cache:
            return self._grid_count_cache[region_name]
        try:
            import ee
            if not getattr(ee.data, '_credentials', None):
                from M0_auth_config import authenticate
                authenticate()
            cim = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")
            fc = ee.FeatureCollection(CONFIG['asset_regions']).filter(ee.Filter.eq(REGION_NAME_PROPERTY, region_name))
            n = cim.filterBounds(fc.geometry()).size().getInfo()
            self._grid_count_cache[region_name] = n
            return n
        except Exception:
            return None

    # --- SCALE BAR ---

    def _nice_scale(self, width_km):
        """Escolhe um valor 'bonito' para a barra de escala (1, 2, 5, 10, 20, 50...)."""
        if width_km <= 0:
            return 1
        mag = 10 ** int(width_km ** 0.5 - 1) if width_km < 1 else 10 ** int(width_km // 10)
        mag = max(1, 10 ** (len(str(int(width_km))) - 1))
        candidates = [mag, mag * 2, mag * 5, mag * 10]
        for c in candidates:
            if c >= width_km * 0.25 and c <= width_km * 0.5:
                return c
        return candidates[0]

    def _build_scale_bar(self, img_width_px, bounds_info):
        """Retorna HTML de barra de escala baseada nos bounds."""
        w, s, e, n = bounds_info
        center_lat = (s + n) / 2
        width_km = (e - w) * 111.32
        scale_km = self._nice_scale(width_km)
        scale_px = int(img_width_px * (scale_km / width_km))
        bar = ('<div style="margin:8px 0;font-size:11px;color:#555;">'
               f'<div style="display:flex;align-items:center;gap:4px;">'
               f'<span style="border-top:2px solid #333;width:{scale_px}px;height:0;"></span>'
               f'<span>{scale_km} km</span>'
               f'<span style="margin-left:auto;">{Lang.SCALE_BAR_WIDTH.format(w=round(width_km))}</span>'
               '</div></div>')
        return bar

    # --- RENDER PENDIENTES ---

    def _render_pending(self):
        try:
            self.plan = load_workplan()
            campaign = CONFIG.get('campaign', 'MONITOR_01')
            all_jobs = [j for j in self.plan if j['status'] in ('PENDING', 'RUNNING') and (not campaign or j.get('campaign', '') == campaign)]
            filtered = self._apply_search_filter(all_jobs, self.f_pend_search)

            tarea_section = self._tarea_section()
            search_box = self.f_pend_search

            debug_info = (f'<div style="font-size:11px;color:#94a3b8;margin:0 0 6px 0;">'
                          f'{Lang.INFO_FILTER_COUNT.format(total=len(all_jobs), filtered=len(filtered))}</div>')
            debug_html = widgets.HTML(debug_info)

            if not filtered:
                pend_vbox = widgets.VBox([tarea_section, search_box, debug_html,
                    make_empty_state(Lang.NO_TASKS)])
            else:
                grouped = {}
                for j in filtered:
                    grouped.setdefault(j['model'], []).append(j)
                cards = []
                for model_name, jobs_list in grouped.items():
                    total_cells = {}
                    done_cells = {}
                    region_jobs = {}
                    card_regions = []
                    for j in jobs_list:
                        r = j['region']
                        if r not in card_regions:
                            card_regions.append(r)
                        region_jobs.setdefault(r, []).append(j)
                        if r not in total_cells:
                            total_cells[r] = 0
                            done_cells[r] = 0

                    fs = _get_fs()
                    for j in jobs_list:
                        if j['status'] == 'RUNNING' and j.get('progress', '0%') != '0%':
                            try:
                                prog = j['progress']
                                if '/' in prog:
                                    parts = prog.split('/')
                                    done_cells[j['region']] = int(parts[0])
                                    total_cells[j['region']] = int(parts[1].split(' ')[0])
                            except Exception:
                                pass

                    thumb_b64 = self._generate_thumb(model_name, size=128, regions=card_regions)
                    left_col = build_thumbnail_column(thumb_b64)

                    chk = widgets.Checkbox(
                        value=self._card_checkboxes.get(model_name, False).value if isinstance(self._card_checkboxes.get(model_name, False), widgets.Checkbox) else self._card_checkboxes.get(model_name, False),
                        description='', indent=False, layout=L(width='24px', margin='2px 6px 0 0'))
                    chk.observe(lambda change, m=model_name: self._sync_card_enabled(m, change['new']), names='value')
                    self._card_checkboxes[model_name] = chk

                    saved_count = sum(1 for j in jobs_list if j.get('_saved'))
                    total_count = len(jobs_list)
                    if saved_count == total_count and total_count > 0:
                        card_badge, badge_color = Lang.CARD_SAVED, "#2563eb"
                    elif saved_count > 0:
                        card_badge, badge_color = Lang.CARD_SAVED_PARTIAL.format(s=saved_count, t=total_count), "#ca8a04"
                    else:
                        card_badge, badge_color = Lang.CARD_TEMP, "#64748b"

                    discard_box, btn_discard, _ = make_refresh_button('trash', lambda m=model_name: self._on_discard_workplan(m), description=Lang.DISCARD_WORKPLAN, width='220px')
                    btn_discard.button_style = 'danger'

                    tarefas_assinadas = sorted(set(j.get('task_name', '') for j in jobs_list if j.get('task_name', '')))
                    task_badges = make_task_badges(tarefas_assinadas)

                    header = widgets.VBox([
                        widgets.HBox([
                            chk,
                            widgets.HTML(f"<b style='font-size:15px;color:#0f172a;'>{model_name}</b>", layout=L(margin='0 10px 0 0')),
                            widgets.HTML(f"<span style='display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;color:white;background:{badge_color};'>{card_badge}</span>", layout=L(margin='0 0 0 auto')),
                        ], layout=L(align_items='center', margin='0 0 4px 0')),
                        widgets.HTML(f"<div style='margin:2px 0 6px 28px;'>{task_badges}</div>" if task_badges else ''),
                    ], layout=L(width='auto'))

                    region_lines = []
                    for r, jbs in sorted(region_jobs.items()):
                        running = any(j['status'] == 'RUNNING' for j in jbs)
                        dot_color = '#f59e0b' if running else '#3b82f6'
                        tc = total_cells.get(r, 0)
                        dc = done_cells.get(r, 0)
                        if tc:
                            pct = min(dc / tc * 100, 100)
                            prog_str = f"{dc}/{tc}"
                        else:
                            pct = 0
                            prog_str = jbs[0].get('progress', '0%')
                        grid_n = self._get_grid_count(r)
                        grid_str = Lang.GRID_COUNT.format(n=grid_n) if grid_n else ""
                        bar_color = '#22c55e' if running else '#3b82f6'
                        bar = (f'<div style="width:100px;height:8px;background:#e2e8f0;border-radius:4px;overflow:hidden;">'
                               f'<div style="width:{pct:.0f}%;height:100%;background:{bar_color};border-radius:4px;"></div></div>')
                        row = widgets.HBox([
                            widgets.HTML(f'<span style="display:inline-block;width:10px;height:10px;background:{dot_color};border-radius:50%;margin:0 6px 0 0;"></span>'),
                            widgets.HTML(f"<span style='font-weight:600;color:#334155;width:140px;display:inline-block;'>{r}</span>"),
                            widgets.HTML(f"<span style='color:#64748b;font-size:12px;width:55px;'>{prog_str}</span>"),
                            widgets.HTML(bar, layout=L(margin='0 6px')),
                            widgets.HTML(f"<span style='color:#94a3b8;font-size:11px;'>{grid_str}</span>", layout=L(width='110px')),
                        ], layout=L(align_items='center', padding='5px 8px', margin='2px 0', border_bottom='1px solid #f1f5f9'))
                        region_lines.append(row)

                    btn_detalhes = widgets.Button(description=Lang.BTN_DETAILS_COLLAPSED, button_style='', layout=L(width='120px', height='26px', font_size='12px', padding='0 4px'))
                    actions_row = widgets.HBox([discard_box, btn_detalhes], layout=L(align_items='center', gap='8px', margin='4px 0 4px 28px'))
                    details_panel = self._build_details_panel(model_name, jobs_list)
                    details_panel.layout.display = 'none'
                    def toggle_details(b, panel=details_panel, btn=btn_detalhes):
                        if panel.layout.display == 'none':
                            panel.layout.display = 'block'
                            btn.description = Lang.BTN_DETAILS_EXPANDED
                        else:
                            panel.layout.display = 'none'
                            btn.description = Lang.BTN_DETAILS_COLLAPSED
                    btn_detalhes.on_click(toggle_details)

                    right_col = widgets.VBox([header, actions_row, details_panel] + region_lines, layout=L(flex='1', margin='0 0 0 12px'))
                    body = make_card_body(left_col, right_col)
                    cards.append(body)

                btn_all_card, btn_none_card, hbox_card = make_select_all_none(
                    on_all=lambda _: self._toggle_all_cards(True),
                    on_none=lambda _: self._toggle_all_cards(False))
                pend_vbox = widgets.VBox([tarea_section, search_box, debug_html, hbox_card] + cards)

            self.tab_pending.children = [pend_vbox]
        except Exception as e:
            import traceback
            err_html = widgets.HTML(f'<div style="color:{THEME["ERROR"]};padding:20px;">'
                                    f'<b>{Lang.ERR_RENDER_PENDING}</b><br>'
                                    f'<pre style="font-size:11px;">{traceback.format_exc()}</pre></div>')
            self.tab_pending.children = [err_html]

    def _get_model_meta(self, model_name):
        """Retorna metadata.json do modelo no GCS, com cache."""
        import json
        if model_name in self._model_meta_cache:
            return self._model_meta_cache[model_name]
        try:
            fs = _get_fs()
            meta_path = f"gs://{CONFIG['bucket']}/{gcs_models_path()}/{model_name}/metadata.json"
            if fs.exists(meta_path):
                with fs.open(meta_path, 'r') as f:
                    meta = json.load(f)
                self._model_meta_cache[model_name] = meta
                return meta
        except Exception:
            pass
        self._model_meta_cache[model_name] = None
        return None

    def _build_details_panel(self, model_name, jobs_list):
        """Retorna VBox ocultavel com metadados do modelo e resumo dos jobs."""
        meta = self._get_model_meta(model_name)
        parts = []

        # -- secao: metadados do modelo --
        if meta:
            sensors = meta.get('sensor', meta.get('sensors', []))
            if isinstance(sensors, str):
                sensors = [sensors]
            periodicities = meta.get('periodicity', meta.get('periodicities', []))
            if isinstance(periodicities, str):
                periodicities = [periodicities]
            num_input = meta.get('num_input', '?')
            n_iters = meta.get('n_iters', meta.get('n_epochs', ''))
            learning_rate = meta.get('learning_rate', meta.get('lr', ''))
            campaign = meta.get('campaign', meta.get('sampling_campaign', ''))

            rows = []
            if sensors:
                rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_SENSORS}</td>',
                             f'<td style="padding:2px 8px;">{", ".join(sensors)}</td>'))
            if periodicities:
                rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_PERIODICITY}</td>',
                             f'<td style="padding:2px 8px;">{", ".join(periodicities)}</td>'))
            rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_BANDS_INPUT}</td>',
                         f'<td style="padding:2px 8px;">{num_input}</td>'))
            if n_iters:
                rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_ITERATIONS}</td>',
                             f'<td style="padding:2px 8px;">{n_iters}</td>'))
            if learning_rate:
                rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_LEARNING_RATE}</td>',
                             f'<td style="padding:2px 8px;">{learning_rate}</td>'))
            if campaign:
                rows.append((f'<td style="padding:2px 8px;font-weight:600;">{Lang.META_CAMPAIGN}</td>',
                             f'<td style="padding:2px 8px;">{campaign}</td>'))

            html = '<table style="font-size:12px;color:#334155;border-collapse:collapse;">'
            for label, value in rows:
                html += f'<tr>{label}{value}</tr>'
            html += '</table>'
            parts.append(widgets.HTML(
                f'<div style="font-size:12px;font-weight:600;color:#0f172a;margin-bottom:4px;">{Lang.MODEL}</div>'
                f'{html}',
                layout=L(margin='4px 0')))
        else:
            parts.append(widgets.HTML(
                f"<span style='font-size:12px;color:#94a3b8;'>{Lang.NO_METADATA}</span>"))

        # -- secao: resumo dos jobs --
        regions = sorted(set(j['region'] for j in jobs_list))
        periods = sorted(set(j['period'] for j in jobs_list))
        total = len(jobs_list)

        html = f'<div style="font-size:12px;color:#334155;margin-top:8px;">'
        html += f'<div style="font-size:12px;font-weight:600;color:#0f172a;margin-bottom:4px;">{Lang.JOBS_REGISTERED}</div>'
        html += f'<div style="margin-bottom:4px;">{Lang.JOBS_SUMMARY.format(total=total, regions=len(regions), periods=len(periods))}</div>'
        html += '<table style="font-size:11px;border-collapse:collapse;width:100%;">'
        html += (f'<tr style="background:#f8fafc;">'
                 f'<th style="padding:3px 6px;text-align:left;border-bottom:1px solid #e2e8f0;">{Lang.JOB_REGION}</th>'
                 f'<th style="padding:3px 6px;text-align:left;border-bottom:1px solid #e2e8f0;">{Lang.JOB_PERIOD}</th>'
                 f'<th style="padding:3px 6px;text-align:left;border-bottom:1px solid #e2e8f0;">{Lang.JOB_STATUS}</th>'
                 f'<th style="padding:3px 6px;text-align:left;border-bottom:1px solid #e2e8f0;">{Lang.JOB_TASK}</th>'
                 f'</tr>')
        for j in sorted(jobs_list, key=lambda x: (x['region'], x['period'])):
            status = j.get('status', 'PENDING')
            color = {'PENDING': '#f59e0b', 'RUNNING': '#3b82f6',
                     'COMPLETED': '#22c55e'}.get(status, '#94a3b8')
            task = j.get('task_name', '')
            html += (f'<tr>'
                     f'<td style="padding:2px 6px;border-bottom:1px solid #f1f5f9;">{j["region"]}</td>'
                     f'<td style="padding:2px 6px;border-bottom:1px solid #f1f5f9;">{j["period"]}</td>'
                     f'<td style="padding:2px 6px;border-bottom:1px solid #f1f5f9;">'
                     f'<span style="color:{color};">{status}</span></td>'
                     f'<td style="padding:2px 6px;border-bottom:1px solid #f1f5f9;color:#64748b;">{task}</td>'
                     f'</tr>')
        html += '</table></div>'
        parts.append(widgets.HTML(html, layout=L(margin='4px 0')))

        return widgets.VBox(
            parts,
            layout=L(margin='4px 0 8px 24px', padding='8px',
                     border='1px solid #e2e8f0', border_radius='6px'))

    def _tarea_save_click(self, model):
        self.plan = load_workplan()
        regions = sorted(set(j['region'] for j in self.plan if j['model'] == model))
        periods = sorted(set(j['period'] for j in self.plan if j['model'] == model))
        save_tarea(model, regions, periods)
        with self.out_msg:
            clear_output()
            display(HTML(f"<span style='color:{THEME['SUCCESS']};'>{Lang.TAREA_SAVED.format(m=model)}</span>"))
        self._refresh_ui()

    def _tarea_delete_click(self, model):
        delete_tarea(model)
        with self.out_msg:
            clear_output()
            display(HTML(f"<span style='color:{THEME['ERROR']};'>{Lang.TAREA_DELETED.format(m=model)}</span>"))
        self._refresh_ui()

    def _on_discard_workplan(self, model_name):
        """Descarta todo o plano de trabalho de um modelo (local + GCS)."""
        self.plan = load_workplan()
        fs = _get_fs()
        removed = 0
        for j in self.plan:
            if j['model'] == model_name:
                if j.get('_saved'):
                    delete_pending_job_gcs(j['model'], j['region'], j['period'], fs=fs)
                removed += 1
        self.plan = [j for j in self.plan if j['model'] != model_name]
        save_workplan(self.plan)
        with self.out_msg:
            clear_output()
            display(HTML(f"<b style='color:{THEME['ERROR']};'>{Lang.WORKPLAN_DISCARDED.format(m=model_name, n=removed)}</b>"))
        self._refresh_ui()

    def _on_clear_click(self):
        self.plan = load_workplan()
        fs = _get_fs()
        for j in self.plan:
            if j.get('_saved'):
                delete_pending_job_gcs(j['model'], j['region'], j['period'], fs=fs)
        self.plan = []
        save_workplan(self.plan)
        with self.out_msg:
            clear_output()
            display(HTML(f"<b style='color:{THEME['ERROR']};'>{Lang.QUEUE_CLEARED}</b>"))
        self._refresh_ui()

    def _on_save_gcs_click(self, model_name):
        """Salva no GCS os jobs enabled de um card."""
        fs = _get_fs()
        self.plan = load_workplan()
        saved = 0
        skipped = 0
        for j in self.plan:
            if j['model'] == model_name and j.get('enabled', True):
                if j.get('_saved'):
                    skipped += 1
                    continue
                ok = save_pending_job_to_gcs(j, fs=fs)
                if ok:
                    j['_saved'] = True
                    saved += 1
        if saved > 0:
            save_workplan(self.plan)
        with self.out_msg:
            clear_output()
            if saved > 0:
                display(HTML(f"<span style='color:{THEME['SUCCESS']};'>{Lang.SAVED_TO_GCS.format(n=saved)}</span>"))
            if skipped > 0:
                display(HTML(f"<span style='color:{THEME['WARNING']};'>{Lang.ALREADY_SAVED.format(n=skipped)}</span>"))
        self._refresh_ui()

    def _on_dismiss_click(self, model_name):
        """Remove do m5_workplan.json os jobs não salvos de um card."""
        self.plan = load_workplan()
        kept = [j for j in self.plan
                if not (j['model'] == model_name and not j.get('_saved'))]
        removed = len(self.plan) - len(kept)
        self.plan = kept
        save_workplan(self.plan)
        with self.out_msg:
            clear_output()
            if removed > 0:
                display(HTML(f"<span style='color:{THEME['ERROR']};'>{Lang.TEMP_JOBS_REMOVED.format(n=removed)}</span>"))
            else:
                display(HTML(f"<span style='color:{THEME['WARNING']};'>{Lang.NO_TEMP_JOBS}</span>"))
        self._refresh_ui()

    def _on_delete_gcs_click(self, model_name):
        """Remove do GCS pending/ todos os jobs salvos de um card."""
        fs = _get_fs()
        self.plan = load_workplan()
        removed = 0
        for j in self.plan:
            if j['model'] == model_name and j.get('_saved'):
                ok = delete_pending_job_gcs(j['model'], j['region'], j['period'], fs=fs)
                if ok:
                    j['_saved'] = False
                    removed += 1
        if removed > 0:
            save_workplan(self.plan)
        with self.out_msg:
            clear_output()
            display(HTML(f"<span style='color:{THEME['ERROR']};'>{Lang.REMOVED_FROM_GCS.format(n=removed)}</span>"))
        self._refresh_ui()

    # --- HABILITACION DE CARDS ---

    def _sync_card_enabled(self, model_name, checked):
        """Seta enabled=True/False em todos jobs de um modelo."""
        self.plan = load_workplan()
        for j in self.plan:
            if j['model'] == model_name:
                j['enabled'] = checked
        save_workplan(self.plan)
        self._card_checkboxes[model_name].value = checked

    def _toggle_all_cards(self, value):
        for model_name in list(self._card_checkboxes.keys()):
            self._card_checkboxes[model_name].value = value
            self._sync_card_enabled(model_name, value)

    # --- LIVE MAPA (PROCESSING CALLBACK) ---

    def _on_tile_progress(self, model, region, cell_id, i, total, status):
        """Callback chamado de M5_classifier a cada tile."""
        try:
            self._processing_state = {
                'running': True,
                'model': model,
                'region': region,
                'current': cell_id,
                'total': total,
                'completed': sorted(set(self._processing_state.get('completed', []) + ([cell_id] if status in ('done', 'skipped') else []))),
                'current_i': i,
                'last_status': status,
            }
            self._render_mapa_live()
            self.tabs.selected_index = 4
        except Exception:
            pass

    def _render_mapa_live(self):
        """Renderiza Mapa con estado de procesamiento en vivo."""
        state = self._processing_state
        if not state or not state.get('running'):
            return

        model, region = state.get('model', '?'), state.get('region', '?')
        total = state.get('total', 0)
        done_set = set(state.get('completed', []))
        current = state.get('current', '')
        done_n = len(done_set)
        pct = f"{done_n}/{total} ({done_n/total:.0%})" if total else '0/0'

        lines = [f"<tr><td style='padding:2px 8px;'>{Lang.MODEL}</td><td style='padding:2px 8px;'><b>{model}</b></td></tr>",
                 f"<tr><td style='padding:2px 8px;'>{Lang.REGION}</td><td style='padding:2px 8px;'><b>{region}</b></td></tr>",
                 f"<tr><td style='padding:2px 8px;'>{Lang.PROGRESS}</td><td style='padding:2px 8px;'><b>{pct}</b></td></tr>"]
        if current:
            lines.append(f"<tr><td style='padding:2px 8px;'>{Lang.CURRENT_TILE}</td><td style='padding:2px 8px;'><span style='color:{THEME['WARNING']};font-weight:bold;'>{current}</span></td></tr>")

        # tiles completados
        if done_set:
            done_cells = sorted(done_set)
            done_html = ', '.join(done_cells[:10])
            if len(done_cells) > 10:
                done_html += f' ... (+{len(done_cells)-10} mas)'
            lines.append(f"<tr><td style='padding:2px 8px;'>{Lang.COMPLETED}</td><td style='padding:2px 8px;color:{THEME['SUCCESS']};'>{done_html}</td></tr>")

        html = (f'<div style="margin:8px 0;padding:10px;border:2px solid {THEME["WARNING"]};border-radius:6px;'
                f'background:#fef9e7;">'
                f'<b style="color:{THEME["WARNING"]};">{Lang.LIVE_PROCESSING}</b>'
                f'<table style="font-size:12px;color:#555;border-collapse:collapse;margin-top:6px;">'
                f'{"".join(lines)}</table></div>')

        with self._live_status_out:
            clear_output()
            display(HTML(html))

    # --- RENDER PUBLISH ---

    def _render_publish(self):
        self.plan = load_workplan()
        campaign = CONFIG.get('campaign', 'MONITOR_01')
        jobs = [j for j in self.plan if j['status'] == 'COMPLETED' and (not campaign or j.get('campaign', '') == campaign)]
        filter_box = widgets.HBox([self.f_pub_model, self.f_pub_region, self.f_pub_year, self.f_pub_task], layout=L(margin='0 0 15px 0'))
        filtered = self._apply_filters(jobs, self.f_pub_model, self.f_pub_region, self.f_pub_year, self.f_pub_task)

        if not filtered:
            self.w_pub_rows.children = [make_empty_state(Lang.NO_TASKS_PUBLISH)]
        else:
            grouped = {}
            for j in filtered:
                grouped.setdefault(j['model'], []).append(j)
            cards = []
            for model_name, jobs_list in grouped.items():
                card_regions = sorted(set(j['region'] for j in jobs_list))

                # -- thumb 128px esquerda --
                thumb_b64 = self._generate_thumb(model_name, size=128, regions=card_regions)
                left_col = build_thumbnail_column(thumb_b64)

                # -- tarefas badges --
                tarefas_assinadas = sorted(set(j.get('task_name', '') for j in jobs_list if j.get('task_name', '')))
                task_badges = make_task_badges(tarefas_assinadas)

                # -- cabecalho direito --
                header = widgets.VBox([
                    widgets.HTML(f"<b style='font-size:15px;color:#0f172a;'>{model_name}</b>",
                                 layout=L(margin='0 0 4px 0')),
                    widgets.HTML(f"<div style='margin:0 0 6px 0;'>{task_badges}</div>" if task_badges else ''),
                ], layout=L(width='auto', margin='0 0 8px 0'))

                # -- linhas de trabalho --
                rows = []
                for job in jobs_list:
                    tile_out = widgets.VBox([])
                    btn_tiles = self._build_tile_expander(job, tile_out)
                    chk_gee = widgets.Checkbox(value=job.get('upload_gee', False), description=f"{job['region']} | {job['period']}", style={'description_width': 'initial'}, layout=L(width='auto', max_width='100%'))
                    chk_gee.observe(lambda change, jid=job['id']: self._toggle_gee(change, jid), names='value')
                    lbl_status = widgets.HTML(f"<span style='color:#16a34a;font-weight:700;font-size:12px;'>{job['status']}</span>")
                    btn_del_all = widgets.Button(description='', icon='trash', button_style='danger', layout=L(width='32px', height='26px', padding='0'))
                    top = widgets.HBox([chk_gee, lbl_status, btn_tiles, btn_del_all],
                                       layout=L(align_items='center', padding='5px 8px', margin='2px 0',
                                                border_bottom='1px solid #f1f5f9'))
                    btn_del_all.on_click(lambda b, jb=job, c=top: inline_confirm(b, lambda: (self._delete_job_tiles_region(jb), self._remove_from_plan(jb['id'], job=jb), self._refresh_ui()), container=c))
                    rows.append(widgets.VBox([top, tile_out]))

                right_col = widgets.VBox([header] + rows, layout=L(flex='1', margin='0 0 0 12px'))
                cards.append(make_card_body(left_col, right_col))
            self.w_pub_rows.children = cards
        self.tab_publish.children = [filter_box, self.w_pub_rows]

    def _build_tile_expander(self, job, tiles_container):
        btn = widgets.Button(description=Lang.VIEW_TILES, icon='list', layout=L(width='110px', height='28px'), button_style='info')
        expanded = [False]
        def _toggle(_):
            if not expanded[0]:
                expanded[0] = True
                btn.description = Lang.HIDE_TILES
                btn.button_style = ''
                tiles_container.children = [widgets.HTML(f"<i>{Lang.LOADING_TILES}</i>")]
                self._refresh_tile_list(job, tiles_container)
            else:
                expanded[0] = False
                btn.description = Lang.VIEW_TILES
                btn.button_style = 'info'
                tiles_container.children = []
        btn.on_click(_toggle)
        return btn

    def _refresh_tile_list(self, job, container):
        fs = _get_fs()
        r, p, m = job['region'], job['period'], job['model']
        c = job.get('campaign', '')
        t_dir = gcs_full(classified_tiles_dir(m, c))
        tile_fqpaths = []
        try:
            tile_fqpaths = sorted([t for t in fs.glob(f"{t_dir}/tile_{r}_{p}.tif") if not t.endswith('.aux.xml')])
        except Exception:
            pass
        if not tile_fqpaths:
            tile_fqpaths = []
            try:
                tile_fqpaths = sorted([t for t in fs.glob(f"{t_dir}/tile_{r}_*.tif") if not t.endswith('.aux.xml')])
            except Exception:
                pass
        if not tile_fqpaths:
            container.children = [widgets.HTML(f"<span style='color:#999; font-size:11px;'>{Lang.NO_TILES_GCS_ALT}</span>")]
            return
        tile_chks = []
        for fp in tile_fqpaths:
            tname = os.path.basename(fp)
            chk = widgets.Checkbox(value=False, description=tname, indent=False, style={'description_width': 'initial'}, layout=L(width='auto', margin='0'))
            tile_chks.append(chk)
        chk_box = widgets.VBox(tile_chks, layout=L(margin='5px 0 5px 20px'))
        btn_del_sel = widgets.Button(description=Lang.DELETE_SELECTED, button_style='danger', layout=L(width='180px', height='26px'))
        def _del_sel(_):
            to_rm = [fp for fp, chk in zip(tile_fqpaths, tile_chks) if chk.value]
            if to_rm:
                n = self._delete_tiles(to_rm, fs)
                with self.out_msg:
                    clear_output()
                    display(HTML(f"<span style='color:{THEME['ERROR']};'>{Lang.TILES_DELETED.format(n=n)}</span>"))
                self._refresh_tile_list(job, container)
            else:
                with self.out_msg:
                    clear_output()
                    display(HTML(f"<span style='color:{THEME['WARNING']};'>{Lang.SELECT_TILES_PROMPT}</span>"))
        btn_del_sel.on_click(_del_sel)
        btn_del_all = widgets.Button(description=Lang.DELETE_ALL, button_style='warning', layout=L(width='130px', height='26px'))
        def _del_all(_):
            n = self._delete_tiles(tile_fqpaths, fs)
            with self.out_msg:
                clear_output()
                display(HTML(f"<span style='color:{THEME['ERROR']};'>{Lang.TILES_DELETED.format(n=n)}</span>"))
            self._refresh_tile_list(job, container)
        btn_del_all.on_click(_del_all)
        actions = widgets.HBox([btn_del_sel, btn_del_all], layout=L(margin='3px 0 0 20px'))
        container.children = [chk_box, actions]

    # --- RENDER MAPA ---

    def _render_mapa(self):
        self.plan = load_workplan()
        campaign = CONFIG.get('campaign', 'MONITOR_01')
        filter_box = widgets.HBox([self.f_mapa_model, self.f_mapa_region, self.f_mapa_year, self.btn_mapa_refresh_container], layout=L(margin='0 0 15px 0'))

        try:
            import ee
            import requests
            if not getattr(ee.data, '_credentials', None):
                from M0_auth_config import authenticate
                authenticate()

            all_regions = ee.FeatureCollection(CONFIG['asset_regions'])
            grid = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")

            region_lines = all_regions.style(**{'color': '2980b9', 'width': 1, 'fillColor': '00000000'})
            grid_lines = grid.style(**{'color': 'bdc3c7', 'width': 0.3, 'fillColor': '00000000'})

            overlay = ee.ImageCollection([region_lines, grid_lines]).mosaic()

            region_val = self.f_mapa_region.value
            if region_val and region_val != Lang.ALL_F:
                sel = all_regions.filter(ee.Filter.eq(REGION_NAME_PROPERTY, region_val))
                sel_style = sel.style(**{'color': 'e74c3c', 'width': 2, 'fillColor': 'e74c3c20'})
                overlay = ee.ImageCollection([overlay, sel_style]).mosaic()

            # bounds para scale bar
            country_bounds = all_regions.geometry().bounds(1, 'EPSG:4326')
            bounds_coords = country_bounds.getInfo()['coordinates'][0]
            xs = [c[0] for c in bounds_coords]
            ys = [c[1] for c in bounds_coords]
            bounds_info = (min(xs), min(ys), max(xs), max(ys))

            img_w = 600
            url = overlay.getThumbURL({'region': country_bounds, 'dimensions': img_w, 'format': 'png'})
            resp = requests.get(url, timeout=60)
            b64 = base64.b64encode(resp.content).decode('ascii')
            scale_html = self._build_scale_bar(img_w, bounds_info)

            # grid counts table
            campaign = CONFIG.get('campaign', 'MONITOR_01')
            region_names = sorted(set(j['region'] for j in self.plan if not campaign or j.get('campaign', '') == campaign))
            if not region_names:
                try:
                    region_names = all_regions.aggregate_array(REGION_NAME_PROPERTY).distinct().getInfo()
                    region_names = sorted([n for n in region_names if n])
                except Exception:
                    region_names = []
            count_rows = ""
            for rn in region_names:
                n = self._get_grid_count(rn)
                if n is not None:
                    count_rows += f"<tr><td style='padding:2px 8px;'>{rn}</td><td style='padding:2px 8px;text-align:right;'>{n}</td></tr>"
            grid_table = ""
            if count_rows:
                grid_table = ('<div style="margin:8px 0;"><table style="font-size:12px;color:#555;border-collapse:collapse;">'
                              '<tr><th style="padding:2px 8px;border-bottom:1px solid #ccc;text-align:left;">'
                              f'{Lang.GRID_REGION}</th>'
                              f'<th style="padding:2px 8px;border-bottom:1px solid #ccc;text-align:right;">'
                              f'{Lang.GRID_CELLS}</th></tr>'
                              f'{count_rows}</table></div>')

            legenda = f"""
            <div style="display:flex;gap:15px;margin:10px 0;font-size:12px;color:#555;">
                <span><span style="display:inline-block;width:12px;height:12px;background:#2c3e50;border-radius:2px;"></span> {Lang.MAP_LEGEND_COUNTRY}</span>
                <span><span style="display:inline-block;width:12px;height:12px;background:#2980b9;border-radius:2px;"></span> {Lang.MAP_LEGEND_REGIONS}</span>
                <span><span style="display:inline-block;width:12px;height:12px;background:#bdc3c7;border-radius:2px;"></span> {Lang.MAP_LEGEND_GRID}</span>
            </div>
            """
            img_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; border:1px solid #ccc; border-radius:4px;">'
            self.w_mapa_rows.children = [widgets.HTML(img_html + scale_html + legenda + grid_table), self._live_status_out]
        except Exception:
            self.w_mapa_rows.children = [make_empty_state(Lang.NO_MAP), self._live_status_out]

        self.tab_mapa.children = [filter_box, self.w_mapa_rows]

    # --- RENDER DONE ---

    def _render_done(self):
        self.plan = load_workplan()
        campaign = CONFIG.get('campaign', 'MONITOR_01')
        jobs = [j for j in self.plan if j['status'] == 'FINISHED' and (not campaign or j.get('campaign', '') == campaign)]
        filter_box = widgets.HBox([self.f_done_model, self.f_done_region, self.f_done_year, self.f_done_task], layout=L(margin='0 0 15px 0'))
        filtered = self._apply_filters(jobs, self.f_done_model, self.f_done_region, self.f_done_year, self.f_done_task)

        if not filtered:
            self.w_done_rows.children = [make_empty_state(Lang.NO_TASKS_DONE)]
        else:
            grouped = {}
            for j in filtered:
                grouped.setdefault(j['model'], []).append(j)
            cards = []
            all_periods = self._get_all_periods()

            for model_name, jobs_list in grouped.items():
                region_jobs = {}
                card_regions = []
                for j in jobs_list:
                    r = j['region']
                    if r not in card_regions:
                        card_regions.append(r)
                    region_jobs.setdefault(r, []).append(j)

                # -- thumb 128px esquerda --
                thumb_b64 = self._generate_thumb(model_name, size=128, regions=card_regions)
                left_col = build_thumbnail_column(thumb_b64)

                # -- tarefas badges --
                tarefas_assinadas = sorted(set(j.get('task_name', '') for j in jobs_list if j.get('task_name', '')))
                task_badges = make_task_badges(tarefas_assinadas)

                # -- cabecalho direito --
                region_lines = []
                for r, jbs in sorted(region_jobs.items()):
                    timelines = []
                    for year, months in sorted(all_periods.items()):
                        tl = self._build_year_line(year, months, jbs)
                        if tl.strip():
                            timelines.append(tl)
                    tl_html = "<div style='overflow-x:auto;'>" + "".join(timelines) + "</div>"

                    btn_del_region = widgets.Button(description='', icon='trash', button_style='danger',
                                                    layout=L(width='32px', height='26px', padding='0'))

                    line = widgets.HBox([
                        widgets.HTML(f"<b style='width:120px;color:#334155;'>{r}</b>", layout=L(margin='0 8px 0 0')),
                        widgets.HTML(tl_html, layout=L(margin='0 8px 0 0', flex='1')),
                        btn_del_region
                    ], layout=L(align_items='center', padding='5px 8px', margin='2px 0',
                                border_bottom='1px solid #f1f5f9'))
                    btn_del_region.on_click(lambda b, m=model_name, rg=r, c=line: inline_confirm(b, lambda: (self._delete_model_region(m, rg), self._refresh_ui()), container=c))
                    region_lines.append(line)

                btn_del_model = widgets.Button(description=Lang.DELETE_MODEL, button_style='danger', layout=L(width='150px', height='28px'))

                header = widgets.VBox([
                    widgets.HBox([
                        widgets.HTML(f"<b style='font-size:15px;color:#0f172a;'>{model_name}</b>",
                                     layout=L(margin='0 0 4px 0')),
                    ]),
                    widgets.HTML(f"<div style='margin:0 0 6px 0;'>{task_badges}</div>" if task_badges else ''),
                ], layout=L(width='auto', margin='0 0 8px 0'))

                right_col = widgets.VBox([
                    header,
                    widgets.VBox(region_lines, layout=L(margin='0 0 6px 0')),
                    btn_del_model,
                ], layout=L(flex='1', margin='0 0 0 12px'))
                btn_del_model.on_click(lambda b, m=model_name, c=right_col: inline_confirm(b, lambda: (self._delete_model_all(m), self._refresh_ui()), container=c))

                card = make_card_body(left_col, right_col, border_color='#bbf7d0', background='#f0fdf4')
                cards.append(card)

            self.w_done_rows.children = cards
        self.tab_done.children = [filter_box, self.w_done_rows]

    # --- MANIPULADORES ---

    def _toggle_gee(self, change, job_id):
        if 'new' in change:
            self.plan = load_workplan()
            for j in self.plan:
                if j['id'] == job_id:
                    j['upload_gee'] = change['new']
            save_workplan(self.plan)

    def _apply_filters(self, jobs, f_model, f_region, f_year, f_task=None):
        if f_model.value != Lang.ALL:
            jobs = [j for j in jobs if j['model'] == f_model.value]
        if f_region.value != Lang.ALL_F:
            jobs = [j for j in jobs if j['region'] == f_region.value]
        if f_year.value != Lang.ALL:
            jobs = [j for j in jobs if j['period'].split('_')[0] == f_year.value]
        if f_task is not None and f_task.value != Lang.ALL_F:
            jobs = [j for j in jobs if j.get('task_name', '') == f_task.value]
        return jobs

    def _apply_search_filter(self, jobs, search):
        q = search.value.strip().lower()
        if not q:
            return jobs
        return [j for j in jobs if q in j['id'].lower() or q in j['model'].lower()
                or q in j['region'].lower() or q in j['period'].lower()
                or q in j.get('task_name', '').lower()]

    # --- REFRESH ---

    def _sync_data(self):
        from M_cache import CacheManager
        CacheManager.build_full_cache()
        self._populate_dropdowns()
        self._refresh_ui()

    def _refresh_ui(self):
        self.plan = load_workplan()

        def _safe_update(f, new_ops):
            if not hasattr(f, 'options') or not new_ops:
                return
            old = f.value
            f.options = new_ops
            if old in new_ops:
                f.value = old
            else:
                f.value = new_ops[0]

        campaign = CONFIG.get('campaign', 'MONITOR_01')
        # Mapa filters
        filtered_plan = [j for j in self.plan if not campaign or j.get('campaign', '') == campaign]
        all_models = sorted(set(j['model'] for j in filtered_plan))
        all_regions = sorted(set(j['region'] for j in filtered_plan))
        all_years = sorted(set(j['period'].split('_')[0] for j in filtered_plan), reverse=True)
        for f_m, f_r, f_y in [(self.f_mapa_model, self.f_mapa_region, self.f_mapa_year)]:
            _safe_update(f_m, [Lang.ALL] + all_models)
            _safe_update(f_r, [Lang.ALL_F] + all_regions)
            _safe_update(f_y, [Lang.ALL] + all_years)

        self._render_pending()
        self._render_mapa()

        # Update tab title counter
        n_pend = len([j for j in self.plan if j['status'] in ('PENDING', 'RUNNING') and (not campaign or j.get('campaign', '') == campaign)])
        if hasattr(self, 'tabs'):
            self.tabs.set_title(2, f"{Lang.TAB_PENDING} ({n_pend})")

    def display(self):
        # Sincroniza jobs do GCS pendente com a fila local
        n_sync = sync_gcs_to_local_workplan()
        if n_sync > 0:
            with self.out_msg:
                clear_output()
                display(HTML(f"<span style='color:{THEME['SUCCESS']};'>{Lang.SYNC_FROM_GCS.format(n=n_sync)}</span>"))
        self._build_guide()
        self._refresh_ui()

        form = widgets.VBox([
            self.w_model_box, self.w_region_box, self.w_period_box,
            widgets.VBox([self.w_campaign, self.w_task_name], layout=L(margin='15px 0 5px 0')),
            widgets.HBox([self.btn_add_container], layout=L(margin='5px 0 10px 0', align_items='center')),
            self.out_msg
        ], layout=L(padding='20px'))

        self.tabs.children = [self.w_guide, form, self.tab_pending, self.tab_mapa]
        self.tabs.set_title(0, Lang.TAB_GUIDE)
        self.tabs.set_title(1, Lang.TAB_REGISTER)

        campaign = CONFIG.get('campaign', 'MONITOR_01')
        n_pend = len([j for j in self.plan if j['status'] in ('PENDING', 'RUNNING') and (not campaign or j.get('campaign', '') == campaign)])
        self.tabs.set_title(2, f"{Lang.TAB_PENDING} ({n_pend})")
        self.tabs.set_title(3, Lang.TAB_MAP)

        header_actions = widgets.HBox([
            widgets.HTML(f"<b style='color:#2c3e50; font-size:14px; margin-right:15px;'>{Lang.GLOBAL_ACTIONS}:</b>"),
            self.btn_sync
        ], layout=L(margin='0 0 15px 0', align_items='center', padding='10px', border='1px solid #e0e0e0', background_color='#fcfcfc', border_radius='5px'))

        display(widgets.VBox([header_actions, self.tabs]))


def run_m5_ui(years=None, periodicity_active=None):
    ui = M5WorkplanUI(years=years, periodicity_active=periodicity_active)
    ui.display()
    return ui
