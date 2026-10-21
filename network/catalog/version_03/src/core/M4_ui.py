import os
import json
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from M0_auth_config import CONFIG, GLOBAL_OPTS, gcs_path, model_path
from M_cache import CacheManager, _get_fs
from M_ui_components import PipelineStepUI, cell_log, make_spinner, Layout, make_empty_state, make_select_all_none, make_search_box, make_sync_button
from M_lang import L as Lang

from M4_data_extractor import extract_pixels_from_gcs, list_sample_collections_gcs, list_campaigns_gcs
from M4_algorithms_dnn import ModelTrainer, _get_tf
from M4_analytics import view_analytics, render_diagnostic_dashboard, render_model_card_html
from M4_hub_manager import list_trained_models, register_model, _load_m4_metadata, _save_m4_metadata
class ModelTrainerUI(PipelineStepUI):
    def __init__(self):
        super().__init__(
            title=Lang.M4_HEADER_TITLE, 
            description=Lang.CANVAS_TITLE
        )
        self.trainer_instance = None
        self.chk_dict = {}  # legacy
        self._selected_samples = set()
        self.search_query_samples = ""
        self.search_query_models = "" 
        self.sort_column = "acc"      
        self.sort_ascending = False   
        self.sampling_campaign = CONFIG.get('campaign', 'MONITOR_01')
        
        # --- INTENÇÃO DE RETREINAMENTO ---
        self.retrain_intent = {'mode': None, 'hp': None} # Guarda a intenção atual de re-treinamento
        self.cb_retrain = widgets.Checkbox(value=False, layout={'display': 'none'})
        self.cb_reextract = widgets.Checkbox(value=False, layout={'display': 'none'})
        self.cb_borrar_retrain = widgets.Checkbox(value=False, layout={'display': 'none'})
        self.cb_retrain.observe(self._on_intent_cb_change, names='value')
        self.cb_reextract.observe(self._on_intent_cb_change, names='value')
        self.cb_borrar_retrain.observe(self._on_intent_cb_change, names='value')
        
        # --- ESTADO DEL CANVAS ---
        self.selected_models = {} # ID -> info (Active in Canvas)
        self.canvas_history = {} # ID -> info (Ever viewed in session)
        self.canvas_search_query = ""
        self.canvas_sort_col = "acc"
        self.canvas_sort_asc = False
        self.canvas_output = widgets.Output(layout=widgets.Layout(background_color='white', padding='20px'))
        self.analytics_dashboard_output = widgets.Output() # Para carregar card após treino
        self._live_plots_out = widgets.Output()            # Para evitar "piscar" no treino
        self.canvas_slider_val = 0
        self.band_chk_map = {} # (sensor, mosaic, band) -> checkbox
        
        # --- CONFIGURACIÓN DE VISIBILIDAD ---
        self.viz_config = {
            'title': True, 'scores': True, 'cm': True, 'history': True, 
            'prob': True, 'pr': True, 
            'pca2d': False, 'pca3d': False,
            'pca3d_static': False,
            'management': False
        }
        
        # --- WIDGETS DO CANVAS (CONTROLES GLOBAIS) ---
        self.w_global_slider = widgets.IntSlider(
            value=0, min=0, max=10, description=Lang.HP_EPOCHS,
            layout=widgets.Layout(width='98%', margin='5px 0 15px 0'),
            style={'description_width': 'initial'}
        )
        self.w_global_slider.observe(self._on_global_slider_change, names='value')
        
        self.w_apply_btn = widgets.Button(
            description=Lang.APPLY_VISIBILITY, icon="play",
            button_style='success', layout=widgets.Layout(width='180px')
        )
        self.w_apply_btn.on_click(lambda _: self._update_canvas())
        
        # Sidebar containers
        self.canvas_available_box = widgets.VBox([], layout=widgets.Layout(flex='1', border='1px solid #ddd', overflow_y='auto'))
        self.canvas_selected_box = widgets.VBox([], layout=widgets.Layout(flex='1', border='1px solid #ddd', overflow_y='auto'))
        
        self.main_area.children = [widgets.HTML(f"<i>{Lang.LOADING_INTERFACE}</i>")]

    def _load_config_into_widgets(self, hp):
        """Carrega os parâmetros de um modelo de volta para os widgets de configuração."""
        self.w_training_id.value = hp.get('training_id', '')
        self.w_shortname.value = hp.get('shortname', '')
        self.w_layers.value = ",".join(map(str, hp.get('layers', [64, 32])))
        self.w_lr.value = str(hp.get('lr', 0.001))
        self.w_iters.value = str(hp.get('n_iters', 5000))
        self.w_batch.value = str(hp.get('batch_size', 1000))
        self.w_comment.value = hp.get('comment', '')
        
        # Muestras
        sc = hp.get('sample_collections', [])
        self._selected_samples = set(sc)
            
        # Bandas
        b_cfg = hp.get('bands_config', {})
        for (s, m, p, b), chk in self.band_chk_map.items():
            chk.value = (b in b_cfg and b_cfg[b]['sensor'] == s and b_cfg[b]['mosaic'] == m)

        # Sync Intent Checkboxes
        mode = self.retrain_intent.get('mode')
        # Temporarily unobserve to avoid feedback loops
        for cb in [self.cb_retrain, self.cb_reextract, self.cb_borrar_retrain]:
            cb.unobserve(self._on_intent_cb_change, names='value')
            cb.value = False
            
        if mode == 'retrain': self.cb_retrain.value = True
        elif mode == 're-extract': self.cb_reextract.value = True
        elif mode == 'borrar': self.cb_borrar_retrain.value = True
        
        for cb in [self.cb_retrain, self.cb_reextract, self.cb_borrar_retrain]:
            cb.observe(self._on_intent_cb_change, names='value')

    def _on_intent_cb_change(self, change):
        """Ensures exclusivity between retraining intent checkboxes."""
        if not change['new']: return
        owner = change['owner']
        for cb in [self.cb_retrain, self.cb_reextract, self.cb_borrar_retrain]:
            if cb != owner:
                cb.unobserve(self._on_intent_cb_change, names='value')
                cb.value = False
                cb.observe(self._on_intent_cb_change, names='value')
        mode = 'retrain' if self.cb_retrain.value else \
               're-extract' if self.cb_reextract.value else \
               'borrar' if self.cb_borrar_retrain.value else None
        self.retrain_intent['mode'] = mode

    def display(self):
        # Ensure locale matches GLOBAL_OPTS
        lang = GLOBAL_OPTS.get('LANGUAGE', 'en')
        Lang.load_locale(lang)
        # 1. NOVO TREINO (Fluxo Completo)
        hp_sec = self._build_hp_section()
        dest_sec = self._build_dest_section()
        
        # Build areas sem fazer chamadas ao GCS (usando cache ou vazio)
        self.samples_area = self._build_matrix()
        self.extraction_area = self._build_extraction_matrix()
        
        self.new_training_tab = widgets.VBox([
            widgets.HTML(f"<h3 style='color:#2c3e50; margin:0 0 5px 0;'>{Lang.SAMPLE_SELECTION}</h3>"),
            self.samples_area,
            widgets.HTML(f"<h3 style='color:#2c3e50; margin:15px 0 5px 0;'>{Lang.EXTRACTION_TITLE}</h3>"),
            self.extraction_area,
            widgets.HTML(f"<h3 style='color:#2c3e50; margin:15px 0 5px 0;'>{Lang.MODEL_CONFIG}</h3>"),
            hp_sec,
            widgets.HTML(f"<h3 style='color:#2c3e50; margin:15px 0 5px 0;'>{Lang.GCS_DEST}</h3>"),
            dest_sec,
        ], layout=widgets.Layout(padding='10px', background_color='white'))
        
        # 2. CANVAS (Visualização + Ranking Sidebar)
        # --- SIDEBAR (ESQUERDA) ---
        self.w_canvas_search = widgets.Text(placeholder=Lang.SEARCH_REPO, layout=widgets.Layout(width='100%'))
        self.w_canvas_search.observe(lambda c: self._on_canvas_search_change(c['new']), names='value')
        
        self.w_canvas_sort = widgets.Dropdown(
            options=[(Lang.ACCURACY, 'acc'), (Lang.F1_SCORE, 'f1'), (Lang.ID, 'id')],
            value=self.canvas_sort_col,
            description=Lang.SORT_BY,
            layout=widgets.Layout(width='100%'),
            style={'description_width': '60px'}
        )
        def _on_sort_change(change):
            self.canvas_sort_col = change['new']
            self._refresh_canvas_hub()
        self.w_canvas_sort.observe(_on_sort_change, names='value')

        btn_all_canvas = widgets.Button(description=Lang.ALL, icon="check-square", layout=widgets.Layout(width='48%'), button_style='info')
        btn_none_canvas = widgets.Button(description=Lang.CLEAR, icon="square-o", layout=widgets.Layout(width='48%'), button_style='warning')
        btn_all_canvas.on_click(lambda _: self._on_canvas_batch_action('all'))
        btn_none_canvas.on_click(lambda _: self._on_canvas_batch_action('none'))
        
        sidebar_vbox = widgets.VBox([
            widgets.HTML(f"<b style='font-size:13px; color:#2c3e50;'> {Lang.REPO_TITLE}</b>"),
            self.w_canvas_search,
            self.w_canvas_sort,
            self.canvas_available_box,
            widgets.HBox([btn_all_canvas, btn_none_canvas], layout=widgets.Layout(justify_content='space-between', margin='5px 0')),
            widgets.HTML(f"<b style='font-size:13px; color:#2c3e50; margin-top:10px;'> {Lang.SELECTED_CANVAS}</b>"),
            self.canvas_selected_box
        ], layout=widgets.Layout(width='320px', padding='10px', background_color='#fcfcfc', border_right='2px solid #eee'))

        main_canvas_vbox = widgets.VBox([
            self._build_viz_toolbar(), 
            self.w_global_slider,      
            self.canvas_output         
        ], layout=widgets.Layout(flex='1', padding='15px'))

        self.canvas_area = widgets.HBox([sidebar_vbox, main_canvas_vbox], 
                                       layout=widgets.Layout(background_color='white', border='1px solid #ddd', min_height='800px'))

        # 3. ASSEMBLY TABS
        self.tab = widgets.Tab()
        self.tab.children = [
            self._build_guide_tab(),
            self.new_training_tab,
            self.canvas_area,
        ]
        self.tab.set_title(0, f' {Lang.USAGE_GUIDE}')
        self.tab.set_title(1, f' {Lang.NEW_TRAINING}')
        self.tab.set_title(2, f' {Lang.TRAININGS}')
        
        self.tab.selected_index = 0

        # Global sync header
        btn_sync_all, _ = make_sync_button(Lang.SYNC_DATA, "refresh", self._sync_all,
            width='auto', button_style='success', ui=self)
        sync_header = widgets.HBox([
            widgets.HTML(f"<b style='font-size:16px; color:#2c3e50;'>{Lang.M4_HEADER_TITLE}</b>"),
            widgets.HTML("<div style='flex:1;'></div>"),
            btn_sync_all
        ], layout=widgets.Layout(align_items='center', margin='0 0 5px 0'))
        
        self.main_area.children = [sync_header, self.tab]
        super().display()

        # Startup refresh: use the cache when possible; never wipe it here.
        # Full rebuilds stay behind the explicit "Sync Data" button.
        try:
            self._sync_repository(show_loader=False, force_refresh=False)
        except Exception as e:
            print(f"  [WARN] Startup refresh: {e}")

    def _build_guide_tab(self):
        return widgets.HTML(Lang.GUIDE_M4_HTML.format(
            usage_guide=Lang.USAGE_GUIDE,
            new_training=Lang.NEW_TRAINING,
            trainings=Lang.TRAININGS
        ))

    def _populate_pane(self, pane, items, action):
        """Fill a pane with clickable sample buttons."""
        btns = []
        for s in items:
            bg = '#e3f2fd' if action == 'remove' else '#f8f9fa'
            btn = widgets.Button(description=s, layout=Layout(width='100%', min_height='26px', margin='1px 0'),
                style={'button_color': bg})
            if action == 'add':
                btn.on_click(lambda _, name=s: (self._selected_samples.add(name), self._refresh_panes()))
            else:
                btn.on_click(lambda _, name=s: (self._selected_samples.discard(name), self._refresh_panes()))
            btns.append(btn)
        pane.children = btns

    def _refresh_panes(self):
        """Redraw available and selected panes."""
        q = (self.txt_search_samples.value or '').lower()
        available = sorted([s for s in self._available_samples if q in s.lower() and s not in self._selected_samples])
        selected = sorted(self._selected_samples)
        self._populate_pane(self.available_pane, available, 'add')
        self._populate_pane(self.selected_pane, selected, 'remove')
        self._update_shortname()

    def _select_all_samples(self, _):
        q = (self.txt_search_samples.value or '').lower()
        for s in self._available_samples:
            if q in s.lower():
                self._selected_samples.add(s)
        self._refresh_panes()

    def _clear_all_samples(self, _):
        self._selected_samples.clear()
        self._refresh_panes()

    def _update_shortname(self):
        """Update the shortname widget based on selected samples and bands."""
        import re
        sel = sorted(getattr(self, '_selected_samples', set()))
        if not sel:
            short = CONFIG.get('campaign', 'MONITOR_01').lower()
        else:
            regions = set()
            for s in sel:
                m = re.search(r'_r(\d+)_', s)
                if m:
                    regions.add(f"r{m.group(1)}")
            if not regions:
                short = f"{CONFIG.get('country', 'fire')}_v1"
            else:
                region_part = '_'.join(sorted(regions)) if len(regions) == 1 else 'multiregion'
                n_bands = 4
                if hasattr(self, 'band_chk_map') and self.band_chk_map:
                    n_bands = sum(1 for chk in self.band_chk_map.values() if chk.value)
                    if n_bands == 0:
                        n_bands = 4
                short = f"{region_part}_{n_bands}b"
        if hasattr(self, 'w_shortname'):
            self.w_shortname.value = short

    def _build_matrix(self):
        L = widgets.Layout
        css = PipelineStepUI.get_status_css()

        if not hasattr(self, '_selected_samples'):
            self._selected_samples = set()
        self._available_samples = set(list_sample_collections_gcs())

        # Campaign dropdown
        # Search + All/Clear buttons
        self.txt_search_samples = widgets.Text(placeholder=Lang.SEARCH_SAMPLES, layout=L(width='200px'))
        self.txt_search_samples.observe(lambda c: self._refresh_panes(), names='value')

        btn_all = widgets.Button(description=Lang.ALL, icon='check-square',
            button_style='info', layout=L(width='60px'))
        btn_all.on_click(self._select_all_samples)

        btn_clear = widgets.Button(description=Lang.CLEAR, icon='square-o',
            button_style='warning', layout=L(width='60px'))
        btn_clear.on_click(self._clear_all_samples)

        # Panes
        self.available_pane = widgets.VBox([], layout=L(
            border='1px solid #ddd', height='150px', overflow_y='auto', padding='0'))
        self.selected_pane = widgets.VBox([], layout=L(
            border='1px solid #ddd', height='150px', overflow_y='auto', padding='0'))

        # Toolbar
        toolbar = widgets.HBox([
            self.txt_search_samples,
            btn_all, btn_clear
        ], layout=L(gap='4px', margin='0 0 5px 0', width='100%', align_items='center'))

        # Dual pane
        left_pane = widgets.VBox([
            widgets.HTML(f"<b style='font-size:12px; color:#555;'>{Lang.AVAILABLE}</b>"),
            self.available_pane
        ], layout=L(flex='1'))
        right_pane = widgets.VBox([
            widgets.HTML(f"<b style='font-size:12px; color:#555;'>{Lang.SELECTED_OK}</b>"),
            self.selected_pane
        ], layout=L(flex='1'))
        dual_pane = widgets.HBox([left_pane, right_pane], layout=L(gap='20px', padding='5px 10px 10px 10px'))

        self._refresh_panes()
        return widgets.VBox([css, toolbar, dual_pane])

    def _build_extraction_matrix(self):
        """Constrói a matriz dinâmica priorizando o cache local 'state.json'."""
        L = widgets.Layout
        
        available_combos = {} # (sensor, mosaic, period) -> set(bands)
        
        try:
            # 1. Tenta carregar o cache. O CacheManager agora tentará local primeiro.
            state = CacheManager.load()
            selected_period = GLOBAL_OPTS['PERIODICITY'][0]
            all_cogs = CacheManager.cogs_for(selected_period)
            
            if not all_cogs:
                raise ValueError("Cache vazio")

            def _parse_cog_agnostic(name):
                """Parse agnóstico: identifica banda e sensor pelo padrão do arquivo."""
                # padrão: image_peru_fire_{sensor}_{mosaic}_{band}_{date}
                p = name.lower().split('fire_')[-1].split('_')
                if len(p) < 4: return None
                
                # O sensor é sempre o primeiro
                sensor = p[0]
                # A data costuma ser os últimos 1 ou 2 campos (YYYY ou YYYY_MM)
                date_idx = -1
                if p[-2].isdigit() and len(p[-2]) == 4: date_idx = -2 # YYYY_MM
                elif p[-1].isdigit() and len(p[-1]) == 4: date_idx = -1 # YYYY
                
                if date_idx == -1:
                    band = p[-2]
                    mosaic = "_".join(p[1:-2])
                else:
                    band = p[-3]
                    mosaic = "_".join(p[1:-3])
                
                return {'sensor': sensor, 'mosaic': mosaic, 'band': band}

            period_label = {'monthly': 'mensal', 'yearly': 'anual'}.get(selected_period, selected_period)
            for cog_name in all_cogs:
                p = _parse_cog_agnostic(cog_name)
                if p:
                    combo = (p['sensor'], p['mosaic'], period_label)
                    if combo not in available_combos: available_combos[combo] = set()
                    available_combos[combo].add(p['band'])
        except Exception:
            # Fallback offline fixo se nem o cache existir
            selected_period = GLOBAL_OPTS['PERIODICITY'][0]
            period_label = {'monthly': 'mensal', 'yearly': 'anual'}.get(selected_period, selected_period)
            all_combos = {
                (sensor, mosaic, period_label): {'red', 'nir', 'swir1', 'swir2', 'nbr', 'ndvi', 'dayOfYear'}
                for sensor in ('sentinel2', 'landsat')
                for mosaic in ('minnbr', 'minnbr_buffer')
            }
            active = [s.lower() for s in GLOBAL_OPTS.get('SENSOR', ['sentinel2'])]
            available_combos = {k: v for k, v in all_combos.items() if k[0] in active}

        if not available_combos:
            return make_empty_state(Lang.NO_COGS)

        self.band_chk_map = {} 
        matrix_rows = []
        
        # PRIORIDADE DE BANDAS (Ordem sugerida)
        BANDS_PRIORITY = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
        
        for (s, m, p) in sorted(available_combos.keys()):
            found_bands = available_combos[(s, m, p)]
            label_text = f"{s.upper()} {m.replace('_', ' ').title()} ({p.title()})"
            label_html = widgets.HTML(f'<div style="width:200px; font-weight:bold; color:#333; font-size:11px;">{label_text}</div>')
            
            # Ordenação dinâmica: Prioritárias primeiro, resto depois (em ordem alfabética)
            sorted_bands = sorted(list(found_bands), key=lambda x: BANDS_PRIORITY.index(x) if x in BANDS_PRIORITY else 100 + ord(x[0]))
            
            band_widgets = []
            for b in sorted_bands:
                chk = widgets.Checkbox(value=False, indent=False, layout=L(width='18px', height='18px', margin='0'))
                if s == 'sentinel2' and m == 'minnbr' and b in ('red', 'nir', 'swir1', 'swir2'): chk.value = True
                
                self.band_chk_map[(s, m, p, b)] = chk
                
                status_cell = PipelineStepUI.make_status_cell(chk, b.upper(), 'mfm-ok', width='110px')
                band_widgets.append(status_cell)
            
            row = widgets.HBox([label_html] + band_widgets, layout=L(align_items='center', padding='5px 0', border_bottom='1px solid #eee'))
            matrix_rows.append(row)

        matrix_vbox = widgets.VBox(matrix_rows, layout=L(
            border='1px solid #dee2e6', padding='10px', margin='10px 0',
            background_color='#fff', border_radius='4px', max_height='400px', 
            overflow_y='auto', overflow_x='auto'
        ))

        return matrix_vbox

    def _build_hp_section(self):
        L = widgets.Layout
        self.w_iters = widgets.Text(value="7000", description=Lang.ITERATIONS + ':', style={'description_width': '150px'}, layout=L(width='350px'))
        self.w_batch = widgets.Text(value="1000", description=Lang.BATCH_SIZE + ':', style={'description_width': '150px'}, layout=L(width='350px'))
        self.w_lr = widgets.Text(value="0.001", description=Lang.LEARNING_RATE + ':', style={'description_width': '150px'}, layout=L(width='350px'))
        self.w_layers = widgets.Text(value="7, 14, 7", description=Lang.HIDDEN_LAYERS + ':', style={'description_width': '150px'}, layout=L(width='350px'))
        
        return widgets.VBox([
            widgets.HBox([self.w_iters, self.w_batch], layout=L(gap='10px')),
            widgets.HBox([self.w_lr, self.w_layers], layout=L(gap='10px')),
        ], layout=L(padding='15px', border='1px solid #eee', border_radius='8px', margin='0 0 15px 0', flex='1'))

    def _suggest_next_id(self):
        """Sugere o MENOR ID de treino (001, 002...) disponível (preenchendo lacunas)."""
        models = list_trained_models()
        used_ids = set()
        import re
        for m in models:
            # Garante compatibilidade caso o cache ainda tenha o formato antigo
            m_id = m if isinstance(m, str) else m.get('training_id', '')
            match = re.search(r'training_(\d{3})', m_id)
            if match:
                used_ids.add(int(match.group(1)))
        
        # Encontra o primeiro buraco na sequência começando de 1
        for i in range(1, 1000):
            if i not in used_ids:
                return f"{i:03d}"
        return "001"

    def _build_dest_section(self):
        L = widgets.Layout
        next_id = self._suggest_next_id()

        self.w_training_id = widgets.Text(value=next_id, description=Lang.TRAINING_ID,
            style={'description_width': '120px'}, layout=L(width='300px'))
        self.w_shortname = widgets.Text(value='', description=Lang.SHORTNAME,
            layout=L(width='300px'))
        self._update_shortname()
        self.w_comment = widgets.Textarea(placeholder=Lang.COMMENTS, layout=L(width='98%', height='60px'))
        
        return widgets.VBox([
            widgets.HBox([self.w_training_id, self.w_shortname], layout=L(gap='15px')),
            self.w_comment,
        ], layout=L(padding='15px', border='1px solid #eee', border_radius='8px', margin='0 0 15px 0'))

    def _refresh_ui(self):
        self._refresh_canvas_hub()

    def _rebuild_dest_section(self):
        """Rebuild the destination section so the suggested training ID advances."""
        try:
            new_dest = self._build_dest_section()
            children = list(self.new_training_tab.children)
            if children:
                children[-1] = new_dest
                self.new_training_tab.children = children
        except Exception as e:
            print(f"  [WARN] Could not refresh destination section: {e}")
        

    def make_spinner(self, msg=Lang.LOADING):
        return make_spinner(msg=msg)

    def _sync_all(self):
        """Sync all M4 data: models, samples, extraction cache."""
        print(Lang.REPO_SCANNING)
        try:
            CacheManager.clear()
        except Exception:
            pass
        # Delete stale local caches
        for f in ['m4_ranking_cache.json', 'state.json']:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        try:
            CacheManager.build_full_cache()
        except Exception as e:
            print(f"  [WARN] Cache sync: {e}")
        try:
            self._available_samples = set(list_sample_collections_gcs(force_refresh=True))
            if hasattr(self, '_refresh_panes') and self._refresh_panes:
                self._refresh_panes()
        except Exception as e:
            print(f"  [WARN] Samples sync: {e}")
        try:
            self._sync_repository(show_loader=False, force_refresh=True)
        except Exception as e:
            print(f"  [WARN] Repo sync: {e}")
        print(Lang.REPO_SCAN_DONE)

    def _sync_repository(self, show_loader=False, force_refresh=False):
        if show_loader: self.show_loader(Lang.SYNCING)
        
        # list_trained_models e list_sample_collections_gcs agora respeitam o cache por padrão
        models = list_trained_models(force_refresh=force_refresh)
        
        # Atualiza também a lista de amostras se for um refresh forçado
        if force_refresh:
            list_sample_collections_gcs(force_refresh=True)
            self.samples_area = self._build_matrix() # Reconstroi a matriz de amostras
            self.new_training_tab.children = [
                self.new_training_tab.children[0], # Title
                self.samples_area,                 # New Matrix
                *self.new_training_tab.children[2:]# Rest
            ]

        fs = _get_fs()
        
        cache = _load_m4_metadata()
        metadata_cache = cache.get('metadata', {})
        
        updated_cache = False
        for m_id in models:
            # Só baixa metadados se for novo OU se pedirmos refresh total
            if m_id not in metadata_cache or force_refresh:
                try:
                    from M0_auth_config import CONFIG, gcs_models_path
                    clean_path = f"{gcs_models_path()}/{m_id}"
                    with fs.open(f"{CONFIG['bucket']}/{clean_path}/metadata.json", 'r') as f:
                        meta = json.load(f)
                    try:
                        with fs.open(f"{CONFIG['bucket']}/{clean_path}/metrics.json", 'r') as f:
                            metrics = json.load(f)
                        meta['metrics'] = metrics
                    except Exception as e:
                        pass  # metrics.json é opcional
                    
                    metadata_cache[m_id] = meta
                    updated_cache = True
                except Exception as e:
                    print(f"[WARN] Erro ao carregar metadados de {m_id}: {e}")
        
        if updated_cache:
            cache['metadata'] = metadata_cache
            _save_m4_metadata(cache)
            
        if show_loader: self.hide_loader()
        self._refresh_canvas_hub()


    def _update_canvas_live(self, history, embeds, preds, y_true, samples, b_cfg):
        """Atualiza apenas o painel de gráficos vivos, sem tocar na estrutura estável do canvas."""
        # Inicializa a estrutura estática do cabeçalho UMA só VEZ por sessão de treino
        if not getattr(self, '_live_initialized', False):
            self._live_header_html = widgets.HTML()
            self._live_plots_out = widgets.Output()
            self.canvas_output.clear_output(wait=True)
            with self.canvas_output:
                display(HTML(f"<h2 style='color:#2c3e50; border-bottom:3px solid #3498db; padding-bottom:5px; margin-bottom:15px;'>{Lang.LIVE_TRAINING}</h2>"))
                display(self._live_header_html)
                display(self._live_plots_out)
            self._live_initialized = True

        # 1. HEADER DO TREINO ATUAL (Metadados Live via Reatividade .value para não piscar)
        from sklearn.metrics import classification_report
        try:
            rep = classification_report(y_true, (preds > 0.5).astype(int), output_dict=True, zero_division=0)
        except Exception:
            rep = {}
        hp_live = {
            'training_id': self.w_training_id.value, 'shortname': self.w_shortname.value,
            'sample_collections': samples, 'bands_input': sorted(b_cfg.keys()),
            'layers': self.w_layers.value, 'lr': self.w_lr.value,
            'sample_count': self.trainer_instance._sample_count, 'comment': self.w_comment.value,
            'training_date': Lang.TRAINING_IN_PROGRESS
        }
        
        if self.viz_config.get('title'): 
            self._live_header_html.value = render_model_card_html(hp_live, {'classification_report': rep})
        else:
            self._live_header_html.value = ""

        # Atualiza SOMENTE o sub-container de gráficos
        with self._live_plots_out:
            self._live_plots_out.clear_output(wait=True)

            # Atualiza métricas no ranking lateral (LIVE)
            if hasattr(self, 'live_training_info') and self.live_training_info:
                self.live_training_info['acc'] = history['val_acc'][-1] if history['val_acc'] else 0
                self.live_training_info['f1'] = rep.get('1', {}).get('f1-score', 0)
                self._refresh_canvas_hub()
            
            # --- SLIDER COMO BARRA DE PROGRESSO ---
            current_step = len(history.get('steps', [])) - 1
            if current_step >= 0:
                if self.w_global_slider.max < current_step:
                    self.w_global_slider.max = current_step
                # Desabilita o observer temporariamente para evitar recálculo de canvas
                self.w_global_slider.unobserve(self._on_global_slider_change, names='value')
                self.w_global_slider.value = current_step
                self.w_global_slider.observe(self._on_global_slider_change, names='value')

            render_diagnostic_dashboard(history, None, None, None, viz_config={'history': True})
            
            # 2. MODELOS DO RANKING (Comparação em Tempo Real)
            if self.selected_models:
                display(HTML("<div style='margin:50px 0; border-top:3px solid #3498db;'></div>"))
                for mid, info in self.selected_models.items():
                    view_analytics(info, out_widget=None, clear_before=False, viz_config=self.viz_config, ui=self)
                    display(HTML("<div style='margin:40px 0; border-top:1px dashed #ccc;'></div>"))

    def _on_canvas_search_change(self, val):
        self.canvas_search_query = val
        self._refresh_canvas_hub()

    def _on_global_slider_change(self, change):
        self.canvas_slider_val = change['new']
        # Se não estamos em treino ativo, atualiza todos os cards
        if not getattr(self, '_live_initialized', False):
            self._update_canvas()

    def _refresh_canvas_hub(self):
        """Redesenha o Ranking no Painel Lateral do Canvas."""
        # 1. Carregar lista e metadados
        m_ids = list_trained_models()
        cache = _load_m4_metadata()

        # Normaliza seleções antigas (entradas sem 'id' causavam KeyError no analytics)
        for key, info in list(self.selected_models.items()):
            if isinstance(info, str):
                self.selected_models[key] = {'id': key, 'training_id': key}
            elif isinstance(info, dict):
                info.setdefault('id', key)
        
        full_data = []
        for mid in m_ids:
            # Garante que mid seja string mesmo se vier um dicionário de um cache antigo
            if isinstance(mid, dict): mid = mid.get('training_id')
            if not mid: continue
            
            # O path e outros dados extras foram movidos para a sub-chave 'meta'
            meta = dict(cache.get('meta', {}).get(mid, {}))
            # Se não estiver no cache, info mínima
            if not meta and mid in self.canvas_history:
                meta = dict(self.canvas_history[mid])
            
            # As métricas e dados pós-treino salvos no GCS ficam sob 'metadata'
            metadata_rich = cache.get('metadata', {}).get(mid, {})
            if metadata_rich:
                meta.update(metadata_rich)
            # Keep the GCS directory name separate from metadata['training_id']
            # (which is the short numeric id used when retraining).
            meta['id'] = mid
            
            metrics = meta.get('metrics', {})
            rep = metrics.get('classification_report', {})
            acc = rep.get('accuracy', 0)
            f1 = rep.get('1', {}).get('f1-score', 0)
            
            full_data.append({
                'id': mid, 'training_id': mid, 'acc': acc, 'f1': f1, 'meta': meta,
                'shortname': meta.get('shortname', '')
            })
            # Atualiza histórico local
            if mid not in self.canvas_history: self.canvas_history[mid] = meta
            
        # 2. Filtrar
        q = self.canvas_search_query.lower()
        if q:
            full_data = [d for d in full_data if q in d['id'].lower() or q in d['shortname'].lower()]
            
        # 0. Adicionar Treino ao Vivo (se existir)
        if hasattr(self, 'live_training_info') and self.live_training_info:
            full_data.append(self.live_training_info)

        # 3. Ordenar
        rev = not self.canvas_sort_asc
        if self.canvas_sort_col == 'acc':
            full_data.sort(key=lambda x: x['acc'], reverse=rev)
        elif self.canvas_sort_col == 'f1':
            full_data.sort(key=lambda x: x['f1'], reverse=rev)
        else:
            full_data.sort(key=lambda x: x['id'], reverse=self.canvas_sort_asc)

        # 4. Construir Widgets
        available_widgets = []
        selected_widgets = []
        
        for d in full_data:
            mid = d['id']
            is_selected = mid in self.selected_models
            
            # KPI string minimalista
            kpi_str = f"{Lang.ACC_ABBR}: {d['acc']:.1%} | {Lang.F1_ABBR}: {d['f1']:.2f}" if d['acc'] > 0 else Lang.NO_METRICS
            
            # Botão de Ação
            btn = widgets.Button(
                icon='plus' if not is_selected else 'times',
                tooltip=Lang.ADD_TO_CANVAS if not is_selected else Lang.REMOVE_FROM_CANVAS,
                layout=widgets.Layout(width='32px', height='32px', margin='0 5px 0 0'),
                button_style='success' if not is_selected else 'danger'
            )
            
            if not is_selected:
                btn.on_click(lambda _, r=mid, i=d: self._on_canvas_batch_action('add', r, i))
            else:
                btn.on_click(lambda _, r=mid: self._on_canvas_batch_action('remove', r))
                
            info_html = widgets.HTML(f"""
                <div style='line-height:1.2; cursor:default; width:100%;'>
                    <div style='font-size:11px; font-weight:bold; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{mid}</div>
                    <div style='font-size:10px; color:#666;'>{kpi_str}</div>
                </div>
            """, layout=widgets.Layout(flex='1'))
            
            row = widgets.HBox([btn, info_html], layout=widgets.Layout(
                align_items='center', padding='4px', border_bottom='1px solid #eee',
                background_color='#fff' if not is_selected else '#e3f2fd'
            ))
            
            if is_selected: selected_widgets.append(row)
            else: available_widgets.append(row)
            
        self.canvas_available_box.children = available_widgets
        self.canvas_selected_box.children = selected_widgets

    def _on_canvas_batch_action(self, action, rid=None, info=None):
        """Gerencia ações de adição/remoção individual ou em lote no Canvas."""
        if action == 'add' and rid:
            self.selected_models[rid] = info
        elif action == 'remove' and rid:
            self.selected_models.pop(rid, None)
        elif action == 'all':
            q = self.canvas_search_query.lower()
            for r, i in self.canvas_history.items():
                if q in r.lower() or q in i.get('shortname','').lower():
                    self.selected_models[r] = i
        elif action == 'none':
            self.selected_models = {}
            
        self._refresh_canvas_hub()
        self._update_canvas()

    
    def _auto_generate_shortname(self, *_):
        selected_samples = sorted(self._selected_samples)
        if not selected_samples:
            self.w_shortname.value = ""
            return

        if len(selected_samples) == 1:
            first_sample = selected_samples[0]
            region_part = first_sample.replace('_samples', '').replace('library_samples_', '')
        else:
            region_part = f"{len(selected_samples)}regions"

        methods = set()
        bands_count = 0
        for (s, m, p, b), chk in self.band_chk_map.items():
            if chk.value:
                bands_count += 1
                methods.add(m)

        if bands_count == 0:
            return

        method_part = list(methods)[0] if len(methods) == 1 else 'mixed'

        new_name = f"{region_part}_{bands_count}bands_{method_part}"
        self.w_shortname.value = new_name

    def _build_viz_toolbar(self):
        L = widgets.Layout
        
        # 1. Labels e Criação
        labels = {
            'title': Lang.VIZ_METADATA, 'scores': Lang.VIZ_KPIS, 'cm': Lang.VIZ_CONFUSION, 
            'history': Lang.VIZ_HISTORY, 'prob': Lang.VIZ_PROB, 'pr': Lang.VIZ_PR_CURVE, 
            'pca2d': Lang.VIZ_PCA2D, 'pca3d_static': Lang.VIZ_PCA3D_STATIC, 'pca3d': Lang.VIZ_PCA3D_INTERACTIVE,
            'management': Lang.VIZ_MANAGEMENT
        }
        
        chks = {}
        for key, label in labels.items():
            cb = widgets.Checkbox(value=self.viz_config[key], description=label, layout=L(width='auto', margin='0 5px 0 0'))
            def _on_local_change(change, k=key):
                self.viz_config[k] = change['new']
            cb.observe(_on_local_change, names='value')
            chks[key] = cb
            
        def _set_all(val):
            for k in labels.keys():
                chks[k].value = val
                self.viz_config[k] = val

        # 2. Agrupamento por Categorias (Linhas)
        def _make_row(title, keys):
            return widgets.HBox([
                widgets.HTML(f"<b style='width:160px; display:inline-block; color:#2c3e50; font-size: 13px;'>{title}:</b>"),
                widgets.HBox([chks[k] for k in keys], layout=L(flex_flow='row wrap'))
            ], layout=L(align_items='center', margin='2px 0'))
            
        row1 = _make_row(Lang.VIZ_METADATA, ['title', 'scores'])
        row2 = _make_row(Lang.BASIC_STATS, ['cm', 'history', 'prob', 'pr'])
        row3 = _make_row(Lang.PCA_LATENT, ['pca2d', 'pca3d_static', 'pca3d'])
        row4 = _make_row(Lang.VIZ_MANAGEMENT, ['management'])
        
        chk_container = widgets.VBox([row1, row2, row3, row4])
        
        # 3. Botões de Ação na parte inferior
        btn_all = widgets.Button(description=Lang.ALL, layout=L(width='70px'), button_style='info')
        btn_none = widgets.Button(description=Lang.NONE, layout=L(width='70px'), button_style='warning')
        btn_all.on_click(lambda _: _set_all(True))
        btn_none.on_click(lambda _: _set_all(False))
        
        btn_container = widgets.HBox([btn_all, btn_none, widgets.HTML("<div style='width:20px'></div>"), self.w_apply_btn], layout=L(margin='15px 0 0 0', align_items='center'))
        
        return widgets.VBox([
            widgets.HTML(f"<h4 style='margin:0 0 10px 0; color:#34495e; border-bottom:1px solid #ddd; padding-bottom:5px;'>{Lang.VIZ_OPTIONS}</h4>"),
            chk_container,
            btn_container
        ], layout=L(margin='10px 0', padding='15px', background_color='#f8f9fa', border_radius='5px', border='1px solid #dee2e6'))

    def _update_canvas(self):
        """Renderiza o GridBox responsivo com os cards dos modelos selecionados."""
        self._live_initialized = False
        self._refresh_canvas_hub()
        self.canvas_output.clear_output(wait=True)
        
        with self.canvas_output:
            if not self.selected_models and not self.trainer_instance:
                display(HTML(f"<div style='padding:100px; text-align:center; background:white; border-radius:8px;'> <span style='font-size:50px;'></span><br><h3 style='color:#999;'>{Lang.CANVAS_EMPTY}</h3><p style='color:#ccc;'>{Lang.CANVAS_HINT}</p></div>"))
                return
            
            # --- AJUSTE DINAMICO DO SLIDER GLOBAL ---
            max_steps = 1
            for mid, info in self.selected_models.items():
                hp_override = info.get('_hp_override')
                if hp_override:
                    h = hp_override.get('history', {})
                else:
                    meta = info.get('meta', {})
                    h = meta.get('history', {})
                if 'steps' in h and len(h['steps']) > 0:
                    max_steps = max(max_steps, len(h['steps']))
            self.w_global_slider.max = max_steps - 1
            self.w_global_slider.unobserve(self._on_global_slider_change, names='value')
            self.w_global_slider.value = max_steps - 1
            self.canvas_slider_val = max_steps - 1
            self.w_global_slider.observe(self._on_global_slider_change, names='value')
            
            # --- CONSTRUÇÃO DOS CARDS ---
            cards = []
            for mid, info in self.selected_models.items():
                # Cada card é um Output individual para isolar erros e estilos
                card_out = widgets.Output(layout=widgets.Layout(
                    border='1px solid #eee', padding='10px', border_radius='8px', background_color='#fff'
                ))
                with card_out:
                    view_analytics(info, out_widget=None, clear_before=False, viz_config=self.viz_config, epoch_index=self.canvas_slider_val, hp_override=info.get('_hp_override'), ui=self)
                cards.append(card_out)
            
            # Grid responsivo: ocupa o espaço disponível, quebrando linhas conforme necessário
            grid = widgets.GridBox(cards, layout=widgets.Layout(
                grid_template_columns='repeat(auto-fill, minmax(550px, 1fr))',
                grid_gap='20px',
                width='100%'
            ))
            display(grid)

def start_training(ui):
    tf_avail = _get_tf(force=True)
    if tf_avail is None:
        print("\n" + "="*70)
        print(f" [WARNING] {Lang.ERR_INCOMPATIBLE_ENV}")
        print(f" {Lang.ERR_NO_AVX}")
        print(f" {Lang.ERR_RUN_ON_COLAB}")
        print("="*70 + "\n")
        return
    # -----------------------------------------------------------------
    # 1 Retraining intent (checked via the UI state)
    # -----------------------------------------------------------------
    intent = ui.retrain_intent
    if intent.get('mode'):
        hp = intent.get('hp')
        
        # Determine training ID and shortname for the target model
        target_id = hp['training_id'] if hp else ui.w_training_id.value
        target_short = hp['shortname'] if hp else ui.w_shortname.value
        
        # If 'borrar' mode is selected, delete the target model first.
        if intent['mode'] == 'borrar':
            print(f" Deleting previous model: {target_id} ({target_short})")
            ModelTrainer.delete_model(target_id, target_short)
            
        # If hp is provided, load its full configuration into the widgets.
        if hp:
            ui._load_config_into_widgets(hp)
            selected_samples = hp.get('sample_collections', [])
            
        print(f" Mode '{intent['mode']}' activated for {target_id}")
        
        # Reset the intent so it does not fire again accidentally.
        ui.retrain_intent = {'mode': None, 'hp': None}
        # Reset the checkboxes visually too
        for cb in [ui.cb_retrain, ui.cb_reextract, ui.cb_borrar_retrain]:
            cb.unobserve(ui._on_intent_cb_change, names='value')
            cb.value = False
            cb.observe(ui._on_intent_cb_change, names='value')
    
    # 2 Get parameters from UI
    if not intent.get('mode') or not hp:
        selected_samples = sorted(ui._selected_samples)

    if not selected_samples:
        print(Lang.ERR_NO_SAMPLES)
        return

    # 3 Constrói o dicionário de configuração de bandas a partir da Matriz Dinâmica
    bands_config = {}
    sensors_used = set()
    for (s, m, p, b), chk in ui.band_chk_map.items():
        if chk.value:
            # p_norm mapeia os rótulos da UI para as chaves do registry
            p_norm = {'mensal': 'monthly', 'anual': 'yearly'}.get(p, p)
            # A extração espera: bands_config[band_name] = {'sensor': ..., 'mosaic': ..., 'periodicity': ...}
            bands_config[b] = {
                'sensor': s,
                'mosaic': m,
                'periodicity': p_norm
            }
            sensors_used.add(s)
            
    if not bands_config:
        print(Lang.ERR_NO_BANDS)
        return
        
    # Atualiza o sensor global para refletir o que está sendo usado no treinamento
    if len(sensors_used) == 1:
        GLOBAL_OPTS['SENSOR'] = [list(sensors_used)[0]]
    elif len(sensors_used) > 1:
        GLOBAL_OPTS['SENSOR'] = ['multisensor']
        
    # --- PREPARAR INTERFACE PARA NOVO TREINO ---
    ui.selected_models = {}       # Limpa seleções anteriores
    ui.tab.selected_index = 2     # Vai para a aba Treinamentos (renomeada)
    
    # Registra o treino como "LIVE" para aparecer no ranking lateral
    sensor_suffix = GLOBAL_OPTS['SENSOR'][0].lower()
    ui.live_training_info = {
        'id': f"training_{ui.w_training_id.value}_{ui.w_shortname.value}_{sensor_suffix}",
        'shortname': ui.w_shortname.value,
        'acc': 0, 'f1': 0, 'is_live': True,
        'meta': {'path': ''} # Sem path ainda
    }
    
    ui._live_initialized = False  # Reseta estrutura estável para nova sessão
    ui.canvas_output.clear_output(wait=True)
    ui._refresh_canvas_hub()      # Atualiza a barra lateral mostrando o "LIVE"

    try:
        layers = [int(x.strip()) for x in ui.w_layers.value.split(',')]
        iters = int(ui.w_iters.value)
        batch = int(ui.w_batch.value)
        lr = float(ui.w_lr.value)
    except ValueError as e:
        print(f"Error: Invalid hyperparameter value: {e}")
        return
    
    print(f"Extracting pixels from {len(selected_samples)} collections using Flexible Matrix ({len(bands_config)} bands). Please wait...")
    
    def _logger(msg, level="info"):
        print(msg)
        
    X, y = extract_pixels_from_gcs(selected_samples, bands_config, logger=_logger)
    
    if len(X) == 0:
        print("Failed to extract pixels.")
        return
        
    print(f"Success: {len(X)} pixels extracted (Fire: {y.sum()} | No-fire: {(y==0).sum()}).")
    cell_log(f"{len(X)} pixels extracted ({int(y.sum())} fire, {int((y==0).sum())} no-fire)", type='success')
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ui.trainer_instance = ModelTrainer(num_input=len(bands_config), layers=layers, lr=lr)
    ui.trainer_instance._bands_input = sorted(bands_config.keys())
    ui.trainer_instance._bands_config = bands_config
    ui.trainer_instance._sample_collections = selected_samples
    ui.trainer_instance._sample_count = {'burned': int(y.sum()), 'not_burned': int((y==0).sum())}
    
    print("Training DNN...")
    cell_log("Training DNN...", type='info')
    ui.tab.selected_index = 2  # switch to Canvas tab
    from IPython.display import display as ipy_display
    progress_bar = widgets.IntProgress(value=0, min=0, max=100, description=Lang.TRAINING_PROGRESS,
                                        bar_style='info', layout=widgets.Layout(width='100%'))
    ipy_display(progress_bar)
    
    # Snapshot Directory
    m_id = ui.w_training_id.value
    m_short = ui.w_shortname.value
    snap_dir = f"library_images/models/{m_id}_{m_short}"
    os.makedirs(snap_dir, exist_ok=True)

    def update_chart(history, embeds=None, preds=None, y_true=None):
        ui._update_canvas_live(history, embeds, preds, y_true, selected_samples, bands_config)
        if history and history.get('steps'):
            steps = history['steps']
            n_total = iters
            if steps:
                pct = min(100, int(len(steps) / 21 * 100))
                progress_bar.value = pct
                progress_bar.description = f'{Lang.TRAINING_PROGRESS} {pct}%'

    # Iniciar Treino
    ui.trainer_instance.train(X_train, y_train, X_val=X_val, y_val=y_val, 
                              batch_size=batch, n_iters=iters, logger=_logger, 
                              update_chart_fn=update_chart, snapshot_dir=snap_dir)
    
    # --- AUDITORIA FINAL COM t-SNE (INTERATIVO) ---
    print("\n Training completed. Generating high-resolution t-SNE audit...")
    cell_log("Training completed", type='success')
    ui._live_initialized = False
    ui._live_plots_out.clear_output(wait=True)
    with ui._live_plots_out:
        import plotly.graph_objects as go
        from sklearn.manifold import TSNE
        try:
            display(HTML(f"<h4 style='color:#2c3e50; margin-top:20px; font-weight:bold;'>{Lang.LIVE_TSNE_AUDIT}</h4>"))
            display(HTML(f"<p style='font-size:11px; color:#666;'>{Lang.PROCESSING}</p>"))
            
            idx_v = np.random.choice(len(X_val), min(600, len(X_val)), replace=False)
            X_v_sub = X_val[idx_v]
            y_v_sub = y_val[idx_v]
            
            emb_v = ui.trainer_instance.get_embeddings(X_v_sub)
            prd_v = ui.trainer_instance.predict(X_v_sub)
            
            print("  - Computing t-SNE manifold (this may take a while)...")
            tsne = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
            coords_tsne = tsne.fit_transform(emb_v)
            
            ui.trainer_instance.tsne_snapshot = coords_tsne.tolist()
            
            print("  - Generating interactive figure...")
            fig_tsne = go.Figure(data=[go.Scatter3d(
                x=coords_tsne[:,0], y=coords_tsne[:,1], z=coords_tsne[:,2],
                mode='markers',
                marker=dict(size=4, color=prd_v.flatten(), colorscale='RdBu_r', opacity=0.8, showscale=True),
                text=[f"{Lang.FIRE_CLASS if l==1 else Lang.NO_FIRE_CLASS}<br>Pred: {p:.2%}" for l, p in zip(y_v_sub, prd_v.flatten())]
            )])
            fig_tsne.update_layout(
                margin=dict(l=0, r=0, b=0, t=30),
                scene=dict(xaxis_title=Lang.TSNE_AXIS_1, yaxis_title=Lang.TSNE_AXIS_2, zaxis_title=Lang.TSNE_AXIS_3)
            )
            display(HTML(fig_tsne.to_html(include_plotlyjs=True, full_html=False)))
            print("[OK] t-SNE audit ready.")
        except Exception as e:
            print(f"[WARNING] Could not generate final t-SNE: {e}")

    print("Saving structure (samples, pixels, metadata, metrics) to GCS...")
    try:
        saved_meta = ui.trainer_instance.save(ui.w_training_id.value, ui.w_shortname.value, comment=ui.w_comment.value, logger=_logger)
        saved_path = model_path(ui.w_training_id.value, ui.w_shortname.value)
        print("Model and Model Card saved successfully!")
        print(f"Model path: gs://{CONFIG['bucket']}/{saved_path}")
        cell_log(f"Model saved to gs://{CONFIG['bucket']}/{saved_path}", type='success')
        
        # Inserir o modelo fresquinho na Mesa do Canvas automaticamente
        final_id = f"training_{ui.w_training_id.value}_{ui.w_shortname.value}_{GLOBAL_OPTS['SENSOR'][0].lower()}"
        ui.selected_models = {
            final_id: {
                'training_id': final_id,
                'path': model_path(ui.w_training_id.value, ui.w_shortname.value),
                '_hp_override': saved_meta
            }
        }
        
        # Registra no cache sem rebuild global: sidebar e proximo ID na hora.
        register_model(final_id, saved_meta)
        ui.live_training_info = None  # Remove o status de LIVE após conclusão
        ui._live_initialized = False
        ui._rebuild_dest_section()    # Avanca o Training ID sugerido
        ui._update_canvas()           # Pinta o card final e atualiza a sidebar
        
    except Exception as e:
        print(f"Error saving: {e}")
        cell_log(f"Error saving model: {e}", type='error')

def run_ui():
    ui = ModelTrainerUI()
    ui.display()
    return ui

