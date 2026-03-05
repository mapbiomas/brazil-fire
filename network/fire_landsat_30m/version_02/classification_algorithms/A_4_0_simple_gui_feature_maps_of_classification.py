# A_4_0_simple_gui_feature_maps_of_classification.py
# last update: '2025/06/02'
# MapBiomas Fire Classification Algorithms Step A_4_0 Simple graphic user interface for feature maps extraction
# (Versão Final Corrigida e Isolada)

# IMPORT LIBRARIES
import subprocess
import sys
import importlib
import os
import time
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import VBox, HBox
import gcsfs
import numpy as np

# Assumimos que A_4_1_tensorflow_embedding_extraction.py está no mesmo ambiente
# Substitua pelo seu nome de arquivo real
# from A_4_1_tensorflow_embedding_extraction import render_embedding_models 

# Caso o módulo A_4_1 não possa ser importado, a função deve ser mockada
def render_embedding_models(models_to_process, simulate_test=False):
    print("[MOCK] Chamada para render_embedding_models. Processamento simulado.")
    for model in models_to_process:
        print(f"[MOCK] Processando: {model['model']} com camada {model['embedding_layer']}")


# GLOBAL VARIABLES AND DIRECTORY SETUP
bucket_name = 'mapbiomas-fire'
base_folder = 'mapbiomas-fire/sudamerica/'

# Variáveis de estado da GUI para EMBEDDING (A_4_0) - ISOLADAS
EMB_selected_country = '' 
EMB_checkboxes = []
EMB_mosaic_panels = []
EMB_mosaic_checkboxes_dict = {}
EMB_mosaic_checkbox_states = {}

# Variável para armazenar a camada de embedding selecionada e o widget
EMB_selected_embedding_layer = 'h5' 
EMB_embedding_layer_selector = None 

# Função de log simulada
def log_message(msg):
    print(f"[LOG] {msg}")

# CORE CLASSES (ModelRepository adaptado para Embeddings)

class ModelRepository:
    """Gerencia a listagem de modelos, mosaicos e a verificação de embeddings existentes no GCS."""
    def __init__(self, bucket_name, country):
        self.bucket = bucket_name
        self.country = country
        self.base_folder = f'mapbiomas-fire/sudamerica/{country}'
        self.fs = gcsfs.GCSFileSystem(project=bucket_name)

    def list_models(self):
        # CORRIGIDO: Usar models_col1/ e filtrar prefixos
        training_folder = f"{self.base_folder}/models_col1/" 
        try:
            files = self.fs.ls(training_folder)
            
            model_prefixes = set()
            for file in files:
                base_name = file.split('/')[-1]
                
                if 'hyperparameters' in base_name:
                    continue
                    
                prefix = base_name.split('.')[0] 
                if prefix and 'ckpt' in prefix:
                    model_prefixes.add(prefix)
            
            return list(model_prefixes), len(model_prefixes)
            
        except Exception:
            return [], 0

    def list_mosaics(self, region):
        """Lista mosaicos COG (os inputs) filtrando pela região exata."""
        # CORRIGIDO: Mudar para mosaics_col1_cog/ (igual A_3_0)
        mosaics_folder = f"{self.base_folder}/mosaics_col1_cog/"
        
        # CORRIGIDO: Usar o filtro _region_ para isolar (igual A_3_0)
        region_filter = f"_{region}_"
        
        try:
            files = self.fs.ls(mosaics_folder)
            
            return [file.split('/')[-1] for file in files if region_filter in file], len(files)
            
        except Exception:
            return [], 0


    def list_embeddings(self):
        embeddings_folder = f"{self.base_folder}/result_embeddings/"
        try:
            files = self.fs.ls(embeddings_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.tif')], len(files)
        except Exception:
            return [], 0

    def is_embedding_generated(self, mosaic_file):
        """Verifica se um mosaico tem um embedding correspondente já gerado."""
        # Lógica de verificação...
        return False # Mantendo como placeholder por não termos a lógica completa

# SUPPORT FUNCTIONS (GUI Handling)

def display_selected_mosaics_embedding(model, selected_country, region):
    """Exibe o painel de seleção de mosaicos, marcando aqueles que já têm embeddings."""
    repo = ModelRepository(bucket_name=bucket_name, country=selected_country)
    mosaic_files, mosaic_count = repo.list_mosaics(region)
    
    mosaics_panel = widgets.Output(layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'})
    checkboxes_mosaics = []
    saved_states = EMB_mosaic_checkbox_states.get(model, None) 

    with mosaics_panel:
        if mosaic_files:
            for idx, file in enumerate(mosaic_files):
                embedding_generated = repo.is_embedding_generated(file)
                checkbox_mosaic = widgets.Checkbox(
                    value=False,
                    description=file + (" 🌟 (Embedding OK)" if embedding_generated else "")
                )
                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]
                    
                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")
            
    EMB_mosaic_checkboxes_dict[model] = checkboxes_mosaics
    
    def toggle_select_all(change):
        for checkbox in EMB_mosaic_checkboxes_dict.get(model, []): 
            checkbox.value = change['new']
            
    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')
    
    legend_panel = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'})
    with legend_panel:
        print("🌟 Embeddings já gerados para este mosaico. Eles serão sobrescritos se selecionados.")
        
    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])

def update_interface():
    """Atualiza a interface gráfica."""
    global EMB_checkboxes, EMB_mosaic_panels, EMB_embedding_layer_selector
    
    clear_output(wait=True)
    
    # 1. Re-exibe o painel de seleção de camadas
    layer_panel = create_layer_selector_panel()
    display(layer_panel)


    # 2. Exibe os checkboxes de modelos
    display(VBox(EMB_checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')))
    
    # 3. Exibe os painéis de mosaicos (imagens anuais)
    EMB_mosaic_panels_widgets = [panel[2] for panel in EMB_mosaic_panels]
    display(HBox(EMB_mosaic_panels_widgets, layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')))

   # 4. Exibe texto informativo para execução em célula separada
    display(widgets.HTML("<b>Ação:</b> Após selecionar os modelos e anos, execute a célula separada com `execute_embedding_generation_click(None)`"))
    # # 4. Re-cria e exibe o botão de execução
    # execute_button = widgets.Button(
    #     description="EXECUTAR EXTRAÇÃO DE EMBEDDINGS E UPLOAD (A_4_1)",
    #     button_style='info',
    #     layout=widgets.Layout(width='auto')
    # )
    # execute_button.on_click(execute_embedding_generation_click)
    # display(execute_button)

def create_layer_selector_panel():
    """Cria o painel de seleção de camadas de embedding (h1 a h5) com descrições mais ricas."""
    global EMB_selected_embedding_layer, EMB_embedding_layer_selector
    
    # Valores de dimensão baseados no A_2_1: L1=7, L2=14, L3=7, L4=14, L5=7
    layer_options = {
        'L1 (7 Bandas) - Features de Baixo Nível (Espectral)': 'h1',
        'L2 (14 Bandas) - Features Intermediárias (Contexto Temporal Simples)': 'h2',
        'L3 (7 Bandas) - Features Consolidadas (Síntese Compacta)': 'h3',
        'L4 (14 Bandas) - Features de Alto Nível (Padrões Temporais Complexos)': 'h4',
        'L5 (7 Bandas - Padrão) - Representação Latente Final (Discriminação)': 'h5' 
    }
    
    # Se o widget não existe, cria. Se existe, apenas garante que o valor está atualizado.
    if EMB_embedding_layer_selector is None:
        EMB_embedding_layer_selector = widgets.RadioButtons(
            options=layer_options,
            value=EMB_selected_embedding_layer, 
            description='Camada p/ Embedding:',
            disabled=False,
            layout=widgets.Layout(width='auto')
        )
        def update_selected_layer(change):
            global EMB_selected_embedding_layer
            EMB_selected_embedding_layer = change['new']
        EMB_embedding_layer_selector.observe(update_selected_layer, names='value')
    
    panel = widgets.VBox([
        widgets.HTML("<b>Escolha a Camada para Extração de Embedding (Output Dimensão x Bandas):</b>"),
        EMB_embedding_layer_selector
    ], layout=widgets.Layout(border='1px solid blue', padding='10px', margin='10px 0'))
    
    EMB_selected_embedding_layer = EMB_embedding_layer_selector.value
    
    return panel

def collect_selected_models():
    """Coleta todos os nomes de arquivos de modelo selecionados."""
    global EMB_checkboxes 
    selected_models = [checkbox.description for checkbox in EMB_checkboxes if checkbox.value]
    return selected_models

def execute_embedding_generation_click(b):
    """Gatilho principal para iniciar a extração de embeddings."""
    global EMB_selected_embedding_layer, EMB_mosaic_checkboxes_dict
    
    selected_models = collect_selected_models()
    models_to_process = []
    
    current_layer_choice = EMB_selected_embedding_layer
    if not current_layer_choice:
        log_message("Por favor, selecione a camada de embedding para extração.")
        return
        
    if selected_models:
        for model in selected_models:
            model_key = f"{model}.meta"
            
            if model in EMB_mosaic_checkboxes_dict: 
                mosaic_checkboxes = EMB_mosaic_checkboxes_dict[model]
                selected_mosaics = [cb.description.replace(" 🌟 (Embedding OK)", "").strip() for cb in mosaic_checkboxes if cb.value]
                
                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model_key}")
                    continue
                    
                model_obj = {
                    "model": model_key,
                    "mosaics": selected_mosaics,
                    "simulation": False,
                    "embedding_layer": current_layer_choice
                }
                models_to_process.append(model_obj)
            else:
                log_message(f"Nenhum mosaico encontrado para o modelo: {model_key}")

        if models_to_process:
            log_message(f"[INFO] Chamando o extrator de embeddings para: {models_to_process}")
            render_embedding_models(models_to_process, simulate_test=False) 
        else:
            log_message("Nenhum mosaico foi selecionado para nenhum modelo.")
    else:
        log_message("Nenhum modelo selecionado.")

def on_select_country(country_name):
    """Manipula a seleção do país e exibe os modelos disponíveis."""
    global EMB_selected_country, EMB_checkboxes, EMB_mosaic_panels
    
    EMB_selected_country = country_name
    
    repo = ModelRepository(bucket_name=bucket_name, country=EMB_selected_country)
    training_files, file_count = repo.list_models()
    
    if training_files:
        
        EMB_checkboxes = [] 
        EMB_mosaic_panels = [] 
        
        for file in training_files:
            try:
                # Extração da região (Ex: r1)
                parts = file.split('_')
                region_part = parts[3] 

                checkbox = widgets.Checkbox(
                    value=False,
                    description=file,
                    layout=widgets.Layout(width='700px')
                )
                
                checkbox.observe(lambda change, f=file, reg=region_part: update_panels(change, f, reg), names='value')
                EMB_checkboxes.append(checkbox)
                
            except Exception:
                log_message(f"[WARNING] Arquivo de modelo com nome inesperado: {file}")

        update_interface()
    else:
        log_message("Nenhum arquivo de modelo encontrado.")
        clear_output(wait=True)
        layer_panel = create_layer_selector_panel()
        display(layer_panel)
        display(widgets.HTML("<b style='color: red;'>Nenhum modelo encontrado para este país.</b>"))


def update_panels(change, file, region):
    """Atualiza a lista de painéis de mosaico quando um checkbox de modelo é ativado/desativado."""
    global EMB_mosaic_panels, EMB_selected_country, EMB_mosaic_checkboxes_dict, EMB_mosaic_checkbox_states
    
    if change['new']: # Se o checkbox for marcado
        panel = display_selected_mosaics_embedding(file, EMB_selected_country, region)
        EMB_mosaic_panels.append((file, region, panel))
    else: # Se o checkbox for desmarcado
        if file in EMB_mosaic_checkboxes_dict:
            checkbox_list = EMB_mosaic_checkboxes_dict[file]
            EMB_mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]
        
        EMB_mosaic_panels = [p for p in EMB_mosaic_panels if p[0] != file or p[1] != region]
        
    update_interface()
