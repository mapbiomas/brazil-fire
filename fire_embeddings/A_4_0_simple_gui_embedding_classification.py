# A_4_0_simple_gui_embedding_classification.py
# last_update: '2025/06/02'
# MapBiomas Fire Classification Algorithms Step A_4_0 - Simple graphic user interface for embedding extraction

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
# ====================================

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
# Assumimos que A_4_1 está no mesmo ambiente de execução
from A_4_1_tensorflow_embedding_extraction import render_embedding_models 

# Variáveis globais assumidas: country, log_message

# ====================================
# 🌍 GLOBAL VARIABLES AND DIRECTORY SETUP
# ====================================

bucket_name = 'mapbiomas-fire'
base_folder = 'mapbiomas-fire/sudamerica/'
fs = gcsfs.GCSFileSystem(project=bucket_name)

# Variáveis de estado da GUI
selected_country = ''
checkboxes = []
mosaic_panels = [] 
mosaic_checkboxes_dict = {}
mosaic_checkbox_states = {}

# ====================================
# 🧠 CORE CLASSES (ModelRepository adaptado para Embeddings)
# ====================================

class ModelRepository:
    """
    Gerencia a listagem de modelos, mosaicos e a verificação de embeddings existentes no GCS.
    """
    def __init__(self, bucket_name, country):
        self.bucket = bucket_name
        self.country = country
        self.base_folder = f'mapbiomas-fire/sudamerica/{country}'
        self.fs = gcsfs.GCSFileSystem(project=bucket_name)

    def list_models(self):
        # Lista modelos treinados (checkpoint files)
        training_folder = f"{self.base_folder}/models_col1/"
        try:
            files = self.fs.ls(training_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.meta')], len(files)
        except FileNotFoundError:
            return [], 0

    def list_mosaics(self, region):
        # Lista mosaicos COG (os inputs)
        mosaics_folder = f"{self.base_folder}/mosaics_col1_cog/"
        try:
            files = self.fs.ls(mosaics_folder)
            return [file.split('/')[-1] for file in files if f"_{region}_" in file], len(files)
        except FileNotFoundError:
            return [], 0

    def list_embeddings(self):
        # NOVO: Lista arquivos de embedding gerados
        embeddings_folder = f"{self.base_folder}/result_embeddings/"
        try:
            files = self.fs.ls(embeddings_folder)
            # Retorna apenas o nome do arquivo
            return [file.split('/')[-1] for file in files if file.endswith('.tif')], len(files) 
        except FileNotFoundError:
            return [], 0
    
    def is_embedding_generated(self, mosaic_file):
        """
        Verifica se um mosaico tem um embedding correspondente já gerado.
        O nome do arquivo de embedding deve ser prefixado com 'embedding_'
        """
        embedding_files, _ = self.list_embeddings()
        parts = mosaic_file.split('_')
        sat = parts
        region_part = parts[19][1:] # r1 -> 1
        year = parts[62]
        
        expected_prefix = f"embedding_{self.country}_{sat}_v*_region{region_part}_{year}"
        
        for emb_name in embedding_files:
            # Verifica correspondência parcial (ignora a versão do modelo por enquanto)
            if sat in emb_name and f"region{region_part}" in emb_name and year in emb_name:
                return True
        return False

# ====================================
# 🧰 SUPPORT FUNCTIONS (GUI Handling)
# ====================================

def display_selected_mosaics_embedding(model, selected_country, region):
    """
    Exibe o painel de seleção de mosaicos, marcando aqueles que já têm embeddings.
    """
    repo = ModelRepository(bucket_name=bucket_name, country=selected_country)
    mosaic_files, mosaic_count = repo.list_mosaics(region)

    mosaics_panel = widgets.Output(layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'})
    checkboxes_mosaics = []
    saved_states = mosaic_checkbox_states.get(model, None)

    with mosaics_panel:
        if mosaic_files:
            for idx, file in enumerate(mosaic_files):
                embedding_generated = repo.is_embedding_generated(file)
                
                checkbox_mosaic = widgets.Checkbox( 
                    value=False, 
                    description=file + (" ⚠️ (Embedding OK)" if embedding_generated else "") 
                )
                
                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]

                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")

    mosaic_checkboxes_dict[model] = checkboxes_mosaics

    def toggle_select_all(change):
        for checkbox in checkboxes_mosaics:
            checkbox.value = change['new']

    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')

    legend_panel = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'})
    with legend_panel:
        print("⚠️ Embeddings já gerados para este mosaico. Eles serão sobrescritos se selecionados.")

    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])

def update_interface():
    """
    Atualiza a interface gráfica. (Idêntico ao A_3_0 [63])
    """
    clear_output(wait=True)
    
    display(VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')))
    
    mosaic_panels_widgets = [panel[19] for panel in mosaic_panels]
    display(HBox(mosaic_panels_widgets, layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')))

def collect_selected_models():
    """
    Coleta todos os nomes de arquivos de modelo selecionados. (Idêntico ao A_3_0 [64])
    """
    selected_models = [checkbox.description for checkbox in checkboxes if checkbox.value]
    return selected_models

def execute_embedding_generation_click(b):
    """
    Gatilho principal para iniciar a extração de embeddings.
    Define o modo como EMBEDDING e aciona o render_embedding_models.
    """
    selected_models = collect_selected_models()
    models_to_process = [] 

    if selected_models:
        for model in selected_models:
            model_key = f"{model}.meta" 
            
            if model_key in mosaic_checkboxes_dict:
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]
                selected_mosaics = [cb.description.replace(" ⚠️ (Embedding OK)", "").strip() for cb in mosaic_checkboxes if cb.value]

                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model_key}")
                    continue

                model_obj = {
                    "model": model_key, # Usa o nome completo do arquivo .meta
                    "mosaics": selected_mosaics, 
                    "simulation": False,
                }
                models_to_process.append(model_obj)
            else:
                log_message(f"Nenhum mosaico encontrado para o modelo: {model_key}")

        if models_to_process:
            log_message(f"[INFO] Chamando o extrator de embeddings para: {models_to_process}")
            render_embedding_models(models_to_process, simulate_test=False) # Chamar a função de A_4_1
        else:
            log_message("Nenhum mosaico foi selecionado para nenhum modelo.")
    else:
        log_message("Nenhum modelo selecionado.") 

def on_select_country(country_name):
    """
    Manipula a seleção do país e exibe os modelos disponíveis. (Adaptado de A_3_0 [65])
    """
    global selected_country 
    selected_country = country_name

    repo = ModelRepository(bucket_name='mapbiomas-fire', country=country_name)
    training_files, file_count = repo.list_models()

    if training_files:
        global checkboxes, mosaic_panels
        checkboxes = []
        mosaic_panels = [] 

        for file in training_files:
            if file.split('.')[12] == 'meta': 
                region = file.split('_')[-4].split('.') 
                
                checkbox = widgets.Checkbox(
                    value=False,
                    description=file.split('.'), # Exibe o nome base (ex: col1_guyana_v1_r1)
                    layout=widgets.Layout(width='700px') 
                )
                
                # O change.owner aqui será o nome base (sem .meta)
                checkbox.observe(lambda change, f=file, reg=region: update_panels(change, f, reg), names='value')
                checkboxes.append(checkbox)

        update_interface()
    else:
        # (Lógica para display de erro - omitida)
        pass

def update_panels(change, file, region):
    """
    Atualiza a lista de painéis de mosaico quando um checkbox de modelo é ativado/desativado.
    """
    global mosaic_panels, selected_country 
    if change['new']: # Se o checkbox for marcado
        panel = display_selected_mosaics_embedding(file, selected_country, region) # <--- CHAMA FUNÇÃO DE EMBEDDING
        mosaic_panels.append((file, region, panel))
    else: # Se o checkbox for desmarcado
        # Salva o estado atual
        if file in mosaic_checkboxes_dict:
            checkbox_list = mosaic_checkboxes_dict[file]
            mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]
        
        # Remove o painel
        mosaic_panels = [p for p in mosaic_panels if p != file or p[12] != region]

    update_interface()


# ====================================
# 🚀 RUNNING THE INTERFACE
# ====================================

# Trigger the initial interface setup by selecting the country
# on_select_country(country) # Assume que 'country' é global/ambiente

# Botão de Execução Principal
execute_button = widgets.Button(
    description="EXECUTAR EXTRAÇÃO DE EMBEDDINGS E UPLOAD (A_4_1)", 
    button_style='info', 
    layout=widgets.Layout(width='auto')
)
execute_button.on_click(execute_embedding_generation_click)

# Display principal da interface (assumindo que a seleção de país já ocorreu)
# display(execute_button)
