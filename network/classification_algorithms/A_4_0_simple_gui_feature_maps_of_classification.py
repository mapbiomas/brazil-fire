# A_4_0_simple_gui_embedding_classification.py
# last update: '2025/06/02'
# MapBiomas Fire Classification Algorithms Step A_4_0 Simple graphic user interface for embedding extraction
# (Versão Modificada para Seleção Dinâmica de Camada de Embedding)

# INSTALAR E IMPORTAR LIBRARIES (assumindo que a maioria já está instalada)
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
# Certifique-se de que este arquivo exista e esteja no mesmo diretório
# Se não estiver, substitua esta linha pela função render_embedding_models completa
# from A_4_1_tensorflow_embedding_extraction import render_embedding_models

# GLOBAL VARIABLES AND DIRECTORY SETUP
bucket_name = 'mapbiomas-fire'
base_folder = 'mapbiomas-fire/sudamerica/'
# fs é inicializado dentro do ModelRepository e na execução de A_4_1
# (Pode ser inicializado aqui para uso global se necessário, mas não é estritamente usado na GUI)

# Variáveis de estado da GUI para EMBEDDING (A_4_0) - Isoladas de A_3_0
EMB_selected_country = '' 
EMB_checkboxes = []
EMB_mosaic_panels = []
EMB_mosaic_checkboxes_dict = {}
EMB_mosaic_checkbox_states = {}

# Variável para armazenar a camada de embedding selecionada e o widget
EMB_selected_embedding_layer = 'h5' # Default para a última camada oculta (L5)
EMB_embedding_layer_selector = None # O widget em si

# Função de log simulada (usada para debug e mensagens de feedback)
# Substitua por sua função de log real se ela estiver definida em outro lugar
def log_message(msg):
    print(f"[LOG] {msg}")

# CORE CLASSES (ModelRepository adaptado para Embeddings)

class ModelRepository:
    """Gerencia a listagem de modelos, mosaicos e a verificação de embeddings existentes no GCS."""
    def __init__(self, bucket_name, country):
        self.bucket = bucket_name
        self.country = country
        self.base_folder = f'mapbiomas-fire/sudamerica/{country}'
        # Inicializa o GCSFS para as operações do repo
        self.fs = gcsfs.GCSFileSystem(project=bucket_name)

    def list_models(self):
        # Lista modelos treinados (checkpoint files)
        training_folder = f"{self.base_folder}/models_coll/"
        try:
            files = self.fs.ls(training_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.meta')], len(files)
        except Exception:
            return [], 0

    def list_mosaics(self, region):
        # Lista mosaicos COG (os inputs)
        mosaics_folder = f"{self.base_folder}/mosaics_coll_cog/"
        try:
            files = self.fs.ls(mosaics_folder)
            # Filtra por região (assumindo formato de nome de arquivo)
            return [file.split('/')[-1] for file in files if f"_{region}" in file], len(files)
        except Exception:
            return [], 0

    def list_embeddings(self):
        # NOVO: Lista arquivos de embedding gerados
        embeddings_folder = f"{self.base_folder}/result_embeddings/"
        try:
            files = self.fs.ls(embeddings_folder)
            return [file.split('/')[-1] for file in files if file.endswith('.tif')], len(files)
        except Exception:
            return [], 0

    def is_embedding_generated(self, mosaic_file):
        """Verifica se um mosaico tem um embedding correspondente já gerado."""
        embedding_files, _ = self.list_embeddings()
        parts = mosaic_file.split('_')
        
        try:
            sat = parts[0]
            # Adapte esta lógica de parsing de região e ano se o formato de nome for diferente
            # Exemplo: coll_guyana_v1_r1.meta para mosaicos como L5_167020_2020.tif
            region_part = parts[2] 
            year = parts[3].split('.')[0]
            
            for emb_name in embedding_files:
                # Checa por satélite, região e ano no nome do arquivo de embedding
                if f"_{sat}_" in emb_name and f"region{region_part}" in emb_name and f"_{year}" in emb_name:
                    return True
            return False
        except Exception:
            # Caso o formato do nome do arquivo seja inesperado
            return False

# SUPPORT FUNCTIONS (GUI Handling)

def display_selected_mosaics_embedding(model, selected_country, region):
    """Exibe o painel de seleção de mosaicos, marcando aqueles que já têm embeddings."""
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
                    description=file + (" 🌟 (Embedding OK)" if embedding_generated else "")
                )
                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]
                    
                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")
            
    mosaic_checkboxes_dict[model] = checkboxes_mosaics
    
    # Função para selecionar/desselecionar todos
    def toggle_select_all(change):
        for checkbox in mosaic_checkboxes_dict.get(model, []):
            checkbox.value = change['new']
            
    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')
    
    legend_panel = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'})
    with legend_panel:
        print("🌟 Embeddings já gerados para este mosaico. Eles serão sobrescritos se selecionados.")
        
    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])

def update_interface():
    """Atualiza a interface gráfica."""
    clear_output(wait=True)
    
    # Re-exibe o seletor de camadas e o botão
    if embedding_layer_selector:
        display(embedding_layer_selector.parent)

    display(VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')))
    
    # Exibe os painéis de mosaicos
    mosaic_panels_widgets = [panel[2] for panel in mosaic_panels]
    display(HBox(mosaic_panels_widgets, layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')))

    # Re-exibe o botão de execução
    execute_button = widgets.Button(description="EXECUTAR EXTRAÇÃO DE EMBEDDINGS E UPLOAD (A_4_1)")
    execute_button.on_click(execute_embedding_generation_click)
    display(execute_button)

def create_layer_selector_panel():
    """Cria o painel de seleção de camadas de embedding (h1 a h5)."""
    global selected_embedding_layer, embedding_layer_selector
    
    layer_options = {
        'Camada 1 (L1) - Tensor h1:0': 'h1',
        'Camada 2 (L2) - Tensor h2:0': 'h2',
        'Camada 3 (L3) - Tensor h3:0': 'h3',
        'Camada 4 (L4) - Tensor h4:0': 'h4',
        'Camada 5 (L5 - Padrão) - Tensor extracted_embedding:0': 'h5' 
    }
    
    embedding_layer_selector = widgets.RadioButtons(
        options=layer_options,
        value='h5', # Valor padrão
        description='Camada p/ Embedding:',
        disabled=False,
        layout=widgets.Layout(width='auto')
    )
    
    def update_selected_layer(change):
        global selected_embedding_layer
        selected_embedding_layer = change['new']
        
    embedding_layer_selector.observe(update_selected_layer, names='value')
    
    panel = widgets.VBox([
        widgets.HTML("<b>Escolha a Camada para Extração de Embedding:</b>"),
        embedding_layer_selector
    ], layout=widgets.Layout(border='1px solid blue', padding='10px', margin='10px 0'))
    
    selected_embedding_layer = embedding_layer_selector.value
    
    return panel

def collect_selected_models():
    """Coleta todos os nomes de arquivos de modelo selecionados."""
    selected_models = [checkbox.description for checkbox in checkboxes if checkbox.value]
    return selected_models

def execute_embedding_generation_click(b):
    """Gatilho principal para iniciar a extração de embeddings."""
    global selected_embedding_layer
    
    selected_models = collect_selected_models()
    models_to_process = []
    
    current_layer_choice = selected_embedding_layer
    if not current_layer_choice:
        log_message("Por favor, selecione a camada de embedding para extração.")
        return
        
    if selected_models:
        for model in selected_models:
            model_key = f"{model}.meta"
            
            if model_key in mosaic_checkboxes_dict:
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]
                selected_mosaics = [cb.description.replace(" 🌟 (Embedding OK)", "").strip() for cb in mosaic_checkboxes if cb.value]
                
                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model_key}")
                    continue
                    
                model_obj = {
                    "model": model_key,
                    "mosaics": selected_mosaics,
                    "simulation": False,
                    "embedding_layer": current_layer_choice # Inclui a camada
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
    global selected_country
    selected_country = country_name
    
    repo = ModelRepository(bucket_name=bucket_name, country=selected_country)
    training_files, file_count = repo.list_models()
    
    if training_files:
        global checkboxes, mosaic_panels
        checkboxes = []
        mosaic_panels = []
        
        for file in training_files:
            try:
                # Extrai a região para uso em update_panels e na busca por mosaicos
                # Lógica baseada no nome esperado: coll_country_vX_rY.meta
                region = file.split('_')[-1].split('.')[0] 
                
                checkbox = widgets.Checkbox(
                    value=False,
                    description=file.split('.')[0], # Nome base do modelo
                    layout=widgets.Layout(width='700px')
                )
                
                checkbox.observe(lambda change, f=file.split('.')[0], reg=region: update_panels(change, f, reg), names='value')
                checkboxes.append(checkbox)
            except Exception:
                log_message(f"[WARNING] Arquivo de modelo com nome inesperado: {file}")

        update_interface()
    else:
        log_message("Nenhum arquivo de modelo encontrado.")

def update_panels(change, file, region):
    """Atualiza a lista de painéis de mosaico quando um checkbox de modelo é ativado/desativado."""
    # Note o uso das variáveis globais EMB_
    global EMB_mosaic_panels, EMB_selected_country, EMB_mosaic_checkboxes_dict, EMB_mosaic_checkbox_states
    
    if change['new']: # Se o checkbox for marcado
        panel = display_selected_mosaics_embedding(file, EMB_selected_country, region)
        EMB_mosaic_panels.append((file, region, panel))
    else: # Se o checkbox for desmarcado
        # Salva o estado atual
        if file in EMB_mosaic_checkboxes_dict:
            checkbox_list = EMB_mosaic_checkboxes_dict[file]
            EMB_mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]
        
        # Remove o painel
        EMB_mosaic_panels = [p for p in EMB_mosaic_panels if p[0] != file or p[1] != region]
        
    update_interface() # Esta função deve usar EMB_checkboxes e EMB_mosaic_panels


# CÉLULA DE EXECUÇÃO: Este bloco deve ser executado em uma célula separada do seu notebook.
# --------------------------------------------------------------------------------------
# Exemplo de uso:

# 1. Defina o país
# country_choice = 'guyana' # Defina seu país aqui
# log_message(f"Configurando interface para o país: {country_choice}")

# 2. Cria o painel de seleção de camadas
# layer_selector_panel = create_layer_selector_panel()
# display(layer_selector_panel)

# 3. Trigger o setup inicial da interface
# on_select_country(country_choice)

# O botão de execução é criado e exibido automaticamente dentro de update_interface
