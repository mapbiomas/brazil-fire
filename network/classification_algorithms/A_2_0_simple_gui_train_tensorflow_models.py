# last_update: '2025/04/24', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_2_0 - Simple Graphic User Interface for Training Models

# ====================================
# 📦 IMPORT LIBRARIES
# ====================================

import re
import time
import gcsfs
import ipywidgets as widgets
import sys
from IPython.display import display, clear_output
from ipywidgets import VBox, HBox

# TensorFlow in compatibility mode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ====================================
# 🌍 GLOBAL SETTINGS AND FILESYSTEM
# ====================================

bucket_name = 'mapbiomas-fire'
base_path = 'mapbiomas-fire/sudamerica/'

# Initialize Google Cloud Storage file system
fs = gcsfs.GCSFileSystem(project=bucket_name)

# ====================================
# 🎛️ INTERFACE CLASS
# ====================================

class TrainingInterface:
    """
    Interface for listing training sample files and triggering model training.
    """

    def __init__(self, country, preparation_function, log_func):
        self.country = country
        self.preparation_function = preparation_function
        self.log = log_func
        self.checkboxes = []
        self.training_files = []
        self.render_interface()

    def list_training_samples_folder(self):
        """
        List files in 'training_samples' folder for the selected country.
        """
        path = f"{base_path}{self.country}/training_samples/"
        try:
            return [file.split('/')[-1] for file in fs.ls(path) if file.split('/')[-1]]
        except FileNotFoundError:
            return []

    def get_active_checkbox(self):
        """
        Returns the label of the selected checkbox.
        """
        for checkbox in self.checkboxes:
            if checkbox.value:
                return checkbox.description
        return None
        
    def generate_checkboxes(self):
        """
        Generate checkboxes for unique model IDs, matching training script naming and flagging existing models.
        """
        seen_ids = set()
        fs.invalidate_cache()
        existing_models = self.list_existing_models()

        formatted_checkboxes = []

        for file in self.training_files:
            # Agora extrai versão, região e ano!
            match = re.search(r'_v(\d+)_.*?_r(\d+)(?:_[^_]+)*?_(\d{4})', file)
            if match:
                version = match.group(1)  # ex: 1, 2, 111
                region = match.group(2)   # ex: 01, 05
                model_id = f'v{version}_r{region}'
                model_ckpt = f'col1_{self.country}_{model_id}_rnn_lstm_ckpt'

                if model_id not in seen_ids:
                    label = f'trainings_{model_id}'
                    exists = model_ckpt in existing_models

                    if exists:
                        label += '⚠️'

                    checkbox = widgets.Checkbox(value=False, description=label, layout=widgets.Layout(width='auto'))
                    checkbox.observe(self.on_checkbox_click, names='value')
                    formatted_checkboxes.append(checkbox)
                    seen_ids.add(model_id)

                    seen_ids.add(model_id)

        self.checkboxes = formatted_checkboxes
        return widgets.VBox(formatted_checkboxes, layout=widgets.Layout(
            border='1px solid black', padding='10px', margin='10px 0'
        ))


    def list_existing_models(self):
        """
        Return a set of model checkpoint base names (excluding hyperparameters).
        """
        prefix_path = f"{base_path}{self.country}/models_col1/"
        try:
            files = fs.ls(prefix_path)
            model_files = [
                os.path.basename(f).split('.')[0]
                for f in files
                if 'ckpt' in f and 'hyperparameters' not in f
            ]
            return set(model_files)
        except Exception as e:
            self.log(f"[WARNING] Could not list existing models: {str(e)}")
            return set()



    def on_checkbox_click(self, change):
        """
        Ensure that only one checkbox is selected at a time.
        """
        if change.new:  # Checkbox was just selected
            for checkbox in self.checkboxes:
                if checkbox != change.owner:
                    checkbox.value = False

    def train_models_click(self, b):
        """
        Handles the training button click. Extracts selected checkbox info,
        matches training sample filenames, and calls the preparation function.
        """
        active_description = self.get_active_checkbox()
        if not active_description:
            self.log("[INFO] No checkbox selected.")
            return
    
        # Clean label: remove emojis and whitespace
        clean_label = active_description.replace('✅', '').replace('⚠️', '').strip()
        check_parts = clean_label.split('_')
    
        # Verifica se temos partes suficientes: trainings_v1_r01
        if len(check_parts) < 3:
            self.log(f"[ERROR] Unexpected checkbox label format: {clean_label}")
            return
        
        
        version = check_parts[1]  # ex: v1 ou v2
        region = check_parts[2]   # ex: r01 ou r04
    
        # Padrão: busca por arquivos que contenham essa região
        pattern = re.compile(rf".*_({version})_.*_{region}_.*\.tif")
    
        # Filtra arquivos correspondentes
        selected_files = [f for f in self.training_files if pattern.search(f)]
    
        if selected_files:
            self.log(f"[INFO] Selected files for training: {selected_files}")
            self.preparation_function(selected_files)
        else:
            self.log(f"[WARNING] No matching training samples found for region: {region}")
    
    def display_existing_models(self):
        """
        Display a scrollable list of existing models from the GCS bucket.
        """
        fs.invalidate_cache()
        existing = sorted(self.list_existing_models())
        output = widgets.Output(layout={'border': '1px solid green', 'height': '150px', 'overflow_y': 'scroll', 'margin': '10px 0'})
        display(widgets.HTML(value=f"<b>Existing trained models ({len(existing)}):</b>"))
        with output:
            for model in existing:
                print(f'  - {model}')
        display(output)

    def render_interface(self):
        """
        Renders the full interface: title, file list, checkboxes, button.
        """
        self.training_files = self.list_training_samples_folder()
        num_files = len(self.training_files)

        header = widgets.HTML(value=f"<b>Selected country: {self.country} ({num_files} files found)</b>")
        display(header)

        files_panel = widgets.Output(layout={'border': '1px solid black', 'height': '150px', 'overflow_y': 'scroll', 'margin': '10px 0'})
        with files_panel:
            for f in self.training_files:
                print(f'  - {f}')
        display(files_panel)

        if num_files == 0:
            display(widgets.HTML("<b style='color: red;'>No files found in 'training_samples'.</b>"))
            return

        # ⬇️ Show models before checkboxes
        self.display_existing_models()

        samples_title = widgets.HTML(value="<b>Sample by region, and versions available to run the training:</b>")
        display(samples_title)

        checkboxes_panel = self.generate_checkboxes()
        display(checkboxes_panel)

        train_button = widgets.Button(description="Train Models", button_style='success', layout=widgets.Layout(width='200px'))
        train_button.on_click(self.train_models_click)
        display(HBox([train_button], layout=widgets.Layout(justify_content='flex-start', margin='20px 0')))

        footer = widgets.HTML("<b style='color: orange;'>⚠️ Existing models will be overwritten if selected again.</b>")
        display(footer)

# ====================================
# 🚀 RUNNING THE INTERFACE
# ====================================
# TrainingInterface(
#     country=country,
#     preparation_function=sample_download_and_preparation,
#     log_func=log_message
# )
