import subprocess
import sys
import importlib
import os
from datetime import datetime
import time


# Função para verificar e instalar bibliotecas
def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        clear_console()

# Função para instalar pacotes do sistema via apt-get
def apt_get_install(package):
    subprocess.check_call(['sudo', 'apt-get', 'install', '-y', package], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    clear_console()

# Função para limpar o console
def clear_console():
    # Limpa o console de acordo com o sistema operacional
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # MacOS/Linux
        os.system('clear')

# Verificar e instalar pacotes Python
install_and_import('rasterio')
install_and_import('gcsfs')
install_and_import('ipywidgets')

# Verificar e instalar pacotes Python
install_and_import('rasterio')
install_and_import('gcsfs')
install_and_import('ipywidgets')

# Instalar dependências de sistema (GDAL)
apt_get_install('libgdal-dev')
apt_get_install('gdal-bin')
apt_get_install('python3-gdal')

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import VBox, HBox


import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee  # For Google Earth Engine integration
from tqdm import tqdm  # For progress bars
import time
import math
from shapely.geometry import shape, box, mapping
import shutil  # For file and folder operations
import datetime  # For handling timestamps and date operations

# Configuration for Google Cloud Storage
bucket_name = 'mapbiomas-fire'
base_folder = 'mapbiomas-fire/sudamerica/'

# Initialize the Google Cloud Storage file system
fs = gcsfs.GCSFileSystem(project=bucket_name)

def list_model_training_samples(country_folder):
    """
    Lists all model files from the 'models_col1' folder in the bucket for the selected country.

    Args:
    - country_folder (str): The folder path for the specific country within the GCS structure.

    Returns:
    - tuple: A list of file names from the 'models_col1' folder and the total file count.
    """
    training_folder = f"{base_folder}{country_folder}/models_col1/"
    try:
        files = fs.ls(training_folder)
        return [file.split('/')[-1] for file in files if file.split('/')[-1]], len(files)  # Extract just file names and return file count
    except FileNotFoundError:
        return [], 0  # Return an empty list if the folder doesn't exist

def list_mosaics(country_folder, region):
    """
    Lists mosaic files (COG) from the 'mosaics_col1_cog' folder in the bucket, filtered by the given region.

    Args:
    - country_folder (str): The folder path for the specific country.
    - region (str): The region filter (e.g., r1, r2).

    Returns:
    - tuple: A list of filtered mosaic file names from the 'mosaics_col1_cog' folder and the total file count.
    """
    mosaics_folder = f"{base_folder}{country_folder}/mosaics_col1_cog/"
    try:
        files = fs.ls(mosaics_folder)
        return [file.split('/')[-1] for file in files if f"_{region}_" in file], len(files)
    except FileNotFoundError:
        return [], 0  # Return an empty list if the folder doesn't exist

def list_classified(country_folder):
    """
    Lists the classified files from the 'result_classified' folder in the bucket.

    Args:
    - country_folder (str): The folder path for the specific country.

    Returns:
    - tuple: A list of file names from the 'result_classified' folder and the total file count.
    """
    classified_folder = f"{base_folder}{country_folder}/result_classified/"
    try:
        files = fs.ls(classified_folder)
        return [file.split('/')[-1] for file in files], len(files)
    except FileNotFoundError:
        return [], 0  # Return an empty list if the folder doesn't exist

def verify_classified(country_folder, mosaic_file):
    """
    Checks if a mosaic has already been classified in the GCS bucket by comparing with flexible naming patterns.

    Args:
    - country_folder (str): The path to the country's folder in GCS.
    - mosaic_file (str): The name of the mosaic file to verify (e.g., l78_bolivia_r1_2013_cog.tif).

    Returns:
    - bool: True if the file is already classified in the bucket, False otherwise.
    """
    classified_folder = f"{base_folder}{country_folder}/result_classified/"

    # Extract relevant parts from the mosaic file name
    parts = mosaic_file.split('_')
    sat = parts[0]        # Satellite (e.g., 'l78')
    region = parts[2]     # Region (e.g., 'r1')
    year = parts[3]       # Year (e.g., '2013')

    # Expected classified file name pattern: 'burned_area_{country}_{sat}_v{version}_region{region}_{year}.tif'
    # Search the GCS bucket for files matching this pattern.

    try:
        # List files in the 'result_classified' folder directly in the bucket
        classified_files = fs.ls(classified_folder)

        # Check if there is a classified file that corresponds to this mosaic
        for classified_file in classified_files:
            classified_name = classified_file.split('/')[-1]  # Get only the file name
            # Extract parts from the classified file name
            if (sat in classified_name and
                f"region{region[1:]}" in classified_name and  # Remove 'r' from region to match 'region'
                year in classified_name):
                return True  # Corresponding file found

        return False  # No corresponding file found
    except FileNotFoundError:
        # If the folder doesn't exist, return False (not classified)
        return False
# Global dictionary to store mosaic checkboxes for each model
mosaic_checkboxes_dict = {}

def display_selected_mosaics(model, selected_country, region):
    """
    Displays the list of available mosaics for a specific model and region, with a checkbox
    to select or deselect all mosaics within the subpanel.

    Args:
    - model (str): Name of the selected model.
    - selected_country (str): Selected country.
    - region (str): Region of interest.

    Returns:
    - VBox: A container with the available mosaic checkboxes and the file count.
    """
    mosaic_files, mosaic_count = list_mosaics(selected_country, region)
    classified_files, classified_count = list_classified(selected_country)

    # Create mosaic panel
    mosaics_panel = widgets.Output(layout={'border': '1px solid black', 'height': '200px', 'overflow_y': 'scroll'})

    # List of checkboxes for mosaics
    checkboxes_mosaics = []

    # Retrieve saved states for checkboxes if they exist
    saved_states = mosaic_checkbox_states.get(model, None)

    with mosaics_panel:
        if mosaic_files:
            for idx, file in enumerate(mosaic_files):
                # Check if the mosaic has already been classified
                classified = verify_classified(selected_country, file)

                # Create checkbox for the mosaic; show warning ⚠️ if already classified
                checkbox_mosaic = widgets.Checkbox(
                    value=not classified,  # Default state
                    description=file + (" ⚠️" if classified else "")
                )

                # If there are saved states, restore the checkbox value
                if saved_states and idx < len(saved_states):
                    checkbox_mosaic.value = saved_states[idx]

                checkboxes_mosaics.append(checkbox_mosaic)
                display(checkbox_mosaic)
        else:
            log_message(f"No mosaics found for region {region}")

    # Store checkboxes for this specific model globally for later access
    mosaic_checkboxes_dict[model] = checkboxes_mosaics

    # Function to select or deselect all checkboxes
    def toggle_select_all(change):
        for checkbox in checkboxes_mosaics:
            checkbox.value = change['new']

    # Create "Select All" checkbox
    select_all_checkbox = widgets.Checkbox(value=False, description="Select All")
    select_all_checkbox.observe(toggle_select_all, names='value')

    # Legend panel for classified mosaics
    legend_panel = widgets.Output(layout={'border': '1px solid black', 'padding': '5px', 'margin-top': '10px'})
    with legend_panel:
        print("⚠️ Files already classified. They will overwrite previous classifications if the checkbox remains checked.")

    # Return a VBox containing the select all checkbox and individual mosaic checkboxes
    return widgets.VBox([select_all_checkbox, mosaics_panel, legend_panel])

def update_interface():
    """
    Updates the graphical interface based on the selected mosaic panels.
    """
    clear_output(wait=True)

    # Display model checkboxes
    display(VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0', width='700px')))

    # Display mosaic panels, ensuring they are unique
    mosaic_panels_widgets = [panel[2] for panel in mosaic_panels]
    display(HBox(mosaic_panels_widgets, layout=widgets.Layout(margin='10px 0', display='flex', flex_flow='row', overflow_x='auto')))

    # Display buttons at the end of the interface
    footer_layout = widgets.HBox(
        [simulate_button, classify_button],
        layout=widgets.Layout(justify_content='flex-start', margin='20px 0')
    )
    display(footer_layout)

def collect_selected_models():
    """
    Collects all selected model files from the checkboxes.

    Returns:
    - list: A list of selected model file names.
    """
    # Gather the names of models where the checkboxes are selected (checked)
    selected_models = [checkbox.description for checkbox in checkboxes if checkbox.value]
    return selected_models

def simulate_processing_click(b):
    """
    Simula o processamento de classificação de áreas queimadas com base nos modelos e mosaicos selecionados.

    Ele prepara a lista de objetos da mesma forma que `classify_burned_area_click`, mas executa apenas a simulação,
    sem realizar o processamento real.
    """
    selected_models = collect_selected_models()  # Coleta os modelos selecionados dos checkboxes
    models_to_simulate = []  # Essa lista vai armazenar os objetos para a simulação

    if selected_models:
        for model in selected_models:
            log_message(f'Simulando o processamento do modelo: {model}')

            # Normaliza o nome do modelo para garantir que estamos usando o formato correto
            model_key = f"{model}.meta" if not model.endswith('.meta') else model

            # Verifica se temos os mosaicos associados ao modelo
            if model_key in mosaic_checkboxes_dict:
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]

                # Coleta os mosaicos selecionados
                selected_mosaics = [cb.description.replace(" ⚠️", "").strip() for cb in mosaic_checkboxes if cb.value]
                log_message(f'Mosaicos selecionados para simulação: {selected_mosaics}')

                if not selected_mosaics:
                    log_message(f"Nenhum mosaico selecionado para o modelo: {model}")
                    continue

                # Cria o objeto do modelo para a simulação
                model_obj = {
                    "model": model,  # Nome do modelo
                    "mosaics": selected_mosaics,  # Lista de mosaicos selecionados
                    "simulation": True  # Marca como simulação
                }

                # Adiciona à lista de simulação
                models_to_simulate.append(model_obj)
            else:
                log_message(f'Nenhum mosaico encontrado para o modelo: {model_key}')

        # Se tivermos modelos para simular, chamamos a render_classify_models
        if models_to_simulate:
            log_message(f'Chamando render_classify_models para simulação com: {models_to_simulate}')
            render_classify_models(models_to_simulate)  # Chamamos a função com a lista preparada para simulação
        else:
            log_message("Nenhum mosaico foi selecionado para nenhum modelo.")
    else:
        log_message("Nenhum modelo selecionado.")  # Caso nenhum modelo tenha sido selecionado


def classify_burned_area_click(b):
    """
    Handles the classification of burned areas based on the selected models when the button is clicked.

    It prepares a list of objects containing the model name, a list of selected mosaic file names (from checkboxes),
    and a flag indicating whether it is a simulation or not. It then triggers the rendering/classification process for each model.
    """

    selected_models = collect_selected_models()  # Collect the selected models from checkboxes
    models_to_classify = []  # This will store the objects to pass to render_classify_models
    log_message('selected_models')
    log_message(selected_models)
    log_message('mosaic_checkboxes_dict')
    log_message(mosaic_checkboxes_dict)

    if selected_models:
        for model in selected_models:
            log_message(f'Processing model: {model}')
            # Normalize the model key by ensuring it has ".meta"
            model_key = f"{model}.meta" if not model.endswith('.meta') else model  # Ensure the model key has ".meta"
            log_message(f'Checking for model_key: {model_key}')

            # Retrieve the mosaic checkboxes for the current model
            if model_key in mosaic_checkboxes_dict:
                log_message(f'Found mosaic checkboxes for model: {model_key}')
                mosaic_checkboxes = mosaic_checkboxes_dict[model_key]  # Get the checkboxes for the current model
                log_message('mosaic_checkboxes')
                log_message(mosaic_checkboxes)

                # Get selected mosaics (checkboxes that are checked)
                selected_mosaics = [cb.description.replace(" ⚠️", "").strip() for cb in mosaic_checkboxes if cb.value]

                # Debugging output to ensure we are capturing the selected mosaics
                log_message(f"Selected mosaics for model {model_key}: {selected_mosaics}")

                if not selected_mosaics:
                    log_message(f"No mosaics selected for model: {model_key}")
                    continue

                # Build the model object to classify
                model_obj = {
                    "model": model,  # The name of the model
                    "mosaics": selected_mosaics,  # List of selected mosaic file names (without " ⚠️")
                    "simulation": False  # Set this flag as needed; you might want to add logic to set it dynamically
                }

                # Append the model object to the list
                models_to_classify.append(model_obj)
            else:
                log_message(f"No mosaics found for model: {model_key}")  # This shouldn't happen unless there's a bug

        # If we have models to classify, call render_classify_models with the list of objects
        if models_to_classify:
            log_message(f"Calling render_classify_models with: {models_to_classify}")  # Debugging output
            render_classify_models(models_to_classify)  # Call the function with the built data structure
        else:
            log_message("No mosaics were selected for any models.")  # No mosaics were selected for any models
    else:
        log_message("No models selected.")  # No models were selected

def on_select_country(country_name):
    """
    Handles the selection of a country and displays the available models for that country.

    Args:
    - country_name (str): The name of the selected country.
    """
    global selected_country  # Make the variable global so it can be accessed in other functions
    selected_country = country_name

    # List files in the 'models_col1' folder
    training_files, file_count = list_model_training_samples(country_name)

    # If there are files, create checkboxes for each model
    if training_files:
        global checkboxes, mosaic_panels
        checkboxes = []
        mosaic_panels = []  # Stores the mosaic panels

        # Create checkboxes for each file in 'models_col1'
        for file in training_files:
            if file.split('.')[1] == 'meta':  # Ensure working with 'meta' files
                region = file.split('_')[-4].split('.')[0]  # Extract region (e.g., r1, r2) from the file name
                checkbox = widgets.Checkbox(
                    value=False,
                    description=file.split('.')[0],
                    layout=widgets.Layout(width='700px')  # Adjust width
                )
                # Observe changes in the checkbox and update panels accordingly
                checkbox.observe(lambda change, f=file, reg=region: update_panels(change, f, reg), names='value')
                checkboxes.append(checkbox)

        # Initial interface update
        update_interface()

    else:
        # Display error message if no files are found
        message = widgets.HTML(
            value=f"<b style='color: red;'>No files found in the 'models' folder (Total: {file_count}).</b>"
        )
        clear_output(wait=True)
        display(message)

# Dictionary to store the state of mosaic checkboxes for each model
mosaic_checkbox_states = {}

def update_panels(change, file, region):
    """
    Updates the list of mosaic panels when a model checkbox is toggled.

    Args:
    - change (dict): The change event of the checkbox state.
    - file (str): The name of the model file.
    - region (str): The selected region.
    """
    global mosaic_panels, selected_country  # Ensure accessing the global variables

    if change['new']:  # If the checkbox is checked
        panel = display_selected_mosaics(file, selected_country, region)
        mosaic_panels.append((file, region, panel))
    else:  # If the checkbox is unchecked
        # Save the current state of checkboxes for this model
        if file in mosaic_checkboxes_dict:
            checkbox_list = mosaic_checkboxes_dict[file]
            # Save the checkbox values (True/False) in mosaic_checkbox_states
            mosaic_checkbox_states[file] = [cb.value for cb in checkbox_list]

        # Remove the corresponding panel
        mosaic_panels = [p for p in mosaic_panels if p[0] != file or p[1] != region]

    # Update the interface
    update_interface()

# Add buttons at the bottom of the interface
simulate_button = widgets.Button(
    description="Simulate Processing!",
    button_style='warning',
    layout=widgets.Layout(width='200px')  # Yellow button for simulation
)
classify_button = widgets.Button(
    description="Classify Burned Area",
    button_style='success',
    layout=widgets.Layout(width='200px')  # Green button for classification
)

# Link the "Simulate Processing" button to its event handler
simulate_button.on_click(simulate_processing_click)

# Link the "Classify Burned Area" button to its event handler
classify_button.on_click(classify_burned_area_click)

# Trigger the initial interface setup by selecting the country
on_select_country(country)
