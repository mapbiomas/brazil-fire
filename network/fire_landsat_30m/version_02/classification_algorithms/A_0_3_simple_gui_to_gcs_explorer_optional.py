# last_update: '2024/10/22', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_0_3_simple_gui_to_gcs_explorer_optional.py 
### Step A_0_3 - Optional step to visualize files in the folders of the Google Cloud Storage bucket

import gcsfs
import ipywidgets as widgets  # Biblioteca para criar widgets interativos em notebooks Jupyter
from IPython.display import display, HTML, clear_output  # Utilitários para exibir conteúdo em notebooks Jupyter
from ipywidgets import VBox, HBox  # Containers para organizar widgets em caixas verticais ou horizontais

# 3.1 Interface to explore files in Google Cloud Storage (optional)
fs = gcsfs.GCSFileSystem(project=bucket_name)

# Set the initial path for the countries
base_folder = 'mapbiomas-fire/sudamerica/'

# Function to list countries
def list_countries(base_folder):
    fs.invalidate_cache()
    folders = fs.ls(base_folder)
    countries = [folder.split('/')[-1] for folder in folders]
    return countries

# Function to list subfolders or files of a country, ignoring directories
def list_content(folder):
    fs.invalidate_cache()
    folders = fs.ls(folder)
    content = [folder.split('/')[-1] for folder in folders if not folder.endswith('/')]  # Ignore directories
    return content

# Function to count the number of files in a subfolder
def count_files(folder):
    files = list_content(folder)
    return len(files)

# Widget to select the country
dropdown_countries = widgets.Dropdown(
    options=list_countries(base_folder),
    description='Countries:',
    disabled=False,
)

# Horizontal layout to align panels side by side
panel_layout = widgets.Layout(display='flex', flex_flow='row', justify_content='space-between')

# Function to list files and create a dynamic panel for each subfolder
def create_file_panel(selected_subfolder):
    subfolder_path = f"{base_folder}{dropdown_countries.value}/{selected_subfolder}/"
    files = list_content(subfolder_path)
    panel = widgets.Output(layout={'border': '1px solid black', 'height': '200px', 'overflow': 'auto'})
    with panel:
        print(f"Subfolder: {selected_subfolder}")
        print(f"Number of files: {len(files)}")
        for file in files:
            print(f'"{file}",')
    return panel

# Function to manage the updating of panels based on the checkboxes
def update_panels(change):
    # Clear the display area of the panels
    clear_output(wait=True)
    display(dropdown_countries, selection_box)

    # List of active panels
    active_panels = []

    # Add panels for the selected subfolders
    for checkbox in subfolder_checkboxes:
        if checkbox.value:
            panel = create_file_panel(checkbox.description)
            active_panels.append(panel)

    # Display the panels side by side
    display(HBox(active_panels, layout=panel_layout))

# Function to be called when selecting a country
def on_country_selection(change):
    selected_country = change['new']
    country_folder = f"{base_folder}{selected_country}/"
    subfolders = list_content(country_folder)

    global subfolder_checkboxes
    subfolder_checkboxes = []

    # Create checkboxes for each subfolder
    for subfolder in subfolders:
        checkbox = widgets.Checkbox(value=False, description=subfolder)
        checkbox.observe(update_panels, names='value')
        subfolder_checkboxes.append(checkbox)

    # Update the selection box
    selection_box.children = subfolder_checkboxes
    clear_output(wait=True)
    display(dropdown_countries, selection_box)

# Container for the subfolder checkboxes
selection_box = widgets.VBox()

# Initially display the country selection
display(dropdown_countries, selection_box)

# Bind the event of value change in the country dropdown
dropdown_countries.observe(on_country_selection, names='value')
