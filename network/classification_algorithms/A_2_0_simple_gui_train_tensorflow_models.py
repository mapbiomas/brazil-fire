
import gcsfs
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
from ipywidgets import VBox, HBox
import time

import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style



bucketName = 'mapbiomas-fire'
pastaBase = 'mapbiomas-fire/sudamerica/'

# Initialize the Google Cloud Storage file system
fs = gcsfs.GCSFileSystem(project=bucketName)

# Function to list the content of the "training_samples" subfolder for each country
def list_training_samples_folder(country_folder):
    training_folder = f"{pastaBase}{country_folder}/training_samples/"
    try:
        files = fs.ls(training_folder)
        return [file.split('/')[-1] for file in files if file.split('/')[-1]]  # Remove empty items
    except FileNotFoundError:
        return []  # Return an empty list if the subfolder doesn't exist

def list_training_samples(files):
    check_select = get_active_checkbox().split('_')
    pattern = re.compile(rf".*{check_select[1]}.*{check_select[3]}.*\.tif")

    filtered_files = [file for file in files if pattern.search(file)]
    log_message(f"[INFO] Filtered files: {filtered_files}")  # Check the filtered files

    return filtered_files

def get_active_checkbox():
    for checkbox in checkboxes:
        if checkbox.value:  # Check if the checkbox is selected
            return checkbox.description  # Return the description of the active checkbox
    return None  # Return None if no checkbox is selected

# Function to display the content of "training_samples" when a country is selected
def select_country(country_name):

    # List and display the files in the "training_samples" folder
    training_files = list_training_samples_folder(country_name)
    num_files = len(training_files)

    # Display the total number of files and the selected country at the top
    country_title = widgets.HTML(value=f"<b>Selected country: {country_name} ({num_files} files found)</b>")
    display(country_title)

    # display(dropdown_countries)  # Re-display the dropdown

    # Scrollable panel for files
    files_panel = widgets.Output(layout={'border': '1px solid black', 'height': '150px', 'overflow_y': 'scroll', 'margin': '10px 0'})

    with files_panel:
        for file in training_files:
            print(f'  - {file}')
    display(files_panel)  # Display the scrollable panel

    if training_files:
        # Format the files
        formatted_list = []

        for file in training_files:
            split = file.split('_')  # Split the file name into parts
            if len(split) >= 6:  # Ensure there are enough parts for formatting
                formatted = f'trainings_{split[2]}_{split[4]}_{split[5]}'  # Custom formatting
                if formatted not in formatted_list:
                    formatted_list.append(formatted)  # Add if not already in the list

        formatted_files = formatted_list

        # Title for samples by sensor, region, and version
        num_samples = len(formatted_files)
        samples_title = widgets.HTML(value=f"<b>Samples by region, and version available to run the training ({num_samples} samples):</b>")
        display(samples_title)

        # Display checkboxes for each formatted file
        global checkboxes  # Access the checkboxes globally
        checkboxes = []

        def checkbox_click(change):
            # If the checkbox was selected (True value)
            if change.new:
                # Uncheck all other checkboxes
                for checkbox in checkboxes:
                    if checkbox != change.owner:  # Uncheck all except the current one
                        checkbox.value = False

        # Create checkboxes and add an observer
        for file in formatted_files:
            checkbox = widgets.Checkbox(value=False, description=file, layout=widgets.Layout(width='auto'))
            checkboxes.append(checkbox)
            checkbox.observe(checkbox_click, names='value')  # Adding observer

        # Panel to organize checkboxes in vertical columns without limiting height
        checkboxes_panel = widgets.VBox(checkboxes, layout=widgets.Layout(border='1px solid black', padding='10px', margin='10px 0'))
        display(checkboxes_panel)
        print("⚠️Attention, files that already exist, if selected, are reprocessed and overwrite the file at the final address.⚠️")

        # Buttons for simulation and training
        simulate_button = widgets.Button(description="Simulate Processing!", button_style='warning', layout=widgets.Layout(width='200px'))  # Yellow button
        train_button = widgets.Button(description="Train Models", button_style='success', layout=widgets.Layout(width='200px'))  # Green button

        # Function to handle the training button click
        def train_models_click(b):
            selected_samples = list_training_samples(training_files)
            if selected_samples:
                log_message(f"[INFO] Selected samples: {selected_samples}")  # Add print here
                sample_download_and_preparation(selected_samples, simulation=False)
            else:
                log_message("[INFO] No samples selected.")
        def simulate_processing_click(b):
            selected_samples = list_training_samples(training_files)
            if selected_samples:
                sample_download_and_preparation(selected_samples,simulation=True)
            else:
                log_message("No samples selected.")

        # Link the buttons to their respective functions
        simulate_button.on_click(simulate_processing_click)
        train_button.on_click(train_models_click)

        # Footer layout with both buttons side by side
        footer_layout = widgets.HBox([simulate_button, train_button], layout=widgets.Layout(justify_content='flex-start', margin='20px 0'))
        display(footer_layout)

    else:
        message = widgets.HTML(value="<b style='color: red;'>No files found in the folder 'training_samples'.</b>")
        display(message)

select_country(country)
