import os
from datetime import datetime
import pytz  # Library to handle timezones
import subprocess
import json

# Global variables
log_file_path_local = None
bucket_log_folder = None
log_index = 0  # Global variable to store the log index

# Dictionary that maps countries to their respective timezones
timezone_switch = {
    'brazil': 'America/Sao_Paulo',
    'guyana': 'America/Guyana',
    'bolivia': 'America/La_Paz',
    'colombia': 'America/Bogota',
    'chile': 'America/Santiago',
    'peru': 'America/Lima',
    'paraguay': 'America/Asuncion'
}

# Country variable (can be modified as needed)
# country = 'brazil'  # Example: change to another country if needed

# Get the country's timezone using the dictionary
if country in timezone_switch:
    country_tz = pytz.timezone(timezone_switch[country])
else:
    # If the country is not in the dictionary, use UTC by default
    country_tz = pytz.UTC

def log_message(message):
    """
    Records a new log message in the existing file or creates a new log file on first execution.
    """
    global log_file_path_local, bucket_log_folder, log_index
    # On the first execution, create the log path
    if log_file_path_local is None:
        timestamp = datetime.now(country_tz).strftime('%Y-%m-%d_%H-%M-%S')
        log_folder, log_file_path_local, bucket_log_folder = create_log_paths(timestamp)
        
        # Check and create the local directory if necessary
        create_local_directory(log_folder)
    
    # Update the log index
    log_index += 1

    # Format the log message
    log_entry = format_log_entry(message, log_index)
    
    # Display in the desired format
    formatted_log = f"[LOG] [{log_index}] [{datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(formatted_log)
    
    # Write the message to the local log file
    write_log_local(log_file_path_local, log_entry)
    
    # Upload the updated log file to the GCS bucket
    upload_log_to_gcs(log_file_path_local, bucket_log_folder)

def create_log_paths(timestamp):
    """
    Creates the local and GCS paths for the log files.
    """
    log_folder = f'/content/{bucket_name}/sudamerica/{country}/classification_logs'
    log_file_name = f'burned_area_classification_log_{collection_name}_{country}_{timezone_switch[country].replace("/", "_").lower()}_{timestamp}.log'
    log_file_path_local = os.path.join(log_folder, log_file_name)
    bucket_log_folder = f'gs://{bucket_name}/sudamerica/{country}/classification_logs/{log_file_name}'
    
    return log_folder, log_file_path_local, bucket_log_folder

def create_local_directory(log_folder):
    """
    Checks if the local directory exists, and creates it if it does not.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"[LOG INFO] Created local log directory: {log_folder}")
    else:
        print(f"[LOG INFO] Local log directory already exists: {log_folder}")

def format_log_entry(message, log_index):
    """
    Formats the log message with a timestamp and adds an index.
    """
    # Check if the object is serializable; if not, convert it to a string
    if isinstance(message, (dict, list)):
        message = json.dumps(message, default=str)
    elif not isinstance(message, str):
        message = str(message)  # Convert any non-serializable object directly to string
    
    current_time = datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        "index": log_index,
        "timestamp": current_time,
        "message": message
    }
    # Return the entry formatted as JSON
    return json.dumps(log_entry) + "\n"

def write_log_local(log_file_path_local, log_entry):
    """
    Writes the log message to the local file.
    """
    with open(log_file_path_local, 'a') as log_file:
        log_file.write(log_entry)

def upload_log_to_gcs(log_file_path_local, bucket_log_folder):
    """
    Uploads the log file to the bucket on Google Cloud Storage.
    """
    try:
        subprocess.check_call(f'gsutil cp {log_file_path_local} {bucket_log_folder}', shell=True)
    except subprocess.CalledProcessError as e:
        print(f"[LOG ERROR] Failed to upload log file to GCS: {str(e)}")

# Example usage:
# log_message('Classification process started')
# log_message(['coll_guyana_v1_r3_rnn_lstm_ckpt'])
# log_message('mosaic_checkboxes_dict')
