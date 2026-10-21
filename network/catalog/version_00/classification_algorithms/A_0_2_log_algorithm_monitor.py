import os
from datetime import datetime
import pytz  # Library to handle timezones
import subprocess
import json
import psutil
import shutil

# Global variables
log_file_path_local = None
bucket_log_folder = None
log_index = 0  # Global variable to store the log index
header_dict = None  # Global variable to store header information

# Define necessary variables
# country = 'guyana'  # Example country; modify as needed
# collection_name = 'collection_1'
# source_name = 'IPAM'
# specified_timezone = None  # Set a specific timezone if needed, e.g., 'America/Sao_Paulo'
# bucket_name = 'your_bucket_name'

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

# Determine timezone, prioritizing specified_timezone if provided
if specified_timezone:
    country_tz = pytz.timezone(specified_timezone)
else:
    country_tz = pytz.timezone(timezone_switch.get(country, 'UTC'))

def create_header():
    """
    Creates a header for the log file with the initial date, timezone, country, source name, and collection name.
    Returns both a string representation and a dictionary for JSON logs.
    """
    global header_dict  # Ensure header_dict is available globally
    initial_date = datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')
    timezone_str = specified_timezone if specified_timezone else timezone_switch.get(country, 'UTC')
    header_str = (
        f"Processing started on: {initial_date}\n"
        f"Timezone: {timezone_str}\n"
        f"Country: {country}\n"
        f"Source: {source_name}\n"
        f"Collection: {collection_name}\n"
        "---------------------------------\n"
    )
    
    # Header dictionary for JSON logging
    header_dict = {
        "initial_date": initial_date,
        "timezone": timezone_str,
        "country": country,
        "source": source_name,
        "collection": collection_name
    }
    
    print(header_str)  # Print the header only once at the beginning
    return header_str

def log_message(message):
    """
    Records a new log message in the existing file or creates a new log file on first execution.
    Includes system information (RAM and disk) and header info in each log entry.
    """
    global log_file_path_local, bucket_log_folder, log_index, header_dict
    
    # On the first execution, create the log path, add the header, and write to the log file
    if log_file_path_local is None:
        timestamp = datetime.now(country_tz).strftime('%Y-%m-%d_%H-%M-%S')
        log_folder, log_file_path_local, bucket_log_folder = create_log_paths(timestamp)
        
        # Check and create the local directory if necessary
        create_local_directory(log_folder)
        
        # Write header to the log file
        header_str = create_header()
        with open(log_file_path_local, 'w') as log_file:
            log_file.write(header_str)
    
    # Update the log index
    log_index += 1

    # Get system information (RAM and Disk)
    system_info = get_system_info_compact()

    # Format the log message with system info and header for JSON format
    log_entry = format_log_entry(message, log_index, system_info)
    
    # Display a simple print of the log message, without repeating the header
    formatted_log = f"[LOG] [{log_index}] [{datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')}] {message} | {system_info}"
    print(formatted_log)
    
    # Write the message to the local log file
    write_log_local(log_file_path_local, log_entry)
    
    # Upload the updated log file to the GCS bucket
    upload_log_to_gcs(log_file_path_local, bucket_log_folder)

def get_system_info_compact():
    """
    Returns a compact system info string with RAM and Disk usage.
    Format: disk:x/10x, ram:y/10y
    """
    ram_info = psutil.virtual_memory()
    total_ram = ram_info.total / (1024 ** 3)  # Convert to GB
    available_ram = ram_info.available / (1024 ** 3)  # Convert to GB

    disk_info = shutil.disk_usage('/')
    total_disk = disk_info.total / (1024 ** 3)  # Convert to GB
    free_disk = disk_info.free / (1024 ** 3)  # Convert to GB

    system_info = (
        f"disk:{free_disk:.1f}/{total_disk:.1f}GB, ram:{available_ram:.1f}/{total_ram:.1f}GB"
    )

    return system_info

def create_log_paths(timestamp):
    """
    Creates the local and GCS paths for the log files without including timezone in the filename.
    """
    log_folder = f'/content/{bucket_name}/sudamerica/{country}/classification_logs'
    log_file_name = f'burned_area_classification_log_{collection_name}_{country}_{timestamp}.log'
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

def format_log_entry(message, log_index, system_info):
    """
    Formats the log message with a timestamp, an index, system info, and includes header information for JSON.
    """
    if isinstance(message, (dict, list)):
        message = json.dumps(message, default=str)
    elif not isinstance(message, str):
        message = str(message)
    
    current_time = datetime.now(country_tz).strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        "index": log_index,
        "timestamp": current_time,
        "message": message,
        "system_info": system_info,
        "header": header_dict  # Embed header information in each log entry
    }
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
