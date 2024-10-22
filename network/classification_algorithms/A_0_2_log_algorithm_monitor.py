# last_update: '2024/10/22', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_0_2_log_algorithm_monitor.py 
### Step A_0_2 - Algoritmo para registrar logs de monitoramento da interface em um arquivo JSON na Google Cloud

import os
from datetime import datetime
import time
import subprocess

# def log_message(message, country, collection_name, bucket_name):
def log_message(message):

    """
    Log messages to both a local file and a GCS bucket. Combines previous logs from the bucket 
    with new messages to create a continuous log file.

    Args:
    - message: The message to log.
    - country: The country name (part of log file naming convention).
    - collection_name: The collection name (part of log file naming convention).
    - bucket_name: The GCS bucket name to store the log file.
    """

    import os
    from datetime import datetime
    import time
    import subprocess

    # Current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{current_time}] {message}\n"
    print('[LOG]',log_entry)
    # Paths for log files
    log_folder = f'/content/{bucket_name}/sudamerica/{country}/classification_logs'
    log_file_name = f'burned_area_classification_log_{collection_name}_{country}.log'
    log_file_path_local = os.path.join(log_folder, log_file_name)

    # Create the local directory if it doesn't exist
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Path for the log file in the GCS bucket
    bucket_log_folder = f'gs://{bucket_name}/sudamerica/{country}/classification_logs/'
    gcs_log_file_path = f'{bucket_log_folder}{log_file_name}'

    # Check if log file exists in the bucket and download it
    try:
        # Try to copy the existing log file from GCS to the local folder (if it exists)
        subprocess.check_call(f'gsutil cp {gcs_log_file_path} {log_file_path_local}', shell=True)
        # print(f"[INFO] Existing log file downloaded from GCS: {gcs_log_file_path}")
    except subprocess.CalledProcessError:
        # If the file doesn't exist, proceed with creating a new one
        print(f"[LOG INFO] No existing log file found in GCS. A new log will be created.")

    # Write the new log entry to the local file (append mode)
    with open(log_file_path_local, 'a') as log_file:
        log_file.write(log_entry)

    # print(f"[LOG INFO] Log written locally to {log_file_path_local}")

    # Upload the updated log file back to GCS
    try:
        # Copy the updated log file to the GCS bucket
        subprocess.check_call(f'gsutil cp {log_file_path_local} {bucket_log_folder}', shell=True)
        # print(f"[LOG INFO] Log file updated on GCS at {bucket_log_folder}")
    except subprocess.CalledProcessError as e:
        print(f"[LOG ERROR] Failed to update log file on GCS: {str(e)}")
