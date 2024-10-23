# last_update: '2024/10/23', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_0_2_log_algorithm_monitor.py 
### Step A_0_2 - Algoritmo para registrar logs de monitoramento da interface em um arquivo JSON na Google Cloud
import os
from datetime import datetime
import subprocess
import json

# Variáveis globais
log_file_path_local = None
bucket_log_folder = None
log_index = 0  # Variável global para armazenar o índice dos logs

# Variáveis de contexto globais (defina esses valores aqui ou no início do programa)
country = 'Brazil'
collection_name = 'Collection_1'
bucket_name = 'meu_bucket'

def log_message(message):
    """
    Grava uma nova mensagem de log no arquivo existente ou cria um novo arquivo de log na primeira execução.
    """
    global log_file_path_local, bucket_log_folder, log_index
    
    # Na primeira execução, cria o caminho para o log
    if log_file_path_local is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_folder, log_file_path_local, bucket_log_folder = create_log_paths(timestamp)
        
        # Verificar e criar o diretório local se necessário
        create_local_directory(log_folder)
    
    # Atualizar o índice de logs
    log_index += 1

    # Formatar a mensagem de log
    log_entry = format_log_entry(message, log_index)
    
    # Gravar a mensagem no arquivo de log local
    write_log_local(log_file_path_local, log_entry)
    
    # Subir o arquivo de log atualizado para o bucket no GCS
    upload_log_to_gcs(log_file_path_local, bucket_log_folder)


def create_log_paths(timestamp):
    """
    Cria os caminhos locais e no GCS para os arquivos de log.
    """
    log_folder = f'/content/{bucket_name}/sudamerica/{country}/classification_logs'
    log_file_name = f'burned_area_classification_log_{collection_name}_{country}_{timestamp}.log'
    log_file_path_local = os.path.join(log_folder, log_file_name)
    bucket_log_folder = f'gs://{bucket_name}/sudamerica/{country}/classification_logs/{log_file_name}'
    
    return log_folder, log_file_path_local, bucket_log_folder


def create_local_directory(log_folder):
    """
    Verifica se o diretório local existe, e o cria caso não exista.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print(f"[LOG INFO] Created local log directory: {log_folder}")
    else:
        print(f"[LOG INFO] Local log directory already exists: {log_folder}")


def format_log_entry(message, log_index):
    """
    Formata a mensagem de log com timestamp e adiciona um índice.
    """
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        "index": log_index,
        "timestamp": current_time,
        "message": message
    }
    # Retorna a entrada formatada como JSON
    return json.dumps(log_entry) + "\n"


def write_log_local(log_file_path_local, log_entry):
    """
    Escreve a mensagem de log no arquivo local.
    """
    with open(log_file_path_local, 'a') as log_file:
        log_file.write(log_entry)
    # print(f"[LOG INFO] Log written locally to {log_file_path_local}")


def upload_log_to_gcs(log_file_path_local, bucket_log_folder):
    """
    Sobe o arquivo de log para o bucket no Google Cloud Storage.
    """
    try:
        subprocess.check_call(f'gsutil cp {log_file_path_local} {bucket_log_folder}', shell=True)
        # print(f"[LOG INFO] Log file uploaded to GCS at {bucket_log_folder}")
    except subprocess.CalledProcessError as e:
        print(f"[LOG ERROR] Failed to upload log file to GCS: {str(e)}")


# Exemplo de como usar a função
# log_message('Processo de classificação iniciado')
# log_message('Nova etapa do processo concluída')
# log_message('Erro no processamento de dados')
