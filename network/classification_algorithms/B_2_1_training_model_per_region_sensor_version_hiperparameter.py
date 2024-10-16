import os
import numpy as np
import tensorflow as tf
from osgeo import gdal
import datetime
import gcsfs
from google.cloud import storage

# Inicializando o sistema de arquivos do Google Cloud Storage
fs = gcsfs.GCSFileSystem(project='seu_projeto_aqui')  # Altere conforme seu projeto

# Funções de carregamento e processamento de imagens
def load_image(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {image_path}. Verifique o caminho.")
    return dataset

def convert_to_array(dataset):
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(bands_data, axis=2)

def log_message(message, log_file="treinamento_log.txt"):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a") as log:
        log.write(f"{timestamp} - {message}\n")
    print(message)

# Função para fazer upload de modelos para o GCS
def upload_to_gcs(local_file_path, bucket_name, gcs_destination_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_destination_path)
    blob.upload_from_filename(local_file_path)
    log_message(f"[INFO] Arquivo {local_file_path} enviado para {gcs_destination_path} no bucket {bucket_name}.")

# Função para baixar arquivos do GCS
def baixar_arquivos_gcs(bucket_name, file_path, local_dest_path):
    """
    Usa gcsfs para copiar os arquivos do bucket GCS para o sistema de arquivos local.
    """
    try:
        with fs.open(f'{bucket_name}/{file_path}', 'rb') as src_file:
            with open(local_dest_path, 'wb') as dest_file:
                dest_file.write(src_file.read())
        log_message(f'[INFO] Arquivo baixado: {file_path} para {local_dest_path}')
    except Exception as e:
        log_message(f'[ERRO] Não foi possível baixar o arquivo {file_path}: {str(e)}')

# Função principal de treinamento
def treinar_modelo(versao, satelite, regiao, ano, bucket_name, country, hiperparametros=None, simulacao=False):
    """
    Função de treinamento com uma flag para simulação. Se simulacao=True, apenas emite prints simulados.
    """
    
    # Salvando em diretórios específicos no Colab
    folder_samples = f"/content/training_samples/{versao}_{satelite}_{regiao}"  # Define a pasta de amostras local
    folder_model = f"/content/models/{versao}_{satelite}_{regiao}"  # Pasta para salvar os modelos localmente
    
    if not os.path.exists(folder_samples):
        os.makedirs(folder_samples)
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    images_train_test = [f"samples_fire_{versao}_{satelite}_{regiao}_*.tif"]

    if simulacao:
        log_message(f"[SIMULACAO] Iniciando simulação do treinamento para versão: {versao}, satélite: {satelite}, região: {regiao}")
        return

    log_message(f"Iniciando o treinamento para versão: {versao}, satélite: {satelite}, região: {regiao}")

    # Hiperparâmetros padrão
    default_hiperparametros = {
        'lr': 0.001,
        'BATCH_SIZE': 1000,
        'N_ITER': 7000,
        'NUM_N_L1': 7,
        'NUM_N_L2': 14,
        'NUM_N_L3': 7,
        'NUM_N_L4': 14,
        'NUM_N_L5': 7,
        'NUM_CLASSES': 2
    }

    if hiperparametros:
        for param in default_hiperparametros:
            if param in hiperparametros:
                default_hiperparametros[param] = hiperparametros[param]

    log_message(f"Hiperparâmetros utilizados: {default_hiperparametros}")

    # Extração dos hiperparâmetros
    lr = default_hiperparametros['lr']
    BATCH_SIZE = default_hiperparametros['BATCH_SIZE']
    N_ITER = default_hiperparametros['N_ITER']
    NUM_CLASSES = default_hiperparametros['NUM_CLASSES']

    # 4.2 Processamento de Imagens de Treinamento e Teste
    all_data_train_test_vector = []

    # Ajuste: Baixar os arquivos do GCS com gcsfs
    for index, images in enumerate(images_train_test):
        log_message(f"[INFO] Tentando copiar arquivos com padrão: {images}")
        image_path_gcs = f"mapbiomas-fire/sudamerica/{country}/training_samples/{images}"
        local_path = f"{folder_samples}/{images}"
        
        # Aqui baixamos os arquivos do GCS para o Colab
        baixar_arquivos_gcs(bucket_name, image_path_gcs, local_path)

        images_name = glob.glob(f'{folder_samples}/{images}')
        if not images_name:
            log_message(f'[ERRO] Nenhuma imagem correspondente encontrada para o padrão: {images}')
            continue

        for image in images_name:
            log_message(f'Processando a imagem: {image}')
            try:
                dataset_train_test = load_image(image)
                data_train_test = convert_to_array(dataset_train_test)
                vector = data_train_test.reshape([data_train_test.shape[0] * data_train_test.shape[1], data_train_test.shape[2]])
                dataclean = vector[~np.isnan(vector).any(axis=1)]
                all_data_train_test_vector.append(dataclean)
            except Exception as e:
                log_message(f'Erro ao processar a imagem {image}: {str(e)}')
                continue

    if all_data_train_test_vector:
        data_train_test_vector = np.concatenate(all_data_train_test_vector)
        log_message(f"Dados concatenados: {data_train_test_vector.shape}")
    else:
        raise ValueError("Erro: Nenhum dado de treinamento ou teste disponível para concatenar.")

    def filter_valid_data_and_shuffle(data):
        valid_data = data[~np.isnan(data).any(axis=1)]
        np.random.shuffle(valid_data)
        return valid_data

    valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)
    bi = [0, 1, 2, 3]  # Índices para as bandas NBR
    li = 4  # Índice para o rótulo (classe)
    TRAIN_FRACTION = 0.7

    if valid_data_train_test.shape[0] < 2:
        raise ValueError("Erro: Dados insuficientes para dividir em treinamento e validação.")

    training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)
    training_data = valid_data_train_test[:training_size, :]
    validation_data = valid_data_train_test[training_size:, :]

    log_message(f"Tamanho do conjunto de validação: {validation_data.shape[0]} exemplos")

    data_mean = training_data[:, bi].mean(axis=0)
    data_std = training_data[:, bi].std(axis=0)

    # 5.2 Definição da Rede Neural Usando TensorFlow 2.x (Keras API)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(len(bi),)),  # Número de entradas
        tf.keras.layers.Dense(default_hiperparametros['NUM_N_L1'], activation='relu'),
        tf.keras.layers.Dense(default_hiperparametros['NUM_N_L2'], activation='relu'),
        tf.keras.layers.Dense(default_hiperparametros['NUM_N_L3'], activation='relu'),
        tf.keras.layers.Dense(default_hiperparametros['NUM_N_L4'], activation='relu'),
        tf.keras.layers.Dense(default_hiperparametros['NUM_N_L5'], activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    log_message('[INFO] Modelo compilado e pronto para treinamento.')

    model.fit(training_data[:, bi], training_data[:, li], validation_data=(validation_data[:, bi], validation_data[:, li]), epochs=10, batch_size=BATCH_SIZE)

    log_message('[INFO] Treinamento concluído.')

    model_path = f'{folder_model}/model.h5'
    model.save(model_path)
    
    gcs_model_path = f'sudamerica/{country}/models/{versao}_{satelite}_{regiao}/model.h5'
    upload_to_gcs(model_path, bucket_name, gcs_model_path)

    log_message(f'[INFO] Modelo final salvo em {gcs_model_path} no Google Cloud Storage.')
