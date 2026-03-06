# last_update: '2025/04/24', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_2_1_training_tensorflow_model_per_region.py
### Step A_2_1 - Functions for training TensorFlow models per region

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
# ====================================

import importlib

# Função para verificar e instalar bibliotecas
def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
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

import os
import sys
import re
import math
import time
import json
import glob
import subprocess
import numpy as np
import rasterio
from datetime import datetime

# TensorFlow 1.x modo compatível
import tensorflow.compat.v1 as tf
if tf.__version__.startswith('2'):
    tf.disable_v2_behavior()

import gcsfs
from google.cloud import storage
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ====================================
# 🌍 GLOBAL VARIABLES AND DIRECTORY SETUP
# ====================================


# Definir diretórios para o armazenamento de dados e saída do modelo
folder = f'/content/mapbiomas-fire/sudamerica/{country}'  # Diretório principal onde os dados são armazenados

folder_samples = f'{folder}/training_samples'  # Diretório para armazenamento de dados de amostra
folder_model = f'{folder}/models_col1'  # Diretório para armazenamento da saída dos modelos
folder_images = f'{folder}/tmp1'  # Diretório para armazenamento temporário de imagens
folder_mosaic = f'{folder}/mosaics_cog'  # Diretório para arquivos COG (Cloud-Optimized GeoTIFF)

if not os.path.exists(folder_samples):
    os.makedirs(folder_samples)

if not os.path.exists(folder_model):
    os.makedirs(folder_model)

if not os.path.exists(folder_images):
    os.makedirs(folder_images)

if not os.path.exists(folder_mosaic):
    os.makedirs(folder_mosaic)

# ====================================
# 🧠 CORE CLASSES (ModelTrainer, ImageProcessor, FileManager)
# ====================================

class ModelTrainer:
    def __init__(self, bucket_name, country, folder_model, get_active_checkbox_func):
        self.bucket_name = bucket_name
        self.country = country
        self.folder_model = folder_model
        self.get_active_checkbox = get_active_checkbox_func  # função global já existente

    def split_and_train(self, valid_data_train_test, bi, li):
        TRAIN_FRACTION = 0.7
        training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)

        training_data = valid_data_train_test[:training_size, :]
        validation_data = valid_data_train_test[training_size:, :]

        log_message(f"[INFO] Training set size: {training_data.shape[0]}")
        log_message(f"[INFO] Validation set size: {validation_data.shape[0]}")

        data_mean = training_data[:, bi].mean(axis=0)
        data_std = training_data[:, bi].std(axis=0)

        log_message(f"[INFO] Mean of training bands: {data_mean}")
        log_message(f"[INFO] Standard deviation of training bands: {data_std}")

        self.train_model(training_data, validation_data, bi, li, data_mean, data_std, training_size)

    def train_model(self, training_data, validation_data, bi, li, data_mean, data_std, training_size):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        lr = 0.001
        BATCH_SIZE = 1000
        N_ITER = 7000
        NUM_INPUT = len(bi)
        NUM_CLASSES = 2

        NUM_N_L1 = 7
        NUM_N_L2 = 14
        NUM_N_L3 = 7
        NUM_N_L4 = 14
        NUM_N_L5 = 7

        graph = tf.Graph()
        with graph.as_default():
            log_message(f"[INFO] Setting up the TensorFlow graph...")

            x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT], name='x_input')
            y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

            normalized = (x_input - data_mean) / data_std

            hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
            hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
            hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
            hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
            hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')
            logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input)
            )
            optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
            outputs = tf.argmax(logits, 1)
            correct_prediction = tf.equal(outputs, y_input)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        start_time = time.time()

        checkbox_label = self.get_active_checkbox().replace('⚠️', '').replace('✅', '').strip()
        split_name = checkbox_label.split('_')
        
        if len(split_name) < 3:
            log_message(f"[ERROR] Unexpected checkbox format: {checkbox_label}")
            return
        
        version = split_name[1]  # v1
        region = split_name[2]   # r01
        
        model_path = f'{self.folder_model}/col1_{self.country}_{version}_{region}_rnn_lstm_ckpt'
        json_path = f'{model_path}_hyperparameters.json'

        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            validation_dict = {x_input: validation_data[:, bi], y_input: validation_data[:, li]}

            for i in range(N_ITER + 1):
                batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]
                feed_dict = {x_input: batch[:, bi], y_input: batch[:, li]}
                sess.run(optimizer, feed_dict=feed_dict)

                if i % 100 == 0:
                    acc = sess.run(accuracy, feed_dict=validation_dict) * 100
                    saver.save(sess, model_path)
                    log_message(f"[PROGRESS] Iteration {i}/{N_ITER} - Validation Accuracy: {acc:.2f}%")

            hyperparameters = {
                'data_mean': data_mean.tolist(),
                'data_std': data_std.tolist(),
                'lr': lr,
                'NUM_N_L1': NUM_N_L1,
                'NUM_N_L2': NUM_N_L2,
                'NUM_N_L3': NUM_N_L3,
                'NUM_N_L4': NUM_N_L4,
                'NUM_N_L5': NUM_N_L5,
                'NUM_CLASSES': NUM_CLASSES,
                'NUM_INPUT': NUM_INPUT
            }

            with open(json_path, 'w') as json_file:
                json.dump(hyperparameters, json_file)
            log_message(f'[INFO] Hyperparameters saved to: {json_path}')

            bucket_model_path = f'gs://{self.bucket_name}/sudamerica/{self.country}/models_col1/'
            try:
                subprocess.check_call(f'gsutil cp {model_path}.* {json_path} {bucket_model_path}', shell=True)
                log_message(f'[INFO] Model uploaded to GCS at: {bucket_model_path}')
                time.sleep(2)
                fs.invalidate_cache()
            except subprocess.CalledProcessError as e:
                log_message(f'[ERROR] Upload failed: {str(e)}')

            duration = time.time() - start_time
            log_message(f"[INFO] Training completed in: {time.strftime('%H:%M:%S', time.gmtime(duration))}")
            log_message(f"[INFO] Final model saved at: {model_path}")

class ImageProcessor:
    def __init__(self, folder_samples, fs, log_func):
        self.folder_samples = folder_samples
        self.fs = fs
        self.log_message = log_func

    def load_image(self, image_path):
        try:
            dataset = rasterio.open(image_path)
            return dataset
        except rasterio.errors.RasterioIOError as e:
            raise FileNotFoundError(f"Erro ao carregar imagem {image_path}: {str(e)}")

    def convert_to_array(self, dataset):
        data = dataset.read()  # Formato: (bands, height, width)
        data = np.transpose(data, (1, 2, 0))  # Para (height, width, bands)
        return data

    def process_image(self, image_path):
        try:
            self.log_message(f"[INFO] Processando imagem: {image_path}")
            dataset = self.load_image(image_path)
            data = self.convert_to_array(dataset)
            vector = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
            cleaned = vector[~np.isnan(vector).any(axis=1)]
            return cleaned
        except Exception as e:
            self.log_message(f"[ERROR] Falha ao processar a imagem {image_path}: {str(e)}")
            return None

class FileManager:
    def __init__(self, bucket_name, country, folder_samples, fs, log_func):
        self.bucket_name = bucket_name
        self.country = country
        self.folder_samples = folder_samples
        self.fs = fs
        self.log_message = log_func

    def download_image(self, image):
        self.log_message(f"[INFO] Starting download of: {image}")
        download_command = f'gsutil -m cp gs://{self.bucket_name}/sudamerica/{self.country}/training_samples/{image} {self.folder_samples}/'
        process = subprocess.Popen(download_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()

        if process.returncode == 0:
            self.log_message(f"[SUCCESS] Download completed for {image}.")
            time.sleep(2)
            self.fs.invalidate_cache()
            return True
        else:
            _, stderr = process.communicate()
            self.log_message(f"[ERROR] Failed to download {image}: {stderr.decode()}")
            return False

    def monitor_file_progress(self, file_path):
        try:
            initial_size = os.path.getsize(file_path)
        except FileNotFoundError:
            initial_size = 0
        time.sleep(1)
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            current_size = 0
        return current_size - initial_size


# ====================================
# 🧰 SUPPORT FUNCTIONS (utils)
# ====================================


# Função otimizada para remover NaNs e embaralhar os dados
def filter_valid_data_and_shuffle(data):
    """Remove rows with NaN and shuffles the data, optimized."""
    # Remove NaNs de forma vetorizada, sem loops
    mask = np.all(~np.isnan(data), axis=1)  # Cria uma máscara que identifica as linhas válidas
    valid_data = data[mask]  # Aplica a máscara para filtrar as linhas válidas

    # Embaralha os dados de forma eficiente
    np.random.default_rng().shuffle(valid_data)  # Usando Generator mais rápido para embaralhamento
    return valid_data

def fully_connected_layer(input, n_neurons, activation=None):
    """
    Creates a fully connected layer.

    :param input: Input tensor from the previous layer
    :param n_neurons: Number of neurons in this layer
    :param activation: Activation function ('relu' or None)
    :return: Layer output with or without activation applied
    """
    input_size = input.get_shape().as_list()[1]  # Get input size (number of features)

    # Initialize weights (W) with a truncated normal distribution and initialize biases (b) with zeros
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))), name='W')
    b = tf.Variable(tf.zeros([n_neurons]), name='b')

    # Apply the linear transformation (Wx + b)
    layer = tf.matmul(input, W) + b

    # Apply activation function, if specified
    if activation == 'relu':
        layer = tf.nn.relu(layer)

    return layer
    
# ====================================
# 🚀 MAIN EXECUTION LOGIC
# ====================================
# Includes: sample_download_and_preparation()

# Main function for handling image downloads and processing
def sample_download_and_preparation(images_train_test):
    image_processor = ImageProcessor(folder_samples, fs, log_message)
    file_manager = FileManager(bucket_name, country, folder_samples, fs, log_message)
    all_data_train_test_vector = []

    log_message(f"[INFO] Starting image download and preparation for {len(images_train_test)} images...")

    with tqdm(total=len(images_train_test), desc="[INFO] Downloading and processing images") as pbar:
        for image in images_train_test:
            local_file = os.path.join(folder_samples, image)

            if os.path.exists(local_file):
                log_message(f"[INFO] The file {image} already exists. Skipping download, but processing it.")
                images_name = [local_file]
            else:
                success = file_manager.download_image(image)
                if not success:
                    continue
                images_name = [local_file]

            for img in images_name:
                processed_data = image_processor.process_image(img)
                if processed_data is not None:
                    all_data_train_test_vector.append(processed_data)

            pbar.update(1)

    if not all_data_train_test_vector:
        raise ValueError("[ERROR] No training or test data available for concatenation.")
    
    data_train_test_vector = np.concatenate(all_data_train_test_vector)
    log_message(f"[INFO] Concatenated data: {data_train_test_vector.shape}")
    
    valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)
    log_message(f"[INFO] Valid data after filtering: {valid_data_train_test.shape}")
    
    trainer = ModelTrainer(bucket_name, country, folder_model, interface.get_active_checkbox)
    trainer.split_and_train(valid_data_train_test, bi=[0, 1, 2, 3], li=4)

# ====================================
# 🚀 RUNNING THE INTERFACE
# ====================================
# Cria a interface e guarda em uma variável global
interface = TrainingInterface(
    country=country,
    preparation_function=sample_download_and_preparation,
    log_func=log_message
)

