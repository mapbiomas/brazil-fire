# last_update: '2024/10/22', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_2_1_training_tensorflow_model_per_region.py 
### Step A_2_1 - Functions for training TensorFlow models per region

import subprocess
import sys
import importlib
import os
from datetime import datetime
import time
import json
import numpy as np

# Definir diretórios para o armazenamento de dados e saída do modelo
folder = f'/content/mapbiomas-fire/sudamerica/{country}'  # Diretório principal onde os dados são armazenados

folder_samples = f'{folder}/training_samples'  # Diretório para armazenamento de dados de amostra
folder_model = f'{folder}/models_col1'  # Diretório para armazenamento da saída dos modelos
folder_images = f'{folder}/tmp1'  # Diretório para armazenamento temporário de imagens
folder_mosaic = f'{folder}/mosaics_cog'  # Diretório para arquivos COG (Cloud-Optimized GeoTIFF)

import os

if not os.path.exists(folder_samples):
    os.makedirs(folder_samples)

if not os.path.exists(folder_model):
    os.makedirs(folder_model)

if not os.path.exists(folder_images):
    os.makedirs(folder_images)

if not os.path.exists(folder_mosaic):
    os.makedirs(folder_mosaic)



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

import os
import numpy as np
import tensorflow as tf
from osgeo import gdal
import datetime
import gcsfs
from google.cloud import storage
import glob
import re  # Importa a biblioteca de expressões regulares
import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style
import math  # Mathematical functions
import subprocess
import time
from datetime import datetime
from tqdm import tqdm  # Biblioteca para barra de progresso
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import VBox, HBox
# 4.1 Functions for running the training

# Function to load an image using GDAL
def load_image(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

# Function to convert a GDAL dataset to a NumPy array
# def convert_to_array(dataset):
#     bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
#     return np.stack(bands_data, axis=2)  # Stack the bands along the Z axis
def convert_to_array(dataset):
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return np.nan_to_num(stacked_data, nan=0)  # Substitui NaN por 0 # !perguntar para a Vera se tudo bem substituir valores mask, por NaN, no uso do convert_to_array do treinamento e no da classificação


# Function to shuffle data and filter invalid values (NaN)
def filter_valid_data_and_shuffle(data):
    """Removes rows with NaN and shuffles the data."""
    # Filter valid data by removing rows with NaN
    valid_data = data[~np.isnan(data).any(axis=1)]
    np.random.shuffle(valid_data)  # Shuffle the data
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

# Function to monitor local file progress
def monitor_file_progress(file_path):
    try:
        initial_size = os.path.getsize(file_path)
    except FileNotFoundError:
        initial_size = 0  # If the file doesn't exist yet

    time.sleep(1)  # Waits 1 second before checking again

    try:
        current_size = os.path.getsize(file_path)
    except FileNotFoundError:
        current_size = 0  # If the file was removed or hasn't started downloading

    return current_size - initial_size  # Returns the size difference
import os
import subprocess
import numpy as np
from tqdm import tqdm

# Function to handle downloading an image
def download_image(image, local_file, simulation):
    if simulation:
        log_message(f"[SIMULATION] Skipping download of: {image}")
    else:
        log_message(f"[INFO] Starting download of: {image}")
        download_command = f'gsutil -m cp gs://{bucket_name}/sudamerica/{country}/training_samples/{image} {folder_samples}/'
        process = subprocess.Popen(download_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.wait()

        if process.returncode == 0:
            log_message(f"[SUCCESS] Download completed for {image}.")
        else:
            _, stderr = process.communicate()
            log_message(f"[ERROR] Failed to download {image}: {stderr.decode()}")
            return False

    return True

# Function to load and process an image
def process_image(image, simulation):
    if simulation:
        log_message(f"[SIMULATION] Processing image: {image}")
        return None

    try:
        log_message(f"[INFO] Processing image: {image}")
        dataset_train_test = load_image(image)
        # print('dataset_train_test',dataset_train_test.GetRasterBand(1).ReadAsArray())  # Se for GDAL

        data_train_test = convert_to_array(dataset_train_test)
        vector = data_train_test.reshape([data_train_test.shape[0] * data_train_test.shape[1], data_train_test.shape[2]])
        dataclean = vector[~np.isnan(vector).any(axis=1)]  # Remove NaN values
        return dataclean
    except Exception as e:
        log_message(f"[ERROR] Failed to process image {image}: {str(e)}")
        return None

# Main function for handling image downloads and processing
def sample_download_and_preparation(images_train_test, simulation=False):

    # List to store the data vectors of all training and test images
    all_data_train_test_vector = []

    if simulation:
        log_message(f"[SIMULATION] Starting simulation for {len(images_train_test)} images.")
    else:
        log_message(f"[INFO] Starting image download and preparation for {len(images_train_test)} images...")

    # Adding the progress bar with the total number of files to be downloaded
    with tqdm(total=len(images_train_test), desc="[INFO] Downloading and processing images") as pbar:
        for index, image in enumerate(images_train_test):
            local_file = os.path.join(folder_samples, image)

            # Skip download if the file exists, but process it
            if os.path.exists(local_file):
                if simulation:
                    log_message(f"[SIMULATION] The file {image} already exists. Skipping download.")
                else:
                    log_message(f"[INFO] The file {image} already exists. Skipping download, but processing it.")
                images_name = [local_file]  # Treat as available
            else:
                success = download_image(image, local_file, simulation)
                if not success:
                    continue

                images_name = [local_file] if not simulation else [image]

            # Process images (either downloaded or pre-existing)
            for img in images_name:
                processed_data = process_image(img, simulation)
                if processed_data is not None:
                    all_data_train_test_vector.append(processed_data)

            pbar.update(1)

    # Check if any data was added to the list before attempting to concatenate
    if not simulation and all_data_train_test_vector:
        data_train_test_vector = np.concatenate(all_data_train_test_vector)
        log_message(f"[INFO] Concatenated data: {data_train_test_vector.shape}")
    elif not simulation:
        raise ValueError("[ERROR] No training or test data available for concatenation.")

    if not simulation:
        # Filter and shuffle the data
        valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)
        log_message(f"[INFO] Valid data after filtering: {valid_data_train_test.shape}")

        # Additional data splitting and training logic goes here
        split_data_for_training(valid_data_train_test)
    else:
        log_message("[SIMULATION] Simulation completed.")

def split_data_for_training(valid_data_train_test):
    # Indices of input features (NBR bands) and label (class)
    bi = [0, 1, 2, 3]  # Indices for NBR bands
    li = 4  # Index for the label (class)

    TRAIN_FRACTION = 0.7  # Proportion of data to be used for training

    # Check if there is enough data to perform the split
    if valid_data_train_test.shape[0] < 2:
        raise ValueError("[ERROR] Insufficient data to split into training and validation.")

    # Calculate the size of the training dataset
    training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)
    log_message(f"[INFO] Training set size: {training_size} examples")

    # Split the data into training and validation sets
    training_data = valid_data_train_test[:training_size, :]
    validation_data = valid_data_train_test[training_size:, :]

    log_message(f"[INFO] Validation set size: {validation_data.shape[0]} examples")

    # Calculate the mean and standard deviation of each band (NBR) in the training set
    data_mean = training_data[:, bi].mean(axis=0)
    data_std = training_data[:, bi].std(axis=0)
    log_message(f"[INFO] Mean of training bands: {data_mean}")
    log_message(f"[INFO] Standard deviation of training bands: {data_std}")

    # Start model training, passing training_size
    train_model(training_data, validation_data, bi, li, data_mean, data_std, training_size)

# Function to train the model and save hyperparameters
def train_model(training_data, validation_data, bi, li, data_mean, data_std, training_size):
    # ### HYPERPARAMETERS ###

    # Learning rate for the optimizer
    lr = 0.001

    # Batch size for training
    BATCH_SIZE = 1000

    # Number of training iterations
    N_ITER = 7000

    # Number of input features (NBR bands)
    NUM_INPUT = len(bi)

    # Definition of neurons in each hidden layer
    NUM_N_L1 = 7  # Neurons in the first hidden layer
    NUM_N_L2 = 14  # Neurons in the second hidden layer
    NUM_N_L3 = 7  # Neurons in the third hidden layer
    NUM_N_L4 = 14  # Neurons in the fourth hidden layer
    NUM_N_L5 = 7  # Neurons in the fifth hidden layer

    # Number of output classes (e.g., fire vs. no fire)
    NUM_CLASSES = 2

    # Creating a new TensorFlow computational graph
    graph = tf.Graph()
    with graph.as_default():  # Set the graph as the default for operations

        log_message(f"[INFO] Setting up the TensorFlow graph...")

        # Define placeholders for input data and labels
        x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT], name='x_input')  # Placeholder for input data
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')  # Placeholder for labels (class)

        # Normalize input data using the previously calculated mean and standard deviation
        normalized = (x_input - data_mean) / data_std

        # Build the neural network with several fully connected layers
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        """Additional hidden layers can be added here if necessary"""

        # Final output layer to produce the logits (raw values for each class)
        logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

        # Define the loss function: softmax cross-entropy (for multiclass classification)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )

        # Define the optimizer: Adam with the specified learning rate
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        # Operation to get the predicted class (the class with the highest logit)
        outputs = tf.argmax(logits, 1, name='predicted_class')

        # Accuracy metric: proportion of correct predictions
        correct_prediction = tf.equal(outputs, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # Initialize all variables in the graph
        init = tf.global_variables_initializer()

        # Define the saver to save the model state during training
        saver = tf.train.Saver()

        log_message(f"[INFO] TensorFlow graph setup complete.")

    # Record the start time of training
    start_time = time.time()

    # Configure GPU options to limit memory usage (optional)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)

    # Final save after training is complete
    split_name = get_active_checkbox().split('_')

    # Save the model locally and upload to GCS, including the hyperparameters JSON
    model_path = f'{folder_model}/col1_{country}_{split_name[1]}_{split_name[3]}_rnn_lstm_ckpt'
    json_path = f'{model_path}_hyperparameters.json'

    # Start a TensorFlow session to execute the graph
    log_message('[INFO] Starting training session with GPU memory limited to 66.66% of available memory...')
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)  # Initialize all variables
        log_message('[INFO] Initial variables loaded, session started.')

        # Validation data dictionary
        validation_dict = {
            x_input: validation_data[:, bi],
            y_input: validation_data[:, li]
        }

        log_message(f'[INFO] Starting training loop with {N_ITER} iterations...')

        # Training loop: iterate over the specified number of iterations
        for i in range(N_ITER + 1):
            # Select a random batch of training data
            batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]

            # Create the input dictionary for this batch
            feed_dict = {
                x_input: batch[:, bi],
                y_input: batch[:, li]
            }

            # Run one step of the optimizer (training step)
            optimizer.run(feed_dict=feed_dict)

            # Every 100 iterations, evaluate accuracy asave model 
            if i % 100 == 0:
                # Calculate validation accuracy
                acc = accuracy.eval(validation_dict) * 100
                # Save model in TensorFlow session
                saver.save(sess, model_path)

                log_message(f'[PROGRESS] Iteration {i}/{N_ITER} - Validation Accuracy: {acc:.2f}%')

        # Save the hyperparameters to the JSON file locally
        hyperparameters = {
            'data_mean': data_mean.tolist(),
            'data_std': data_std.tolist(),
            'NUM_N_L1': NUM_N_L1,
            'NUM_N_L2': NUM_N_L2,
            'NUM_N_L3': NUM_N_L3,
            'NUM_N_L4': NUM_N_L4,
            'NUM_N_L5': NUM_N_L5,
            'NUM_CLASSES': NUM_CLASSES
        }

        # Save the hyperparameters JSON locally
        with open(json_path, 'w') as json_file:
            json.dump(hyperparameters, json_file)
        log_message(f'[INFO] Hyperparameters saved to JSON file: {json_path}')


        # Upload model files and JSON to GCS only after training completes
        bucket_model_path = f'gs://{bucket_name}/sudamerica/{country}/models_col1/'

        try:
            subprocess.check_call(f'gsutil cp {model_path}.* {json_path} {bucket_model_path}', shell=True)
            log_message(f'[INFO] Model and hyperparameters successfully uploaded to GCS at {bucket_model_path}')
        except subprocess.CalledProcessError as e:
            log_message(f'[ERROR] Failed to upload model or hyperparameters to GCS: {str(e)}')

        # End of training process
        end_time = time.time()
        training_time = end_time - start_time
        log_message(f'[INFO] Total training time: {time.strftime("%H:%M:%S", time.gmtime(training_time))}')

        # Final model save message
        log_message(f'[INFO] Final model saved at: {model_path}')
