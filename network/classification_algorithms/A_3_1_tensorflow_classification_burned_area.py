# last_update: '2024/10/23', github:'mapbiomas/brazil-fire', source: 'IPAM', contact: 'contato@mapbiomas.org'
# MapBiomas Fire Classification Algorithms Step A_3_1_tensorflow_classification_burned_area.py 
### Step A_3_1 - Functions for TensorFlow classification of burned areas
import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Usar somente a versão compatível 1.x
from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee  # For Google Earth Engine integration
from tqdm import tqdm  # For progress bars
import time
from datetime import datetime
import math
from shapely.geometry import shape, box, mapping
import shutil  # For file and folder operations
import json
import subprocess
import numpy as np

# Function to load an image using GDAL
def load_image(image_path):
    log_message(f"[INFO] Loading image from path: {image_path}")
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

# Function to convert a GDAL dataset to a NumPy array
def convert_to_array(dataset):
    log_message(f"[INFO] Converting dataset to NumPy array")
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return stacked_data
    # return np.nan_to_num(stacked_data, nan=0)

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    log_message(f"[INFO] Reshaping classified data back to image format")
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to reshape classified data into a single pixel vector
def reshape_single_vector(data_classify):
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])


# Function to apply spatial filtering on classified images
def filter_spatial(output_image_data):
    log_message(f"[INFO] Applying spatial filtering on classified image")
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))
    # **Converte para uint8 antes de retornar**
    return close_image.astype('uint8')

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    log_message(f"[INFO] Converting array to GeoTIFF raster: {output_image_name}")
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    
    # **Adicione opções de criação para compressão e altere o tipo de dados**
    options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Byte, options=options)
    
    # **Certifique-se de que os dados sejam do tipo uint8**
    image_data_scene_uint8 = image_data_scene.astype('uint8')
    outDs.GetRasterBand(1).WriteArray(image_data_scene_uint8)
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Release the output dataset from memory
    log_message(f"[INFO] Raster conversion completed and saved as: {output_image_name}")

# Function to convert meters into degrees based on latitude
def meters_to_degrees(meters, latitude):
    log_message(f"[INFO] Converting meters to degrees based on latitude: {latitude}")
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Function to expand geometry with a buffer in meters
def expand_geometry(geometry, buffer_distance_meters=50):
    log_message(f"[INFO] Expanding geometry by buffer of {buffer_distance_meters} meters")
    geom = shape(geometry)
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Function to check if there is a significant intersection between the geometry and the image
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    log_message(f"[INFO] Checking for significant intersection with minimum area of {min_intersection_area}")
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area
import time

# Function to clip an image based on the provided geometry
def clip_image_by_grid(geom, image, output, buffer_distance_meters=100, max_attempts=5, retry_delay=5):
    attempt = 0
    while attempt < max_attempts:
        try:
            log_message(f"[INFO] Attempt {attempt+1}/{max_attempts} to clip image: {image}")
            with rasterio.open(image) as src:
                expanded_geom = expand_geometry(geom, buffer_distance_meters)
                if has_significant_intersection(expanded_geom, src.bounds):
                    out_image, out_transform = mask(src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
                    
                    # **Atualize os metadados aqui**
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    
                    with rasterio.open(output, 'w', **out_meta) as dest:
                        dest.write(out_image)
                    log_message(f"[INFO] Image clipped successfully: {output}")
                    return True  # Clipping successful
                else:
                    log_message(f"[INFO] Insufficient overlap for clipping: {image}")
                    return False  # No significant intersection, no need to retry
        except Exception as e:
            log_message(f"[ERROR] Error during clipping: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1

    log_message(f"[ERROR] Failed to clip image after {max_attempts} attempts: {image}")
    return False  # Clipping failed after all attempts

# Function to build a VRT and translate using gdal_translate
def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_str):
    """
    Generates a VRT from multiple TIFF files and converts it to an optimized and compressed TIFF using gdal_translate.

    Args:
    - name_out_vrt: Path to the output VRT file.
    - name_out_tif: Path to the optimized output TIFF file.
    - files_tif_str: String containing the paths to the TIFF files to process.
    """
    log_message(f"[INFO] Building VRT from TIFF files: {files_tif_str}")
    os.system(f'gdalbuildvrt {name_out_vrt} {files_tif_str}')
    log_message(f"[INFO] VRT created: {name_out_vrt}")

    log_message(f"[INFO] Translating VRT to optimized TIFF: {name_out_tif}")
    os.system(f'gdal_translate -a_nodata 0 -co TILED=YES -co compress=DEFLATE -co PREDICTOR=2 -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=YES {name_out_vrt} {name_out_tif}')
    log_message(f"[INFO] Optimized TIFF saved: {name_out_tif}")

# Function to clean directories before processing begins
def clean_directories(directories_to_clean):
    """
    Cleans specified directories by removing all contents and recreating the directory.

    Args:
    - directories_to_clean: List of directories to clean.
    """
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
            log_message(f"[INFO] Cleaned and recreated directory: {directory}")
        else:
            os.makedirs(directory)
            log_message(f"[INFO] Created directory: {directory}")

# Function to check or create a GEE collection and make it public
def check_or_create_collection(collection, ee_project):
    """
    Checks if a GEE collection exists, and if not, creates it and makes it public.

    Args:
    - collection: The GEE collection path.
    - ee_project: The Earth Engine project name.
    """
    log_message(f"[INFO] Checking if collection exists: {collection}")
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        log_message(f"[INFO] Creating new collection: {collection}")
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        create_status = os.system(create_command)
        
        if create_status == 0:
            log_message(f"[INFO] Collection created successfully: {collection}")
            
            # Make the collection public
            log_message(f"[INFO] Making the collection public: {collection}")
            set_acl_command = f'earthengine --project {ee_project} acl set public {collection}'
            acl_status = os.system(set_acl_command)
            
            if acl_status == 0:
                log_message(f"[INFO] Collection made public successfully: {collection}")
            else:
                log_message(f"[ERROR] Failed to make the collection public: {collection}")
        else:
            log_message(f"[ERROR] Failed to create the collection: {collection}")
    else:
        log_message(f"[INFO] Collection already exists: {collection}")

# Função para fazer upload de um arquivo para o GEE
# Função para executar comandos de forma segura
def run_command(command_list):
    try:
        result = subprocess.run(command_list, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode, result.stdout
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stderr

# Função para fazer upload de um arquivo para o GEE
def upload_to_gee(gcs_path, asset_id, satellite, region, year, version, ee_project):
    """
    Faz upload de um arquivo para o GEE, adicionando metadados relevantes, e verifica se o asset já existe.

    Parâmetros:
    - gcs_path: Caminho para o arquivo no Google Cloud Storage (GCS).
    - asset_id: ID do asset para o upload no GEE.
    - satellite: Nome do satélite utilizado.
    - region: Região de destino.
    - year: Ano dos dados.
    - version: Versão do dataset.
    - ee_project: Nome do projeto GEE.
    - source: Fonte dos dados.
    - collection_name: Nome da coleção de dados.
    """
    log_message(f"[INFO] Preparando para fazer upload do arquivo para o GEE: {gcs_path}")
    
    try:
        # Converter ano para timestamps
        timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
        timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    except ValueError as ve:
        log_message(f"[ERRO] Ano inválido fornecido: {year}. Detalhes: {ve}")
        return

    creation_date = datetime.now().strftime('%Y-%m-%d')

    # Etapa 1: Configurar o projeto no gcloud
    log_message(f"[INFO] Configurando o projeto GEE: {ee_project}")
    set_project_command = ['gcloud', 'config', 'set', 'project', ee_project]
    return_code, output = run_command(set_project_command)
    if return_code != 0:
        log_message(f"[ERRO] Falha ao configurar o projeto GEE: {output}")
        return

    # Etapa 2: Verificar se o asset já existe
    log_message(f"[INFO] Verificando se o asset existe: {asset_id}")
    check_asset_command = ['earthengine', 'asset', 'info', asset_id]
    return_code, output = run_command(check_asset_command)

    # Etapa 3: Se o asset existir, excluí-lo
    if return_code == 0:
        log_message(f"[INFO] O asset já existe: {asset_id}, excluindo antes do upload...")
        delete_asset_command = ['earthengine', 'asset', 'delete', asset_id, '--force']
        del_code, del_output = run_command(delete_asset_command)
        
        if del_code == 0:
            log_message(f"[INFO] Asset existente excluído com sucesso: {asset_id}")
        else:
            log_message(f"[ERRO] Falha ao excluir o asset existente: {asset_id}. Detalhes: {del_output}")
            return  # Interromper execução se não for possível excluir o asset existente
    elif 'Asset does not exist' not in output:
        log_message(f"[ERRO] Erro ao verificar o asset: {asset_id}. Detalhes: {output}")
        return

    # Etapa 4: Prosseguir com o upload
    log_message(f"[INFO] Fazendo upload da imagem para o GEE: {asset_id}")

    upload_command = [
        'earthengine', 'upload', 'image',
        '--asset_id', asset_id,
        '--pyramiding_policy', 'mode',
        '--property', f'satellite={satellite}',
        '--property', f'region={region}',
        '--property', f'year={year}',
        '--property', f'version={version}',
        '--property', f'source={source_name}',
        '--property', f'collection_name={collection_name}',
        '--property', 'type=annual_burned_area',
        '--property', f'time_start={timestamp_start}',
        '--property', f'time_end={timestamp_end}',
        '--property', f'create_date={creation_date}',
        gcs_path
    ]

    status, upload_output = run_command(upload_command)

    if status == 0:
        log_message(f"[INFO] Upload para o GEE bem-sucedido: {asset_id}")
    else:
        log_message(f"[ERRO] Falha no upload para o GEE: {asset_id}. Detalhes: {upload_output}")
# Function to remove temporary files
def remove_temporary_files(files_to_remove):
    """
    Removes temporary files from the system.

    Args:
    - files_to_remove: List of file paths to remove.
    """
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                log_message(f"[INFO] Temporary file removed: {file}")
            except Exception as e:
                log_message(f"[ERROR] Failed to remove file: {file}. Details: {str(e)}")

# O resto do código estilo TensorFlow 1.x
def create_model_graph(hyperparameters):
    """
    Cria e retorna um grafo computacional TensorFlow dinamicamente com base nos parâmetros do modelo.
    """
    graph = tf.Graph()

    with graph.as_default():
        # Define placeholders para dados de entrada e rótulos
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        # Normaliza os dados de entrada
        normalized = (x_input - hyperparameters['data_mean']) / hyperparameters['data_std']

        # Constrói as camadas da rede neural com os hiperparâmetros definidos
        hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu')

        # Camada final de saída
        logits = fully_connected_layer(hidden5, n_neurons=hyperparameters['NUM_CLASSES'])
        
        # Define a função de perda (para treinamento, embora não seja necessária na inferência)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )
        
        # Define o otimizador (para treinamento, embora não seja necessária na inferência)
        # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        # Define the optimizer: Adam with the specified learning rate
        optimizer = tf.train.AdamOptimizer(hyperparameters['lr']).minimize(cross_entropy)
        
        # Operação para obter a classe prevista
        outputs = tf.argmax(logits, 1, name='predicted_class')
        
        # Inicializa todas as variáveis
        init = tf.global_variables_initializer()
        # Definir o saver para salvar ou restaurar o estado do modelo
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

# Function to classify data using a TensorFlow model in blocks and handle memory manually
def classify(data_classify_vector, model_path, hyperparameters, block_size=40000000):
    """
    Classifies data in blocks using a TensorFlow model, and resets the session to free memory.

    Args:
    - data_classify_vector: The input data (pixels) to classify.
    - model_path: Path to the TensorFlow model to be restored.
    - hyperparameters: Hyperparameters to create the model graph.
    - block_size: Number of pixels to process per block (default is 4,000,000).
    
    Returns:
    - output_data_classify: Classified data.
    """
    log_message(f"[INFO] Starting classification with model at path: {model_path}")
    
    # Number of pixels in the input data
    num_pixels = data_classify_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size  # Calculate the number of blocks

    output_blocks = []  # List to hold the results of each block

    # Process data in blocks
    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)  # Ensure we don't exceed array length
        log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")

        # Get the current block of data to classify
        data_block = data_classify_vector[start_idx:end_idx]
        
        # Clear the graph before starting a new session for each block
        tf.compat.v1.reset_default_graph()

        # Create model graph using provided hyperparameters for each block
        graph, placeholders, saver = create_model_graph(hyperparameters)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

        # Start a new session and restore the model
        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)
            
            # Classify the current block of data
            output_block = sess.run(
                graph.get_tensor_by_name('predicted_class:0'),
                feed_dict={placeholders['x_input']: data_block}
            )
            
            # Append the classified block to the result list
            output_blocks.append(output_block)

            # No need to manually close the session as it's inside 'with', and will auto-close

    # Concatenate the classified blocks into a single array
    output_data_classify = np.concatenate(output_blocks, axis=0)
    log_message(f"[INFO] Classification completed")
    
    return output_data_classify

def process_single_image(dataset_classify, version, region,folder_temp):
    """
    Processes a single image by applying the classification model and spatial filtering to generate the final result.
    
    Args:
    - dataset_classify: GDAL dataset of the image to be classified.
    - num_classes: Number of classes in the model.
    - data_mean: Mean of the data for normalization.
    - data_std: Standard deviation of the data for normalization.
    - version: Version of the model.
    - region: Target region for classification.
    
    Returns:
    - Filtered classified image.
    """
    # Path to the remote model in Google Cloud Storage (with wildcards)
    gcs_model_file = f'gs://{bucket_name}/sudamerica/{country}/models_col1/col1_{country}_{version}_{region}_rnn_lstm_ckpt*'
    # Local path for the model files
    model_file_local_temp = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt'

    log_message(f"[INFO] Downloading TensorFlow model from GCS {gcs_model_file} to {folder_temp}.")
    
    # Command to download the model files from GCS
    try:
        subprocess.run(f'gsutil cp {gcs_model_file} {folder_temp}', shell=True, check=True)
        log_message(f"[INFO] Model downloaded successfully.")
    except subprocess.CalledProcessError as e:
        log_message(f"[ERROR] Failed to download model from GCS: {e}")
        return None

    # Path to the JSON file containing hyperparameters
    json_path = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt_hyperparameters.json'

    # Load hyperparameters from the JSON file
    with open(json_path, 'r') as json_file:
        hyperparameters = json.load(json_file)

    # Retrieve hyperparameter values from the JSON file
    DATA_MEAN = np.array(hyperparameters['data_mean'])
    DATA_STD = np.array(hyperparameters['data_std'])
    NUM_N_L1 = hyperparameters['NUM_N_L1']
    NUM_N_L2 = hyperparameters['NUM_N_L2']
    NUM_N_L3 = hyperparameters['NUM_N_L3']
    NUM_N_L4 = hyperparameters['NUM_N_L4']
    NUM_N_L5 = hyperparameters['NUM_N_L5']
    NUM_CLASSES = hyperparameters['NUM_CLASSES']
    NUM_INPUT = hyperparameters['NUM_INPUT']

    log_message(f"[INFO] Loaded hyperparameters: DATA_MEAN={DATA_MEAN}, DATA_STD={DATA_STD}, NUM_N_L1={NUM_N_L1}, NUM_N_L2={NUM_N_L2}, NUM_N_L3={NUM_N_L3}, NUM_N_L4={NUM_N_L4}, NUM_N_L5={NUM_N_L5}, NUM_CLASSES={NUM_CLASSES}")

    # Convert GDAL dataset to a NumPy array
    log_message(f"[INFO] Converting GDAL dataset to NumPy array.")
    data_classify = convert_to_array(dataset_classify)
    
    # Reshape into a single pixel vector
    log_message(f"[INFO] Reshaping data into a single pixel vector.")
    data_classify_vector = reshape_single_vector(data_classify)
    # print('data_classify_vector',data_classify_vector)
    # Normalize the input vector using data_mean and data_std
    # log_message(f"[INFO] Normalizing the input vector using data_mean and data_std.")
    # data_classify_vector = (data_classify_vector - DATA_MEAN) / DATA_STD

    # Perform the classification using the model
    log_message(f"[INFO] Running classification using the model.")
    output_data_classified = classify(data_classify_vector, model_file_local_temp, hyperparameters)
    
    # Reshape the classified data back into image format
    log_message(f"[INFO] Reshaping classified data back into image format.")
    output_image_data = reshape_image_output(output_data_classified, data_classify)
    
    # Apply spatial filtering
    log_message(f"[INFO] Applying spatial filtering and completing the processing of this scene.")
    return filter_spatial(output_image_data)

# Main function to process satellite and year data
def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_temp, suffix, ee_project, country, version, region):
    log_message(f"[INFO] Processing year by satellite for country: {country}, version: {version}, region: {region}")
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
    grid_landsat = grid.getInfo()['features']
    start_time = time.time()

    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION/burned_area_{country}_{version}'
    check_or_create_collection(collection_name, ee_project)

    for satellite_year in satellite_years:
        satellite = satellite_year['satellite']
        years = satellite_year['years']

        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                log_message(f"[INFO] Processing year {year} for satellite {satellite.upper()}")
                image_name = f"burned_area_{country}_{satellite}_{version}_region{region[1:]}_{year}{suffix}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'

                if not os.path.exists(local_cog_path):
                    log_message(f"[INFO] Downloading COG from GCS: {gcs_cog_path}")
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')

                input_scenes = []
                with tqdm(total=len(grid_landsat), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid in grid_landsat:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_temp}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'

                        if os.path.isfile(output_image_name):
                            log_message(f"[INFO] Scene {orbit}/{point} already processed. Skipping.")
                            pbar_scenes.update(1)
                            continue

                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_temp}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        log_message(f"[INFO] Clipping image: {local_cog_path}")
                        
                        # Tenta o recorte até ser bem-sucedido ou esgotar as tentativas
                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)

                        if clipping_success:
                            dataset_classify = load_image(NBR_clipped)
                            image_data = process_single_image(dataset_classify, version, region, folder_temp)

                            log_message(f"[INFO] Convert to raster")

                            convert_to_raster(dataset_classify, image_data, output_image_name)
                            input_scenes.append(output_image_name)
                        else:
                            log_message(f"[WARNING] Clipping failed for scene {orbit}/{point}. Proceeding to the next scene.")
                        
                        pbar_scenes.update(1)

                # Merging scenes and uploading to GCS and GEE
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_temp}/merged_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"
                    log_message(f"[INFO] Merging scenes for year {year}")
                    generate_optimized_image(merge_output_temp, output_image, input_scenes_str)
                    os.system(f'gsutil cp {output_image} {gcs_filename}')
                    log_message(f"[INFO] Uploading to GCS completed: {gcs_filename}")
                    upload_to_gee(gcs_filename, f'{collection_name}/{image_name}', satellite, region, year, version, ee_project)

                clean_directories([folder_temp])
                elapsed_time = time.time() - start_time
                log_message(f"[INFO] Year {year} processing completed. 🎉🎉🎉") 
                log_message(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                pbar_years.update(1)

def render_classify_models(models_to_classify):
    """
    Processes a list of models and mosaics to classify burned areas.
    Args:
    - models_to_classify: List of dictionaries containing models, mosaics, and a simulation flag.
    """
    log_message(f"[INFO] [render_classify_models] STARTING PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")
    # Define bucket name
    bucket_name = 'mapbiomas-fire'
    # Loop through each model
    for model_info in models_to_classify:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]
        log_message(f"[INFO] Processing model: {model_name}")
        log_message(f"[INFO] Selected mosaics: {mosaics}")
        log_message(f"[INFO] Simulation mode: {simulation}")
        # Extract model information
        parts = model_name.split('_')
        country = parts[1]
        version = parts[2]
        region = parts[3]
        # Define directories
        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_temp = f'{folder}/tmp1'
        folder_mosaic = f'{folder}/mosaics_cog'
        
        log_message(f"[INFO] Starting the classification process for country: {country}.")
        
        # Ensure necessary directories exist
        for directory in [folder_temp, folder_mosaic]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                # log_message(f"[INFO] Created directory: {directory}")
            else:
                log_message(f"[INFO] Directory already exists: {directory}")
        
        clean_directories([folder_temp, folder_mosaic])
        # Prepare satellite and year list based on mosaics
        satellite_years = []
        for mosaic in mosaics:
            mosaic_parts = mosaic.split('_')
            satellite = mosaic_parts[0]
            year = int(mosaic_parts[3])
            satellite_years.append({
                "satellite": satellite,
                "years": [year]
            })
        # If in simulation mode, just simulate the processing
        if simulation:
            log_message(f"[SIMULATION] Would process model: {model_name} with mosaics: {mosaics}")
        else:
            # Call the main processing function (this will process all years for the satellite)
            process_year_by_satellite(
                satellite_years=satellite_years,
                bucket_name=bucket_name,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=f'mapbiomas-{country}',
                country=country,
                version=version,
                region=region
            )
   
    log_message(f"[INFO] [render_classify_models] FINISH PROCESSINGS FOR CLASSIFY MODELS {models_to_classify}")
