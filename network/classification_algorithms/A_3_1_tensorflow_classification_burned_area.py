import os
import numpy as np
import tensorflow as tf
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
import datetime  # For handling timestamps and date operations

import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style

# Definição de hiperparâmetros fixos
NUM_INPUT = 30  # Exemplo: número de features de entrada (número de bandas ou pixels por vetor de entrada)
NUM_CLASSES = 5  # Exemplo: número de classes para classificação
NUM_N_L1 = 128  # Número de neurônios na camada oculta 1
NUM_N_L2 = 64   # Número de neurônios na camada oculta 2
NUM_N_L3 = 32   # Número de neurônios na camada oculta 3
NUM_N_L4 = 16   # Número de neurônios na camada oculta 4
NUM_N_L5 = 8    # Número de neurônios na camada oculta 5
LEARNING_RATE = 0.001  # Taxa de aprendizado para o otimizador Adam

# Parâmetros de normalização (você pode ajustá-los conforme o seu conjunto de dados)
DATA_MEAN = 0.0  # Média dos dados (para normalização)
DATA_STD = 1.0   # Desvio padrão dos dados (para normalização)


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



# ---------------------------
# Functions for Burned Area Prediction Using the Available Model
# ---------------------------

# Function to reshape classified data into a single pixel vector
def reshape_single_vector(data_classify):
    """
    Reshapes classified image data into a 1D vector of pixels.

    Args:
    - data_classify: 2D classified image array.

    Returns:
    - 1D vector containing pixels from the image.
    """
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

# Function to load an image using GDAL
def load_image(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array(dataset):
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return np.nan_to_num(stacked_data, nan=0)  # Substitui NaN por 0 # !perguntar para a Vera se tudo bem substituir valores mask, por NaN, no uso do convert_to_array do treinamento e no da classificação

import os
import numpy as np
import tensorflow as tf
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
import datetime  # For handling timestamps and date operations

import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style

# Definição de hiperparâmetros fixos
NUM_INPUT = 30  # Exemplo: número de features de entrada (número de bandas ou pixels por vetor de entrada)
NUM_CLASSES = 5  # Exemplo: número de classes para classificação
NUM_N_L1 = 128  # Número de neurônios na camada oculta 1
NUM_N_L2 = 64   # Número de neurônios na camada oculta 2
NUM_N_L3 = 32   # Número de neurônios na camada oculta 3
NUM_N_L4 = 16   # Número de neurônios na camada oculta 4
NUM_N_L5 = 8    # Número de neurônios na camada oculta 5
LEARNING_RATE = 0.001  # Taxa de aprendizado para o otimizador Adam

# Parâmetros de normalização (você pode ajustá-los conforme o seu conjunto de dados)
DATA_MEAN = 0.0  # Média dos dados (para normalização)
DATA_STD = 1.0   # Desvio padrão dos dados (para normalização)


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



# ---------------------------
# Functions for Burned Area Prediction Using the Available Model
# ---------------------------

# Function to reshape classified data into a single pixel vector
def reshape_single_vector(data_classify):
    """
    Reshapes classified image data into a 1D vector of pixels.

    Args:
    - data_classify: 2D classified image array.

    Returns:
    - 1D vector containing pixels from the image.
    """
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

# Function to load an image using GDAL
def load_image(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array(dataset):
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return np.nan_to_num(stacked_data, nan=0)  # Substitui NaN por 0 # !perguntar para a Vera se tudo bem substituir valores mask, por NaN, no uso do convert_to_array do treinamento e no da classificação

def classify(data_classify_vector, model_path, num_input, num_classes, data_mean, data_std):
    """
    Realiza a classificação usando o grafo TensorFlow restaurado.

    Args:
    - data_classify_vector: Vetor de entrada de dados (pixels).
    - model_path: Caminho para o modelo salvo.
    - num_input: Número de entradas (features).
    - num_classes: Número de classes no modelo.
    - data_mean: Média dos dados para normalização.
    - data_std: Desvio padrão dos dados para normalização.

    Returns:
    - output_data_classify: Dados classificados.
    """
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Restaurar o modelo treinado
        saver.restore(sess, model_path)

        # Classificar os dados
        output_data_classify = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),
            feed_dict={placeholders['x_input']: data_classify_vector}
        )

    return output_data_classify

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    """
    Reshapes classified data back into a 2D image format.

    Args:
    - output_data_classified: 1D classified data vector.
    - data_classify: Original image data.

    Returns:
    - 2D classified image.
    """
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to apply spatial filtering on classified images
def filter_spatial(output_image_data):
    """
    Applies spatial filtering to remove small regions in the classified image.

    Args:
    - output_image_data: Binary classified image.

    Returns:
    - Image with spatial filtering applied.
    """
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))  # Removes small white regions
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))  # Fills small black holes
    return close_image

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    """
    Converts a NumPy array into a GeoTIFF raster file with correct geospatial transformation.

    Args:
    - dataset_classify: GDAL dataset with original image metadata.
    - image_data_scene: Classified image data (NumPy array).
    - output_image_name: Name of the output file.
    """
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # Apply GeoTransform and projection from the original dataset
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Release the output dataset from memory

# Function to convert meters into degrees based on latitude
def meters_to_degrees(meters, latitude):
    """
    Converts a distance in meters to degrees, taking latitude into account.

    Args:
    - meters: Distance in meters.
    - latitude: Latitude in degrees.

    Returns:
    - Distance in degrees.
    """
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Function to expand geometry with a buffer in meters
def expand_geometry(geometry, buffer_distance_meters=50):
    """
    Expands the geometry by applying a buffer in meters.

    Args:
    - geometry: Original geometry.
    - buffer_distance_meters: Buffer distance in meters.

    Returns:
    - Expanded geometry.
    """
    geom = shape(geometry)
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Function to check if there is a significant intersection between the geometry and the image
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    """
    Checks if there is a significant intersection between the geometry and image bounds.

    Args:
    - geom: Geometry of the scene.
    - image_bounds: Image boundaries.
    - min_intersection_area: Minimum intersection area to consider (default: 0.01).

    Returns:
    - True if the intersection is significant, False otherwise.
    """
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

# Function to clip an image based on the provided geometry
def clip_image_by_grid(geom, image, output, buffer_distance_meters=100):
    """
    Clips an image based on a given geometry and saves the result.

    Args:
    - geom: Clipping geometry.
    - image: Path to the input image.
    - output: Path to the output image.
    - buffer_distance_meters: Buffer distance for the geometry.
    """
    with rasterio.open(image) as src:
        expanded_geom = expand_geometry(geom, buffer_distance_meters)

        try:
            # Check if there is sufficient intersection to clip the image
            if has_significant_intersection(expanded_geom, src.bounds):
                out_image, out_transform = mask(src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
            else:
                log_message(f'Skipping image: {image} - Insufficient overlap with raster.')
                return
        except ValueError as e:
            log_message(f'Skipping image: {image} - {str(e)}')
            return

    # Update metadata and save the output raster file
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)

# Function to build a VRT and translate using gdal_translate
def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_str):
    """
    Generates a VRT from multiple TIFF files and converts it to an optimized and compressed TIFF using gdal_translate.

    Args:
    - name_out_vrt: Path to the output VRT file.
    - name_out_tif: Path to the optimized output TIFF file.
    - files_tif_str: String containing the paths to the TIFF files to process.
    """
    # Step 1: Create the VRT from multiple scenes
    log_message(f'[INFO] Building VRT: {name_out_vrt}')
    os.system(f'gdalbuildvrt {name_out_vrt} {files_tif_str}')

    # Step 2: Translate the VRT into an optimized and compressed TIFF
    log_message(f'[INFO] Translating to optimized TIFF: {name_out_tif}')
    os.system(f'gdal_translate -a_nodata 0 -co TILED=YES -co compress=DEFLATE -co PREDICTOR=2 -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=YES {name_out_vrt} {name_out_tif}')

    log_message(f'[INFO] Translation complete. Optimized image saved in {name_out_tif}')


import shutil  # For file and directory operations
import datetime  # For handling date and time

# Function to clean directories before processing begins
def clean_directories(directories_to_clean):
    """
    Cleans specified directories by removing all contents and recreating the directory.

    Args:
    - directories_to_clean: List of directories to clean.
    """
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)  # Removes the directory and all its contents
            os.makedirs(directory)  # Recreates the directory empty
            log_message(f'[INFO] Directory cleaned: {directory}')
        else:
            os.makedirs(directory)
            log_message(f'[INFO] Directory created: {directory}')

# Function to check or create a collection in Google Earth Engine (GEE)
def check_or_create_collection(collection, ee_project):
    """
    Checks if a GEE collection exists, and if not, creates it.

    Args:
    - collection: The GEE collection path.
    - ee_project: The Earth Engine project name.
    """
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        log_message(f'[INFO] Creating new collection in GEE: {collection}')
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)
    else:
        log_message(f'[INFO] Collection already exists: {collection}')

# Function to upload a file to GEE with metadata and check if the asset already exists
def upload_to_gee(gcs_path, asset_id, satellite, region, year, version, ee_project):
    """
    Uploads a file to GEE, adding relevant metadata, and checks if the asset already exists.

    Args:
    - gcs_path: Path to the file in Google Cloud Storage (GCS).
    - asset_id: Asset ID for the GEE upload.
    - satellite: Name of the satellite used.
    - region: Target region.
    - year: Year of the data.
    - version: Version of the dataset.
    - ee_project: GEE project name.
    """
    # Define timestamps for start and end of the year
    timestamp_start = int(datetime.datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime.datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Check if the asset already exists in GEE
    check_asset_command = f'earthengine --project {ee_project} asset info {asset_id}'
    asset_status = os.system(check_asset_command)

    if asset_status == 0:
        log_message(f'[INFO] Asset already exists, skipping upload: {asset_id}')
    else:
        # Command to upload the image to GEE with metadata
        upload_command = (
            f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
            f'--pyramiding_policy=mode '
            f'--property satellite={satellite} '
            f'--property region={region} '
            f'--property year={year} '
            f'--property version={version} '
            f'--property source=IPAM '
            f'--property type=annual_burned_area '
            f'--property time_start={timestamp_start} '
            f'--property time_end={timestamp_end} '
            f'--property create_date={creation_date} '
            f'{gcs_path}'
        )

        log_message(f'[INFO] Starting upload to GEE: {asset_id}')
        status = os.system(upload_command)

        if status == 0:
            log_message(f'[INFO] Upload successful: {asset_id}')
        else:
            log_message(f'[ERROR] Upload failed: {asset_id}')
            log_message(f'[ERROR] Command status: {status}')

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
                log_message(f'[INFO] Temporary file removed: {file}')
            except Exception as e:
                prilog_messagent(f'[ERROR] Failed to remove file: {file}. Details: {str(e)}')

# Main processing function for satellite and year classification
def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_images, suffix, ee_project, country, version, region):
    """
    Processes satellite and year data to classify burned areas and upload the results to Google Earth Engine (GEE).

    Args:
    - satellite_years: List of dictionaries containing satellite names and their corresponding years.
    - bucket_name: Name of the Google Cloud Storage bucket.
    - folder_mosaic: Folder path for storing mosaic files.
    - folder_images: Folder path for storing temporary images.
    - suffix: Optional suffix for naming the files.
    - ee_project: Google Earth Engine project name.
    - country: Country name.
    - version: Dataset version.
    - region: Region code (e.g., 'r1').
    """
    # Load the Landsat grid from GEE for the region
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
    grid_landsat = grid.getInfo()['features']  # Load the Landsat grid
    start_time = time.time()

    # Define the GEE collection path
    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION/burned_area_{country}_{version}'
    check_or_create_collection(collection_name, ee_project)  # Check or create the GEE collection

    # Main loop to process satellites and years
    for satellite_year in satellite_years:
        satellite = satellite_year['satellite']
        log_message(f"\n{'='*60}")
        log_message(f'[INFO] Starting processing for satellite: {satellite.upper()}')
        log_message(f"{'='*60}")

        years = satellite_year['years']

        # Progress bar for processing years
        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                log_message(f"\n{'-'*60}")
                log_message(f'[INFO] Processing year {year} for satellite {satellite.upper()}...')
                log_message(f"{'-'*60}")

                # File naming convention for the classified image
                image_name = f"burned_area_{country}_{satellite}_v{version}_region{region[1:]}_{year}{suffix}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                # Paths for local and cloud-optimized images
                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'

                # Copy COG file from GCS if not already copied locally
                if os.path.exists(local_cog_path):
                    log_message(f'[INFO] COG file already copied locally: {local_cog_path}')
                else:
                    log_message(f'[INFO] Copying COG file from GCS to local directory...')
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')

                input_scenes = []
                total_scenes_done = 0

                # Progress bar for processing scenes within a year
                with tqdm(total=len(grid_landsat), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid in grid_landsat:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_images}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'

                        # Skip if the file already exists
                        if os.path.isfile(output_image_name):
                            log_message(f'[INFO] File already exists: {output_image_name}. Skipping...')
                            pbar_scenes.update(1)
                            continue

                        # Clip the image based on the scene geometry
                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_images}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        log_message(f'[INFO] Image clipped: {NBR_clipped}')

                        try:
                            clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)

                            dataset_classify = load_image(NBR_clipped)

                            image_data = process_single_image(dataset_classify, NUM_CLASSES, DATA_MEAN, DATA_STD, version, region)

                            convert_to_raster(dataset_classify, image_data, output_image_name)
                            input_scenes.append(output_image_name)
                            total_scenes_done += 1

                            log_message(f'[PROGRESS] {total_scenes_done}/{len(grid_landsat)} scenes processed.')
                            pbar_scenes.update(1)
                        except Exception as e:
                            log_message(f'[ERROR] Failed to process scene {orbit}/{point}. Details: {str(e)}')
                            pbar_scenes.update(1)
                            continue

                # Merge scenes if there are processed inputs
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_images}/merged_temp_{year}.tif"
                    output_image = f"{folder_images}/{image_name}.tif"

                    try:
                        generate_optimized_image(merge_output_temp, output_image, input_scenes_str)
                        log_message(f'[INFO] Merging {len(input_scenes)} scenes completed for year {year}.')

                        # Upload to GCS
                        upload_status = os.system(f'gsutil cp {output_image} {gcs_filename}')
                        if upload_status == 0:
                            log_message(f'[INFO] Upload to GCS successful: {gcs_filename}')
                        else:
                            log_message(f'[ERROR] Upload failed for file: {output_image}')
                            continue

                        # Upload to GEE within the collection
                        output_asset_id = f'{collection_name}/{image_name}'
                        log_message(f'[INFO] Uploading to GEE: {output_asset_id}')
                        upload_to_gee(gcs_filename, output_asset_id, satellite, region, year, version, ee_project)

                    except Exception as e:
                        log_message(f'[ERROR] Failed to merge scenes for year {year}. Details: {str(e)}')
                        continue

                # Clean up temporary files after processing each year
                temporary_files = [local_cog_path, merge_output_temp] + input_scenes
                remove_temporary_files(temporary_files)

                # Clean the folder after each year's processing
                clean_directories([folder_images])

                elapsed_time = time.time() - start_time
                log_message(f'[INFO] Total time spent so far for year {year}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

                # Update progress bar for year processing
                pbar_years.update(1)

        log_message(f"\n{'='*60}")
        log_message(f'[INFO] Processing for satellite {satellite.upper()} completed.')
        log_message(f"{'='*60}")

    log_message('[INFO] Full processing completed.')

# Função para processar uma única imagem e aplicar o modelo de classificação
def process_single_image(dataset_classify, num_classes, data_mean, data_std, version, region):
    """
    Processa uma imagem classificada, aplicando o modelo e filtragem espacial para gerar o resultado final.

    Args:
    - dataset_classify: Dataset GDAL da imagem a ser classificada.
    - num_classes: Número de classes no modelo.
    - data_mean: Média dos dados (para normalização).
    - data_std: Desvio padrão dos dados (para normalização).
    - version: Versão do modelo.
    - region: Região alvo para classificação.

    Returns:
    - Imagem classificada filtrada.
    """
    # Converte o dataset GDAL em um array NumPy
    data_classify = convert_to_array(dataset_classify)

    # Reshape para vetor único de pixels
    data_classify_vector = reshape_single_vector(data_classify)

    # Normaliza o vetor de entrada usando data_mean e data_std
    data_classify_vector = (data_classify_vector - data_mean) / data_std

    # Realiza a classificação usando o modelo
    model_path = f'{folder_model}/col1_{country}_v{version}_{region}_rnn_lstm_ckpt'
    output_data_classified = classify(data_classify_vector, model_path, NUM_INPUT, NUM_CLASSES, DATA_MEAN, DATA_STD)


    # Reshape para o formato de imagem
    output_image_data = reshape_image_output(output_data_classified, data_classify)

    # Aplica filtro espacial
    return filter_spatial(output_image_data)


def create_model_graph(num_input, num_classes, data_mean, data_std):
    """
    Cria e retorna um grafo computacional TensorFlow dinamicamente com base nos parâmetros do modelo.
    """
    graph = tf.Graph()
    
    with graph.as_default():
        # Define placeholders para dados de entrada e rótulos
        x_input = tf.placeholder(tf.float32, shape=[None, num_input], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        # Normaliza os dados de entrada
        normalized = (x_input - data_mean) / data_std

        # Constrói as camadas da rede neural com os hiperparâmetros definidos
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        # Camada final de saída
        logits = fully_connected_layer(hidden5, n_neurons=num_classes)
        
        # Define a função de perda (para treinamento, embora não seja necessária na inferência)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )
        
        # Define o otimizador (para treinamento, embora não seja necessária na inferência)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        
        # Operação para obter a classe prevista
        outputs = tf.argmax(logits, 1, name='predicted_class')
        
        # Inicializa todas as variáveis
        init = tf.global_variables_initializer()

        # Definir o saver para salvar ou restaurar o estado do modelo
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

def classify_with_model(model_path, data_classify_vector, num_input, num_classes, data_mean, data_std):
    """
    Classify input data using a TensorFlow model restored from the specified path.

    Args:
    - model_path: Path to the saved TensorFlow model checkpoint.
    - data_classify_vector: Input data to be classified (1D vector of pixels).
    - num_input: Number of input features.
    - num_classes: Number of output classes for classification.
    - data_mean: Mean used for normalization.
    - data_std: Standard deviation used for normalization.

    Returns:
    - output_data_classified: Classification results.
    """
    # Create the graph dynamically for the model
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    
    with tf.Session(graph=graph) as sess:
        # Restore the model from the checkpoint
        saver.restore(sess, model_path)

        # Classify the input data
        output_data_classified = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),  # Get the output tensor
            feed_dict={placeholders['x_input']: data_classify_vector}
        )

    return output_data_classified

def render_classify_models(models_to_classify):
    """
    Processes a list of models and mosaics to classify burned areas.

    Args:
    - models_to_classify: List of dictionaries containing models, mosaics, and a simulation flag.
    """
    # Define bucket name
    bucket_name = 'mapbiomas-fire'

    # Loop through each model
    for model_info in models_to_classify:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]

        log_message(f"Processing model: {model_name}")
        log_message(f"Selected mosaics: {mosaics}")
        log_message(f"Simulation mode: {simulation}")

        # Extract model information
        parts = model_name.split('_')
        country = parts[1]
        version = parts[2]
        region = parts[3]

        # Define directories
        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_model = f'{folder}/models_col1'
        folder_images = f'{folder}/tmp1'
        folder_mosaic = f'{folder}/mosaics_cog'

        # Ensure directories exist
        if not os.path.exists(folder_model):
            os.makedirs(folder_model)

        # Clean directories for images and mosaics
        clean_directories([folder_images, folder_mosaic])

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
                folder_images=folder_images,
                suffix='',
                ee_project=f'mapbiomas-{country}',
                country=country,
                version=version,
                region=region
            )

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    """
    Reshapes classified data back into a 2D image format.

    Args:
    - output_data_classified: 1D classified data vector.
    - data_classify: Original image data.

    Returns:
    - 2D classified image.
    """
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to apply spatial filtering on classified images
def filter_spatial(output_image_data):
    """
    Applies spatial filtering to remove small regions in the classified image.

    Args:
    - output_image_data: Binary classified image.

    Returns:
    - Image with spatial filtering applied.
    """
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))  # Removes small white regions
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))  # Fills small black holes
    return close_image

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    """
    Converts a NumPy array into a GeoTIFF raster file with correct geospatial transformation.

    Args:
    - dataset_classify: GDAL dataset with original image metadata.
    - image_data_scene: Classified image data (NumPy array).
    - output_image_name: Name of the output file.
    """
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # Apply GeoTransform and projection from the original dataset
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Release the output dataset from memory

# Function to convert meters into degrees based on latitude
def meters_to_degrees(meters, latitude):
    """
    Converts a distance in meters to degrees, taking latitude into account.

    Args:
    - meters: Distance in meters.
    - latitude: Latitude in degrees.

    Returns:
    - Distance in degrees.
    """
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Function to expand geometry with a buffer in meters
def expand_geometry(geometry, buffer_distance_meters=50):
    """
    Expands the geometry by applying a buffer in meters.

    Args:
    - geometry: Original geometry.
    - buffer_distance_meters: Buffer distance in meters.

    Returns:
    - Expanded geometry.
    """
    geom = shape(geometry)
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Function to check if there is a significant intersection between the geometry and the image
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    """
    Checks if there is a significant intersection between the geometry and image bounds.

    Args:
    - geom: Geometry of the scene.
    - image_bounds: Image boundaries.
    - min_intersection_area: Minimum intersection area to consider (default: 0.01).

    Returns:
    - True if the intersection is significant, False otherwise.
    """
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

# Function to clip an image based on the provided geometry
def clip_image_by_grid(geom, image, output, buffer_distance_meters=100):
    """
    Clips an image based on a given geometry and saves the result.

    Args:
    - geom: Clipping geometry.
    - image: Path to the input image.
    - output: Path to the output image.
    - buffer_distance_meters: Buffer distance for the geometry.
    """
    with rasterio.open(image) as src:
        expanded_geom = expand_geometry(geom, buffer_distance_meters)

        try:
            # Check if there is sufficient intersection to clip the image
            if has_significant_intersection(expanded_geom, src.bounds):
                out_image, out_transform = mask(src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
            else:
                log_message(f'Skipping image: {image} - Insufficient overlap with raster.')
                return
        except ValueError as e:
            log_message(f'Skipping image: {image} - {str(e)}')
            return

    # Update metadata and save the output raster file
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)

# Function to build a VRT and translate using gdal_translate
def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_str):
    """
    Generates a VRT from multiple TIFF files and converts it to an optimized and compressed TIFF using gdal_translate.

    Args:
    - name_out_vrt: Path to the output VRT file.
    - name_out_tif: Path to the optimized output TIFF file.
    - files_tif_str: String containing the paths to the TIFF files to process.
    """
    # Step 1: Create the VRT from multiple scenes
    log_message(f'[INFO] Building VRT: {name_out_vrt}')
    os.system(f'gdalbuildvrt {name_out_vrt} {files_tif_str}')

    # Step 2: Translate the VRT into an optimized and compressed TIFF
    log_message(f'[INFO] Translating to optimized TIFF: {name_out_tif}')
    os.system(f'gdal_translate -a_nodata 0 -co TILED=YES -co compress=DEFLATE -co PREDICTOR=2 -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=YES {name_out_vrt} {name_out_tif}')

    log_message(f'[INFO] Translation complete. Optimized image saved in {name_out_tif}')


import shutil  # For file and directory operations
import datetime  # For handling date and time

# Function to clean directories before processing begins
def clean_directories(directories_to_clean):
    """
    Cleans specified directories by removing all contents and recreating the directory.

    Args:
    - directories_to_clean: List of directories to clean.
    """
    for directory in directories_to_clean:
        if os.path.exists(directory):
            shutil.rmtree(directory)  # Removes the directory and all its contents
            os.makedirs(directory)  # Recreates the directory empty
            log_message(f'[INFO] Directory cleaned: {directory}')
        else:
            os.makedirs(directory)
            log_message(f'[INFO] Directory created: {directory}')

# Function to check or create a collection in Google Earth Engine (GEE)
def check_or_create_collection(collection, ee_project):
    """
    Checks if a GEE collection exists, and if not, creates it.

    Args:
    - collection: The GEE collection path.
    - ee_project: The Earth Engine project name.
    """
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        log_message(f'[INFO] Creating new collection in GEE: {collection}')
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)
    else:
        log_message(f'[INFO] Collection already exists: {collection}')

# Function to upload a file to GEE with metadata and check if the asset already exists
def upload_to_gee(gcs_path, asset_id, satellite, region, year, version, ee_project):
    """
    Uploads a file to GEE, adding relevant metadata, and checks if the asset already exists.

    Args:
    - gcs_path: Path to the file in Google Cloud Storage (GCS).
    - asset_id: Asset ID for the GEE upload.
    - satellite: Name of the satellite used.
    - region: Target region.
    - year: Year of the data.
    - version: Version of the dataset.
    - ee_project: GEE project name.
    """
    # Define timestamps for start and end of the year
    timestamp_start = int(datetime.datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime.datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Check if the asset already exists in GEE
    check_asset_command = f'earthengine --project {ee_project} asset info {asset_id}'
    asset_status = os.system(check_asset_command)

    if asset_status == 0:
        log_message(f'[INFO] Asset already exists, skipping upload: {asset_id}')
    else:
        # Command to upload the image to GEE with metadata
        upload_command = (
            f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
            f'--pyramiding_policy=mode '
            f'--property satellite={satellite} '
            f'--property region={region} '
            f'--property year={year} '
            f'--property version={version} '
            f'--property source=IPAM '
            f'--property type=annual_burned_area '
            f'--property time_start={timestamp_start} '
            f'--property time_end={timestamp_end} '
            f'--property create_date={creation_date} '
            f'{gcs_path}'
        )

        log_message(f'[INFO] Starting upload to GEE: {asset_id}')
        status = os.system(upload_command)

        if status == 0:
            log_message(f'[INFO] Upload successful: {asset_id}')
        else:
            log_message(f'[ERROR] Upload failed: {asset_id}')
            log_message(f'[ERROR] Command status: {status}')

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
                log_message(f'[INFO] Temporary file removed: {file}')
            except Exception as e:
                prilog_messagent(f'[ERROR] Failed to remove file: {file}. Details: {str(e)}')

# Main processing function for satellite and year classification
def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_images, suffix, ee_project, country, version, region):
    """
    Processes satellite and year data to classify burned areas and upload the results to Google Earth Engine (GEE).

    Args:
    - satellite_years: List of dictionaries containing satellite names and their corresponding years.
    - bucket_name: Name of the Google Cloud Storage bucket.
    - folder_mosaic: Folder path for storing mosaic files.
    - folder_images: Folder path for storing temporary images.
    - suffix: Optional suffix for naming the files.
    - ee_project: Google Earth Engine project name.
    - country: Country name.
    - version: Dataset version.
    - region: Region code (e.g., 'r1').
    """
    # Load the Landsat grid from GEE for the region
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
    grid_landsat = grid.getInfo()['features']  # Load the Landsat grid
    start_time = time.time()

    # Define the GEE collection path
    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION/burned_area_{country}_{version}'
    check_or_create_collection(collection_name, ee_project)  # Check or create the GEE collection

    # Main loop to process satellites and years
    for satellite_year in satellite_years:
        satellite = satellite_year['satellite']
        log_message(f"\n{'='*60}")
        log_message(f'[INFO] Starting processing for satellite: {satellite.upper()}')
        log_message(f"{'='*60}")

        years = satellite_year['years']

        # Progress bar for processing years
        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                log_message(f"\n{'-'*60}")
                log_message(f'[INFO] Processing year {year} for satellite {satellite.upper()}...')
                log_message(f"{'-'*60}")

                # File naming convention for the classified image
                image_name = f"burned_area_{country}_{satellite}_v{version}_region{region[1:]}_{year}{suffix}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                # Paths for local and cloud-optimized images
                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'

                # Copy COG file from GCS if not already copied locally
                if os.path.exists(local_cog_path):
                    log_message(f'[INFO] COG file already copied locally: {local_cog_path}')
                else:
                    log_message(f'[INFO] Copying COG file from GCS to local directory...')
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')

                input_scenes = []
                total_scenes_done = 0

                # Progress bar for processing scenes within a year
                with tqdm(total=len(grid_landsat), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid in grid_landsat:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_images}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'

                        # Skip if the file already exists
                        if os.path.isfile(output_image_name):
                            log_message(f'[INFO] File already exists: {output_image_name}. Skipping...')
                            pbar_scenes.update(1)
                            continue

                        # Clip the image based on the scene geometry
                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_images}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        log_message(f'[INFO] Image clipped: {NBR_clipped}')

                        try:
                            clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)

                            dataset_classify = load_image(NBR_clipped)

                            image_data = process_single_image(dataset_classify, NUM_CLASSES, DATA_MEAN, DATA_STD, version, region)

                            convert_to_raster(dataset_classify, image_data, output_image_name)
                            input_scenes.append(output_image_name)
                            total_scenes_done += 1

                            log_message(f'[PROGRESS] {total_scenes_done}/{len(grid_landsat)} scenes processed.')
                            pbar_scenes.update(1)
                        except Exception as e:
                            log_message(f'[ERROR] Failed to process scene {orbit}/{point}. Details: {str(e)}')
                            pbar_scenes.update(1)
                            continue

                # Merge scenes if there are processed inputs
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_images}/merged_temp_{year}.tif"
                    output_image = f"{folder_images}/{image_name}.tif"

                    try:
                        generate_optimized_image(merge_output_temp, output_image, input_scenes_str)
                        log_message(f'[INFO] Merging {len(input_scenes)} scenes completed for year {year}.')

                        # Upload to GCS
                        upload_status = os.system(f'gsutil cp {output_image} {gcs_filename}')
                        if upload_status == 0:
                            log_message(f'[INFO] Upload to GCS successful: {gcs_filename}')
                        else:
                            log_message(f'[ERROR] Upload failed for file: {output_image}')
                            continue

                        # Upload to GEE within the collection
                        output_asset_id = f'{collection_name}/{image_name}'
                        log_message(f'[INFO] Uploading to GEE: {output_asset_id}')
                        upload_to_gee(gcs_filename, output_asset_id, satellite, region, year, version, ee_project)

                    except Exception as e:
                        log_message(f'[ERROR] Failed to merge scenes for year {year}. Details: {str(e)}')
                        continue

                # Clean up temporary files after processing each year
                temporary_files = [local_cog_path, merge_output_temp] + input_scenes
                remove_temporary_files(temporary_files)

                # Clean the folder after each year's processing
                clean_directories([folder_images])

                elapsed_time = time.time() - start_time
                log_message(f'[INFO] Total time spent so far for year {year}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

                # Update progress bar for year processing
                pbar_years.update(1)

        log_message(f"\n{'='*60}")
        log_message(f'[INFO] Processing for satellite {satellite.upper()} completed.')
        log_message(f"{'='*60}")

    log_message('[INFO] Full processing completed.')

# Função para processar uma única imagem e aplicar o modelo de classificação
def process_single_image(dataset_classify, num_classes, data_mean, data_std, version, region, country):
    """
    Processa uma imagem classificada, aplicando o modelo e filtragem espacial para gerar o resultado final.

    Args:
    - dataset_classify: Dataset GDAL da imagem a ser classificada.
    - num_classes: Número de classes no modelo.
    - data_mean: Média dos dados (para normalização).
    - data_std: Desvio padrão dos dados (para normalização).
    - version: Versão do modelo.
    - region: Região alvo para classificação.
    - country: País ao qual o modelo está relacionado.

    Returns:
    - Imagem classificada filtrada.
    """
    # Converte o dataset GDAL em um array NumPy
    data_classify = convert_to_array(dataset_classify)

    # Reshape para vetor único de pixels
    data_classify_vector = reshape_single_vector(data_classify)

    # Normaliza o vetor de entrada usando data_mean e data_std
    data_classify_vector = (data_classify_vector - data_mean) / data_std

    # --- Definir o diretório temporário baseado em país, região e versão ---
    folder = f'/content/mapbiomas-fire/sudamerica/{country}'
    temp_model_dir = f'{folder}/temp_model_dir'

    # Caminho do modelo no GCS baseado em país, versão e região
    gcs_model_path = f'gs://mapbiomas-fire/sudamerica/{country}/models_col1/col1_{country}_{version}_{region}_rnn_lstm_ckpt'

    # Caminho local onde os arquivos de modelo serão armazenados temporariamente
    local_model_path = f'{temp_model_dir}/col1_{country}_{version}_{region}_rnn_lstm_ckpt'

    # Se o diretório temporário não existir, criar
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)

    # Verificar se os arquivos do modelo já foram baixados. Se não, baixar do GCS.
    if not os.path.exists(local_model_path + '.index'):  # Verifica um dos arquivos do modelo, ex: .index
        log_message(f'[INFO] Baixando o modelo do GCS: {gcs_model_path}')
        os.system(f'gsutil cp {gcs_model_path}* {temp_model_dir}/')

    # Atualizar o caminho do modelo para o local temporário
    model_path = local_model_path
    # ---------------------------------------------------------------------------

    # Realiza a classificação usando o modelo
    output_data_classified = classify(data_classify_vector, model_path, NUM_INPUT, NUM_CLASSES, DATA_MEAN, DATA_STD)

    # Reshape para o formato de imagem
    output_image_data = reshape_image_output(output_data_classified, data_classify)

    # Aplica filtro espacial
    return filter_spatial(output_image_data)


def create_model_graph(num_input, num_classes, data_mean, data_std):
    """
    Cria e retorna um grafo computacional TensorFlow dinamicamente com base nos parâmetros do modelo.
    """
    graph = tf.Graph()
    
    with graph.as_default():
        # Define placeholders para dados de entrada e rótulos
        x_input = tf.placeholder(tf.float32, shape=[None, num_input], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        # Normaliza os dados de entrada
        normalized = (x_input - data_mean) / data_std

        # Constrói as camadas da rede neural com os hiperparâmetros definidos
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        # Camada final de saída
        logits = fully_connected_layer(hidden5, n_neurons=num_classes)
        
        # Define a função de perda (para treinamento, embora não seja necessária na inferência)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )
        
        # Define o otimizador (para treinamento, embora não seja necessária na inferência)
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        
        # Operação para obter a classe prevista
        outputs = tf.argmax(logits, 1, name='predicted_class')
        
        # Inicializa todas as variáveis
        init = tf.global_variables_initializer()

        # Definir o saver para salvar ou restaurar o estado do modelo
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

def classify_with_model(model_path, data_classify_vector, num_input, num_classes, data_mean, data_std):
    """
    Classify input data using a TensorFlow model restored from the specified path.

    Args:
    - model_path: Path to the saved TensorFlow model checkpoint.
    - data_classify_vector: Input data to be classified (1D vector of pixels).
    - num_input: Number of input features.
    - num_classes: Number of output classes for classification.
    - data_mean: Mean used for normalization.
    - data_std: Standard deviation used for normalization.

    Returns:
    - output_data_classified: Classification results.
    """
    # Create the graph dynamically for the model
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    
    with tf.Session(graph=graph) as sess:
        # Restore the model from the checkpoint
        saver.restore(sess, model_path)

        # Classify the input data
        output_data_classified = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),  # Get the output tensor
            feed_dict={placeholders['x_input']: data_classify_vector}
        )

    return output_data_classified

def render_classify_models(models_to_classify):
    """
    Processes a list of models and mosaics to classify burned areas.

    Args:
    - models_to_classify: List of dictionaries containing models, mosaics, and a simulation flag.
    """
    # Define bucket name
    bucket_name = 'mapbiomas-fire'

    # Loop through each model
    for model_info in models_to_classify:
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]

        log_message(f"Processing model: {model_name}")
        log_message(f"Selected mosaics: {mosaics}")
        log_message(f"Simulation mode: {simulation}")

        # Extract model information
        parts = model_name.split('_')
        country = parts[1]
        version = parts[2]
        region = parts[3]

        # Define directories
        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_model = f'{folder}/models_col1'
        folder_images = f'{folder}/tmp1'
        folder_mosaic = f'{folder}/mosaics_cog'

        # Ensure directories exist
        if not os.path.exists(folder_model):
            os.makedirs(folder_model)

        # Clean directories for images and mosaics
        clean_directories([folder_images, folder_mosaic])

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
                folder_images=folder_images,
                suffix='',
                ee_project=f'mapbiomas-{country}',
                country=country,
                version=version,
                region=region
            )
