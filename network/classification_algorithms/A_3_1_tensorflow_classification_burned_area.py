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

# TensorFlow compatibility mode for version 1.x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style

# Fixed Hyperparameters
NUM_INPUT = 30  # Example: number of input features (number of bands or pixels per input vector)
NUM_CLASSES = 5  # Example: number of classes for classification
NUM_N_L1 = 128  # Number of neurons in hidden layer 1
NUM_N_L2 = 64   # Number of neurons in hidden layer 2
NUM_N_L3 = 32   # Number of neurons in hidden layer 3
NUM_N_L4 = 16   # Number of neurons in hidden layer 4
NUM_N_L5 = 8    # Number of neurons in hidden layer 5
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer

# Normalization Parameters
DATA_MEAN = 0.0  # Data mean (for normalization)
DATA_STD = 1.0   # Data standard deviation (for normalization)

# Define directories for data and model output
folder = f'/content/mapbiomas-fire/sudamerica/{country}'  # Main directory where data is stored
folder_samples = f'{folder}/training_samples'  # Directory for sample data
folder_model = f'{folder}/models_col1'  # Directory for model output
folder_images = f'{folder}/tmp1'  # Temporary image directory
folder_mosaic = f'{folder}/mosaics_cog'  # Directory for COG (Cloud-Optimized GeoTIFF) files

# Ensure necessary directories exist
for directory in [folder_samples, folder_model, folder_images, folder_mosaic]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to reshape classified data into a single pixel vector
def reshape_single_vector(data_classify):
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

# Function to load an image using GDAL
def load_image(image_path):
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

# Function to convert a GDAL dataset to a NumPy array
def convert_to_array(dataset):
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return np.nan_to_num(stacked_data, nan=0)  # Replace NaN values with 0

# Function to classify data using a TensorFlow model
def classify(data_classify_vector, model_path, num_input, num_classes, data_mean, data_std):
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, model_path)
        output_data_classify = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),
            feed_dict={placeholders['x_input']: data_classify_vector}
        )
    return output_data_classify

# Function to classify data using a TensorFlow model
def classify(data_classify_vector, model_path, num_input, num_classes, data_mean, data_std):
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, model_path)
        output_data_classify = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),
            feed_dict={placeholders['x_input']: data_classify_vector}
        )
    return output_data_classify

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to apply spatial filtering on classified images
def filter_spatial(output_image_data):
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))  # Removes small white regions
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))  # Fills small black holes
    return close_image

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Release the output dataset from memory

# Function to convert meters into degrees based on latitude
def meters_to_degrees(meters, latitude):
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Function to expand geometry with a buffer in meters
def expand_geometry(geometry, buffer_distance_meters=50):
    geom = shape(geometry)
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Function to check if there is a significant intersection between the geometry and the image
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

# Main function to process satellite and year data
def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_images, suffix, ee_project, country, version, region):
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
                image_name = f"burned_area_{country}_{satellite}_v{version}_region{region[1:]}_{year}{suffix}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'

                if not os.path.exists(local_cog_path):
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')

                input_scenes = []
                with tqdm(total=len(grid_landsat), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid in grid_landsat:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_images}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'

                        if os.path.isfile(output_image_name):
                            pbar_scenes.update(1)
                            continue

                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_images}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        
                        clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)
                        dataset_classify = load_image(NBR_clipped)
                        image_data = process_single_image(dataset_classify, NUM_CLASSES, DATA_MEAN, DATA_STD, version, region, country)

                        convert_to_raster(dataset_classify, image_data, output_image_name)
                        input_scenes.append(output_image_name)
                        pbar_scenes.update(1)

                # Merging scenes and uploading to GCS and GEE
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_images}/merged_temp_{year}.tif"
                    output_image = f"{folder_images}/{image_name}.tif"
                    generate_optimized_image(merge_output_temp, output_image, input_scenes_str)
                    os.system(f'gsutil cp {output_image} {gcs_filename}')
                    upload_to_gee(gcs_filename, f'{collection_name}/{image_name}', satellite, region, year, version, ee_project)

                clean_directories([folder_images])
                elapsed_time = time.time() - start_time
                pbar_years.update(1)

# Function to create the TensorFlow model graph dynamically
def create_model_graph(num_input, num_classes, data_mean, data_std):
    graph = tf.Graph()
    
    with graph.as_default():
        x_input = tf.placeholder(tf.float32, shape=[None, num_input], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')
        normalized = (x_input - data_mean) / data_std

        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        logits = fully_connected_layer(hidden5, n_neurons=num_classes)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input), name='cross_entropy_loss')

        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)
        outputs = tf.argmax(logits, 1, name='predicted_class')
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    return graph, {'x_input': x_input, 'y_input': y_input}, saver

# Function to check or create a GEE collection
def check_or_create_collection(collection, ee_project):
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)

    if status != 0:
        os.system(f'earthengine --project {ee_project} create collection {collection}')

# Function to upload a file to GEE
def upload_to_gee(gcs_path, asset_id, satellite, region, year, version, ee_project):
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')

    check_asset_command = f'earthengine --project {ee_project} asset info {asset_id}'
    asset_status = os.system(check_asset_command)

    if asset_status == 0:
        log_message(f'[INFO] Asset already exists: {asset_id}')
    else:
        upload_command = (
            f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
            f'--pyramiding_policy=mode '
            f'--property satellite={satellite} '
            f'--property region={region} '
            f'--property year={year} '
            f'--property version={version} '
            f'{gcs_path}'
        )
        os.system(upload_command)
