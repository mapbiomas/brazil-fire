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
NUM_INPUT = 30
NUM_CLASSES = 5
NUM_N_L1 = 128
NUM_N_L2 = 64
NUM_N_L3 = 32
NUM_N_L4 = 16
NUM_N_L5 = 8
LEARNING_RATE = 0.001

# Normalization Parameters
DATA_MEAN = 0.0
DATA_STD = 1.0

# Define directories for data and model output
folder = f'/content/mapbiomas-fire/sudamerica/{country}'  
folder_samples = f'{folder}/training_samples'
folder_model = f'{folder}/models_col1'
folder_images = f'{folder}/tmp1'
folder_mosaic = f'{folder}/mosaics_cog'

log_message(f"[INFO] Starting the classification process for country: {country}, version: {version}, region: {region}.")

# Ensure necessary directories exist
for directory in [folder_samples, folder_model, folder_images, folder_mosaic]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        log_message(f"[INFO] Created directory: {directory}")
    else:
        log_message(f"[INFO] Directory already exists: {directory}")

# Function to reshape classified data into a single pixel vector
def reshape_single_vector(data_classify):
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

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
    return np.nan_to_num(stacked_data, nan=0)

# Function to classify data using a TensorFlow model
def classify(data_classify_vector, model_path, num_input, num_classes, data_mean, data_std):
    log_message(f"[INFO] Starting classification with model at path: {model_path}")
    graph, placeholders, saver = create_model_graph(num_input, num_classes, data_mean, data_std)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, model_path)
        output_data_classify = sess.run(
            graph.get_tensor_by_name('predicted_class:0'),
            feed_dict={placeholders['x_input']: data_classify_vector}
        )
    log_message(f"[INFO] Classification completed")
    return output_data_classify

# Function to reshape classified data back into image format
def reshape_image_output(output_data_classified, data_classify):
    log_message(f"[INFO] Reshaping classified data back to image format")
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Function to apply spatial filtering on classified images
def filter_spatial(output_image_data):
    log_message(f"[INFO] Applying spatial filtering on classified image")
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))
    return close_image

# Function to convert a NumPy array back into a GeoTIFF raster
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    log_message(f"[INFO] Converting array to GeoTIFF raster: {output_image_name}")
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)
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

# Main function to process satellite and year data
def process_year_by_satellite(satellite_years, bucket_name, folder_mosaic, folder_images, suffix, ee_project, country, version, region):
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
                image_name = f"burned_area_{country}_{satellite}_v{version}_region{region[1:]}_{year}{suffix}"
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
                        output_image_name = f'{folder_images}/image_col3_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'

                        if os.path.isfile(output_image_name):
                            log_message(f"[INFO] Scene {orbit}/{point} already processed. Skipping.")
                            pbar_scenes.update(1)
                            continue

                        geometry_scene = grid['geometry']
                        NBR_clipped = f'{folder_images}/image_mosaic_col3_{country}_{region}_{version}_{orbit}_{point}_clipped_{year}.tif'
                        log_message(f"[INFO] Clipping image: {local_cog_path}")
                        
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
                    log_message(f"[INFO] Merging scenes for year {year}")
                    generate_optimized_image(merge_output_temp, output_image, input_scenes_str)
                    os.system(f'gsutil cp {output_image} {gcs_filename}')
                    log_message(f"[INFO] Uploading to GCS completed: {gcs_filename}")
                    upload_to_gee(gcs_filename, f'{collection_name}/{image_name}', satellite, region, year, version, ee_project)

                clean_directories([folder_images])
                elapsed_time = time.time() - start_time
                log_message(f"[INFO] Year {year} processing completed. Total time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
                pbar_years.update(1)

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
    log_message(f"[INFO] Clipping image: {image} with buffer: {buffer_distance_meters} meters")
    with rasterio.open(image) as src:
        expanded_geom = expand_geometry(geom, buffer_distance_meters)
        try:
            if has_significant_intersection(expanded_geom, src.bounds):
                out_image, out_transform = mask(src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
                log_message(f"[INFO] Image clipped successfully: {output}")
            else:
                log_message(f"[INFO] Insufficient overlap for clipping: {image}")
                return
        except ValueError as e:
            log_message(f"[ERROR] Error during clipping: {str(e)}")
            return

    # Save the clipped image
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)
    log_message(f"[INFO] Clipped image saved: {output}")

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

# Function to check or create a GEE collection
def check_or_create_collection(collection, ee_project):
    """
    Checks if a GEE collection exists, and if not, creates it.

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
        os.system(create_command)
    else:
        log_message(f"[INFO] Collection already exists: {collection}")

# Function to upload a file to GEE
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
    log_message(f"[INFO] Preparing to upload file to GEE: {gcs_path}")
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')

    check_asset_command = f'earthengine --project {ee_project} asset info {asset_id}'
    asset_status = os.system(check_asset_command)

    if asset_status == 0:
        log_message(f"[INFO] Asset already exists: {asset_id}")
    else:
        log_message(f"[INFO] Uploading image to GEE: {asset_id}")
        upload_command = (
            f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
            f'--pyramiding_policy=mode '
            f'--property satellite={satellite} '
            f'--property region={region} '
            f'--property year={year} '
            f'--property version={version} '
            f'{gcs_path}'
        )
        status = os.system(upload_command)

        if status == 0:
            log_message(f"[INFO] Upload to GEE successful: {asset_id}")
        else:
            log_message(f"[ERROR] Upload to GEE failed: {asset_id}")

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

