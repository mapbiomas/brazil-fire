#!/usr/bin/env python
# coding: utf-8

"""
MapBiomas Fire – Burned Area Classification Pipeline
---------------------------------------------------

This script performs burned area classification using a trained
neural network model and Landsat NBR mosaics.

Workflow
--------

1. Load Landsat mosaic (VRT)
2. Clip mosaics by Landsat grid scenes
3. Convert raster data into pixel vectors
4. Apply trained neural network classifier
5. Apply spatial filtering to reduce noise
6. Export classified raster scenes
7. Merge all scenes into a yearly mosaic
8. Upload results to Google Cloud Storage
9. Import final results to Google Earth Engine

Inputs
------

- Landsat mosaic VRT files
- Pre-trained TensorFlow model
- Landsat grid geometry from Earth Engine

Outputs
-------

- Scene-level classified rasters
- Annual merged burned-area maps

Notes
-----

This script assumes that:
- TensorFlow graph and model variables are already defined
- Helper functions such as `convert_to_array()` and `load_image()`
  are available in the environment
"""

# =====================================================================
# Basic Parameters
# =====================================================================

# ee.Authenticate()

# version = 1
# biome = 'amazonia'

# Available biomes:
# ["cerrado", "pampa", "caatinga", "mata_atlantica", "amazonia"]

# region = '1'

folder = '../dados'

folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_col4'

folder_mosaic = f'../../../mnt/Files-Geo/Arquivos/col3_mosaics_landsat_30m/vrt/{biome}'

sulfix = ''


# =====================================================================
# Satellite and Year Configuration
# =====================================================================

satellite_years = [

    # {'satellite': 'l5', 'years': [1985,1986,...]}
    # {'satellite': 'l57', 'years': [1999,...]}
    # {'satellite': 'l78', 'years': [2013,...]}
    # {'satellite': 'l89', 'years': [2022,2023,2024]}

    {'satellite': 'l89', 'years': [2024]}
]


# =====================================================================
# GPU Configuration
# =====================================================================

physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# =====================================================================
# Data Reshaping Utilities
# =====================================================================

def reshape_sigle_vector(data_classify):
    """
    Convert raster cube into a single vector of pixels.
    """
    data_classify_vector = data_classify.reshape(
        [data_classify.shape[0] * data_classify.shape[1],
         data_classify.shape[2]]
    )

    return data_classify_vector


# =====================================================================
# Classification
# =====================================================================

def classify(data_classify_vector):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

    with tf.Session(graph=graph,
                    config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # Restore trained model
        saver.restore(
            sess,
            f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt'
        )

        # Process data in chunks to avoid memory overflow
        output_data_classify0 = outputs.eval(
            {x_input: data_classify_vector[:4000000, bi]})

        output_data_classify1 = outputs.eval(
            {x_input: data_classify_vector[4000000:8000000, bi]})

        output_data_classify2 = outputs.eval(
            {x_input: data_classify_vector[8000000:12000000, bi]})

        output_data_classify3 = outputs.eval(
            {x_input: data_classify_vector[12000000:, bi]})

        output_data_classify = np.concatenate([
            output_data_classify0,
            output_data_classify1,
            output_data_classify2,
            output_data_classify3
        ])

    tf.keras.backend.clear_session()

    return output_data_classify


# =====================================================================
# Raster Reshaping
# =====================================================================

def reshape_image_output(output_data_classified, data_classify):

    output_image_data = output_data_classified.reshape(
        [data_classify.shape[0], data_classify.shape[1]]
    )

    return output_image_data


# =====================================================================
# Spatial Filtering
# =====================================================================

def filter_spacial(output_image_data):

    binary_image = output_image_data > 0

    # Remove small white regions
    open_image = ndimage.binary_opening(
        binary_image,
        structure=np.ones((4, 4))
    )

    # Remove small black holes
    close_image = ndimage.binary_closing(
        open_image,
        structure=np.ones((8, 8))
    )

    return close_image


# =====================================================================
# Raster Export
# =====================================================================

def convert_to_raster(dataset_classify, image_data_scene, output_image_name):

    cols = dataset_classify.RasterXSize
    rows = dataset_classify.RasterYSize

    driver = gdal.GetDriverByName('GTiff')

    outDs = driver.Create(
        output_image_name,
        cols,
        rows,
        1,
        gdal.GDT_Float32
    )

    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    geotrans = dataset_classify.GetGeoTransform()
    proj = dataset_classify.GetProjection()

    outDs.SetGeoTransform(geotrans)
    outDs.SetProjection(proj)

    outDs.FlushCache()
    outDs = None


# =====================================================================
# Scene Classification
# =====================================================================

def render_classify(dataset_classify):

    data_classify = convert_to_array(dataset_classify)

    data_classify_vector = reshape_sigle_vector(data_classify)

    output_data_classified = classify(data_classify_vector)

    output_image_data = reshape_image_output(
        output_data_classified,
        data_classify
    )

    return filter_spacial(output_image_data)


# =====================================================================
# Landsat Grid Loader
# =====================================================================

def read_grid_landsat():

    grid = ee.FeatureCollection(
        f'users/geomapeamentoipam/AUXILIAR/grid_regions/grid-{biome}-{region}'
    )

    grid_features = grid.getInfo()['features']

    return grid_features


# =====================================================================
# Geometry Utilities
# =====================================================================

def meters_to_degrees(meters, latitude):

    return meters / (111320 * abs(math.cos(math.radians(latitude))))


def expand_geometry(geometry, buffer_distance_meters=50):

    geom = shape(geometry)

    centroid_lat = geom.centroid.y

    buffer_distance_degrees = meters_to_degrees(
        buffer_distance_meters,
        centroid_lat
    )

    expanded_geom = geom.buffer(buffer_distance_degrees)

    return mapping(expanded_geom)


def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):

    geom_shape = shape(geom)
    image_shape = box(*image_bounds)

    intersection = geom_shape.intersection(image_shape)

    return intersection.area >= min_intersection_area


# =====================================================================
# Raster Clipping
# =====================================================================

def clip_image_by_grid(geom, image, output, buffer_distance_meters=100):

    with rasterio.open(image) as src:

        expanded_geom = expand_geometry(geom, buffer_distance_meters)

        try:

            if has_significant_intersection(expanded_geom, src.bounds):

                out_image, out_transform = mask(
                    src,
                    [expanded_geom],
                    crop=True,
                    nodata=np.nan,
                    filled=True
                )

            else:

                print(f'Skipping image: {image} - insufficient overlap')
                return

        except ValueError as e:

            print(f'Skipping image: {image} - {str(e)}')
            return

    out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)


# =====================================================================
# Main Processing Loop
# =====================================================================

grid_landsat = read_grid_landsat()

start_time = time.time()

for satellite_year in satellite_years:

    satellite = satellite_year['satellite']
    years = satellite_year['years']

    for year in years:

        vrt_NBR = f'{folder_mosaic}/{satellite}_{biome}_{year}.vrt'
        # os.system(f'gdalbuildvrt {vrt_NBR} {files_NBR}')

        input_scenes = []
        total_scenes_done = 0

        for grid in grid_landsat:

            orbit = grid['properties']['ORBITA']
            point = grid['properties']['PONTO']

            output_image_name = (
                f'{folder}/image_col3_{biome}_r{region}_v{version}_'
                f'{orbit}_{point}_{year}.tif'
            )

            if not os.path.isfile(output_image_name):

                geometry_cena = grid['geometry']
                feature_grid = ee.Feature(grid)

                # Scene area in km²
                area_grid = feature_grid.area().divide(1000 * 1000).getInfo()

                # -----------------------------------------------------
                # Clip NBR mosaic for the scene
                # -----------------------------------------------------

                print(colored(
                    f'Clipping mosaic image for scene {orbit}/{point}',
                    'cyan'
                ))

                NBR_clipped = (
                    f'{folder}/image_mosaic_col3_{biome}_r{region}_v{version}_'
                    f'{orbit}_{point}_clipped_{year}.tif'
                )

                try:

                    clip_image_by_grid(
                        geometry_cena,
                        vrt_NBR,
                        NBR_clipped
                    )

                except ValueError as e:

                    print(colored(
                        f'Error clipping image: {NBR_clipped} - {str(e)}',
                        'red'
                    ))

                    print(
                        f'Full path to clipped image: '
                        f'{os.path.abspath(NBR_clipped)}'
                    )

                # -----------------------------------------------------
                # Load clipped raster
                # -----------------------------------------------------

                images = []
                dataset_classify = None

                try:

                    image = NBR_clipped

                    if os.path.isfile(image):

                        dataset_classify = load_image(image)

                    else:

                        print(colored(
                            f'Image not found: {image}',
                            'red'
                        ))

                except:

                    print(colored(
                        f'Image not found: {NBR_clipped}',
                        'red'
                    ))

                # -----------------------------------------------------
                # Scene classification
                # -----------------------------------------------------

                if dataset_classify:

                    try:

                        image_data = render_classify(dataset_classify)

                        convert_to_raster(
                            dataset_classify,
                            image_data,
                            output_image_name
                        )

                    except Exception as e:

                        print(colored(
                            f'Error during classification of image: {image}',
                            'red'
                        ))

                        print(f'Error details: {str(e)}')

                        continue


            # ---------------------------------------------------------
            # Scene progress tracking
            # ---------------------------------------------------------

            total_scenes_done += 1

            if os.path.isfile(output_image_name):

                input_scenes.append(output_image_name)

                print(colored(
                    f'Done in {year}, {total_scenes_done} scenes of '
                    f'{len(grid_landsat)} scenes '
                    f'({3:.2f}% complete)'
                    .format(total_scenes_done / len(grid_landsat) * 100),
                    'green'
                ))

        # =================================================================
        # Merge all scenes for the year
        # =================================================================

        if len(input_scenes) > 0:

            input_scenes = " ".join(input_scenes)

            image_name = (
                f"queimada_{biome}_{satellite}_v{version}_"
                f"region{region}_{year}{sulfix}"
            )

            output_image = f"{folder}/{image_name}.tif"

            print(colored('Merging all scenes', 'yellow'))

            os.system(
                f'gdal_merge.py -n 0 '
                f'-co COMPRESS=PACKBITS '
                f'-co BIGTIFF=YES '
                f'-of gtiff '
                f'{input_scenes} '
                f'-o {output_image}'
            )

            # -------------------------------------------------------------
            # Upload results to Google Cloud Storage
            # -------------------------------------------------------------

            os.system(
                f'gsutil -m cp {output_image} '
                f'gs://tensorflow-fire-cerrado1/'
                f'result_classified_colecao5/{biome}'
            )

            # -------------------------------------------------------------
            # Import result into Google Earth Engine
            # -------------------------------------------------------------

            outputAssetID = (
                f'projects/ee-geomapeamentoipam/assets/'
                f'MAPBIOMAS_FOGO/COLECAO_5/'
                f'{biome.upper()}/{image_name}'
            )

            bucket = (
                f'gs://tensorflow-fire-cerrado1/'
                f'result_classified_colecao5/'
                f'{biome}/{image_name}.tif'
            )

            os.system(
                f'earthengine upload image '
                f'--asset_id={outputAssetID} {bucket}'
            )

            # -------------------------------------------------------------
            # Clean temporary files
            # -------------------------------------------------------------

            os.system(f'rm -rf {folder}/image_*')

            print(colored(f'Done {year}', 'green'))

            elapsed_time = time.time() - start_time

            print(colored(
                'Spent time: {0}'.format(
                    time.strftime(
                        "%H:%M:%S",
                        time.gmtime(elapsed_time)
                    )
                ),
                'yellow'
            ))

print(colored('Done all.', 'green'))