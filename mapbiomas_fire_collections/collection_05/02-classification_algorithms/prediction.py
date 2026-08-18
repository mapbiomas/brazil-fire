#!/usr/bin/env python
# coding: utf-8

#ee.Authenticate()
# -- PARÂMETROS BÁSICOS --
# ------------------------------------------------
#version = 1
#biome = 'amazonia' #["cerrado", "pampa",  "caatinga", "cerrado", "mata_atlantica","amazonia"] 
#region = '1'
folder = '../dados'# pasta principal onde fica os dados
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_col5'
folder_mosaic = f'../../../mnt/Files-Geo/Arquivos/col3_mosaics_landsat_30m/vrt/{biome}'
sulfix = ''

satellite_years = [
    #{'satellite': 'l5', 'years':  [1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998]}
    #{'satellite': 'l5', 'years':  [1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998]}
    #{'satellite': 'l57', 'years': [1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012]},
    #{'satellite': 'l78', 'years': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]},
    #{'satellite': 'l89', 'years': [2022, 2023, 2024]}
    {'satellite': 'l89', 'years': [2025]}
]

# Set the GPU memory growth to limit memory usage
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# FAZ A PREDIÇÃO

# Reshape into a single vector of pixels for data classify
def reshape_sigle_vector(data_classify):
    data_classify_vector = data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])
    return data_classify_vector

def classify(data_classify_vector):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        # restaura o valor das variáveis
        saver.restore(sess, f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt')

        # faz a classificação - divide em cinco partes para evitar "Kernel memory"
        output_data_classify0 = outputs.eval({x_input: data_classify_vector[:4000000, bi]})
        output_data_classify1 = outputs.eval({x_input: data_classify_vector[4000000:8000000, bi]})
        output_data_classify2 = outputs.eval({x_input: data_classify_vector[8000000:12000000, bi]})
        output_data_classify3 = outputs.eval({x_input: data_classify_vector[12000000:, bi]})

        output_data_classify = np.concatenate([
            output_data_classify0,
            output_data_classify1,
            output_data_classify2,
            output_data_classify3
        ])
    # Libera recursos da sessão TensorFlow
    tf.keras.backend.clear_session()
    return output_data_classify

def reshape_image_output(output_data_classified, data_classify):
    output_image_data = output_data_classified.reshape(
        [data_classify.shape[0], data_classify.shape[1]])
    return output_image_data

def filter_spacial(output_image_data):
    binary_image = output_image_data > 0

    # Remove small white regions
    open_image = ndimage.binary_opening(
        #pampa
        #binary_image, structure=np.ones((2, 2)))
    #outros biomas
        binary_image, structure=np.ones((4, 4)))
    # Remove small black hole
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))
    return close_image

def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    cols = dataset_classify.RasterXSize
    rows = dataset_classify.RasterYSize
    ds = dataset_classify

    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # follow code is adding GeoTransform and Projection
    geotrans = ds.GetGeoTransform()  # get GeoTransform from existed 'data0'
    proj = ds.GetProjection()  # you can get from a existed tif or import
    outDs.SetGeoTransform(geotrans)
    outDs.SetProjection(proj)
    outDs.FlushCache()
    outDs = None

def render_classify(dataset_classify):
    data_classify = convert_to_array(dataset_classify)

    data_classify_vector = reshape_sigle_vector(data_classify)
    # data_classify_vector = add_server_features(data_classify_vector)

    output_data_classified = classify(data_classify_vector)

    output_image_data = reshape_image_output(
        output_data_classified, data_classify)

    return filter_spacial(output_image_data)

def read_grid_landsat():

    grid = ee.FeatureCollection(f'users/geomapeamentoipam/AUXILIAR/grid_regions/grid-{biome}-{region}')

    grid_features = grid.getInfo()['features']
    return grid_features

# Função para converter metros em graus decimais
def meters_to_degrees(meters, latitude):
    # Aproximação para conversão de metros para graus
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Função para expandir a geometria com buffer em metros
def expand_geometry(geometry, buffer_distance_meters=50):
    geom = shape(geometry)
    
    # Obter a latitude do centróide da geometria
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Função para verificar interseção significativa
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)  # Convert BoundingBox to Polygon
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

# Função de recorte ajustada
def clip_image_by_grid(geom, image, output, buffer_distance_meters=100):
    with rasterio.open(image) as src:
        print(f"Image path: {image}")
        print(f"Output path: {output}")
        
        # Expandir a geometria com buffer em metros
        expanded_geom = expand_geometry(geom, buffer_distance_meters)
        
        try:
            if has_significant_intersection(expanded_geom, src.bounds):
                out_image, out_transform = mask(
                    src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
            else:
                print(f'Skipping image: {image} - Insufficient overlap with raster.')
                return
        except ValueError as e:
            print(f'Skipping image: {image} - {str(e)}')
            return

    out_meta = src.meta.copy()

    # Salvar o raster resultante
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)


grid_landsat = read_grid_landsat()
start_time = time.time()

for satellite_year in satellite_years:
    satellite = satellite_year['satellite']
    years = satellite_year['years']

    for year in years:
        vrt_NBR = f'{folder_mosaic}/{satellite}_{biome}_{year}.vrt'
        #os.system(f'gdalbuildvrt {vrt_NBR} {files_NBR}')    

        input_scenes = []
        total_scenes_done = 0
        for grid in grid_landsat:

            orbit = grid['properties']['ORBITA']
            point = grid['properties']['PONTO']

            output_image_name = f'{folder}/image_col3_{biome}_r{region}_v{version}_{orbit}_{point}_{year}.tif'

            if not os.path.isfile(output_image_name):

                geometry_cena = grid['geometry']
                feature_grid = ee.Feature(grid)
                area_grid = feature_grid.area().divide(1000 * 1000).getInfo() # km²

                #clip da imagem de NBR
                print(colored(f'Clipping image mosic for scene {orbit}/{point}', 'cyan'))
                NBR_clipped = f'{folder}/image_mosaic_col3_{biome}_r{region}_v{version}_{orbit}_{point}_clipped_{year}.tif'

                try:
                    clip_image_by_grid(geometry_cena, vrt_NBR, NBR_clipped)
                except ValueError as e:
                    print(colored(f'Error clipping image: {NBR_clipped} - {str(e)}', 'red'))
                    print(f'Full path to the clipped image: {os.path.abspath(NBR_clipped)}')

                images = []
                dataset_classify = None
                try:
                    image = NBR_clipped

                    if os.path.isfile(image):

                        dataset_classify = load_image(image)
                    else:
                        print(colored(f'Image not found: {image}', 'red'))
                except:
                    print(colored(f'Image not found: {NBR_clipped}', 'red'))

                # if dataset_classify:
                #     try:
                #         image_data = render_classify(dataset_classify)
                #         convert_to_raster(dataset_classify, image_data, output_image_name)
                #     except:
                #         print(colored(f'Erro classify image: {image}', 'red'))

                if dataset_classify:
                    try:
                        image_data = render_classify(dataset_classify)
                        convert_to_raster(dataset_classify, image_data, output_image_name)
                    except Exception as e:
                        print(colored(f'Error during classification of image: {image}', 'red'))
                        print(f'Error details: {str(e)}')
                        continue  # Skip to the next iteration in case of an error

            total_scenes_done += 1

            if os.path.isfile(output_image_name):
                input_scenes.append(output_image_name)

                print(colored(f'Done in {year}, {total_scenes_done} scenes of {len(grid_landsat)} scenes ({3:.2f}% complete)'.format(total_scenes_done/len(grid_landsat)*100), 'green'))

            # ----------------------------------------------------

        if len(input_scenes) > 0:
            input_scenes = " ".join(input_scenes)

            image_name = f"queimada_{biome}_{satellite}_v3_region{region}_{year}{sulfix}"
            output_image = f"{folder}/{image_name}.tif"

            print(colored('Merging all scenes', 'yellow'))
            os.system(f'gdal_merge.py -n 0 -co COMPRESS=PACKBITS -co BIGTIFF=YES -of gtiff {input_scenes} -o {output_image}')

            os.system(f'gsutil -m cp {output_image} gs://tensorflow-fire-cerrado1/result_classified_colecao5/{biome}')

            outputAssetID = f'projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_5/{biome.upper()}/{image_name}'
            bucket = f'gs://tensorflow-fire-cerrado1/result_classified_colecao5/{biome}/{image_name}.tif'

            os.system(f'earthengine upload image --asset_id={outputAssetID} {bucket}')

            os.system(f'rm -rf {folder}/image_*')

            print(colored(f'Done {year}', 'green'))

            elapsed_time = time.time() - start_time
            print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))), 'yellow'))

print(colored('Done all.', 'green'))