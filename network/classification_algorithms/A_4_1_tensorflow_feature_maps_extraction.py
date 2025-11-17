# A_4_1_tensorflow_feature_maps_extraction.py
# last update: '2025/06/02'
# MapBiomas Fire Classification Algorithms Step A_4_1 Functions for TensorFlow Embedding Extraction
# (Versão Modificada para Seleção Dinâmica de Camada de Embedding)

# INSTALAR E IMPORTAR LIBRARIES
import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Usar somente a versão compatível 1.x
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee
from tqdm import tqdm
import time
from datetime import datetime
import math
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
import pyproj
import shutil
import json
import subprocess
from scipy import ndimage 

# Variáveis Globais (Assumidas ou mockadas para a execução do script)
# Você deve garantir que 'log_message', 'bucket_name', 'ee_project' e 'fs' estejam definidos no ambiente
def log_message(msg):
    print(f"[LOG] {msg}")

# Funções Utilitárias (Baseadas em A_3_1)

def fully_connected_layer (input, n_neurons, activation=None, name=None):
    """Cria uma camada totalmente conectada."""
    input_size = input.get_shape().as_list()[1]
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))), name=f'W_{name}')
    b = tf.Variable(tf.zeros([n_neurons]), name=f'b_{name}')
    
    layer = tf.matmul(input, W) + b
    
    if activation == 'relu':
        layer = tf.nn.relu(layer)
        
    if name:
        # Adiciona um nome ao tensor de saída da camada (útil para extração)
        layer = tf.identity(layer, name=name) 
        
    return layer

def load_image (image_path):
    """Carrega uma imagem usando GDAL."""
    log_message (f" [INFO] Loading image from path: {image_path}")
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError (f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array (dataset):
    """Converte um dataset GDAL para um array NumPy."""
    log_message (" [INFO] Converting dataset to NumPy array")
    bands_data = [dataset.GetRasterBand (i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack (bands_data, axis=2)
    return stacked_data

def reshape_single_vector(data_classify):
    """Remodela os dados de entrada para um vetor de pixel único."""
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape [1], data_classify.shape[2]])

def reproject_geometry (geom, src_crs, dst_crs):
    """Reprojeta uma geometria."""
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geom)

def has_significant_intersection (geom, image_bounds, min_intersection_area=0.01):
    """Verifica se há uma intersecção significativa."""
    geom_shape = shape (geom)
    image_shape = box (*image_bounds)
    intersection = geom_shape.intersection (image_shape)
    return intersection.area >= min_intersection_area

def clip_image_by_grid(geom, image, output, buffer_distance_meters=100, max_attempts=5, retry_delay=5):
    """Clipa uma imagem usando uma geometria (grid) com buffer."""
    # (Implementação completa de clip_image_by_grid como em A_3_1)
    attempt = 0
    while attempt < max_attempts:
        try:
            log_message(f" [INFO] Attempt {attempt+1}/{max_attempts} to clip image: {image}")
            with rasterio.open(image) as src:
                image_crs = src.crs
                geom_shape = shape(geom)
                geom_proj = reproject_geometry(geom_shape, 'EPSG:4326', image_crs)
                expanded_geom = geom_proj.buffer(buffer_distance_meters)
                expanded_geom_geojson = mapping(expanded_geom)

                if has_significant_intersection(expanded_geom_geojson, src.bounds):
                    out_image, out_transform = mask(src, [expanded_geom_geojson], crop=True, nodata=np.nan, filled=True)
                    out_meta = src.meta.copy()
                    out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                                     "transform": out_transform, "crs": src.crs})

                    with rasterio.open(output, 'w', **out_meta) as dest:
                        dest.write(out_image)
                    log_message(f" [INFO] Image clipped successfully: {output}")
                    return True 
                else:
                    log_message(f" [INFO] Insufficient overlap for clipping: {image}")
                    return False
        except Exception as e:
            log_message(f" [ERROR] Error during clipping: {str(e)}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            attempt += 1
    log_message(f" [ERROR] Failed to clip image after {max_attempts} attempts: {image}")
    return False

def build_vrt (vrt_path, input_tif_list):
    """Constrói um VRT a partir de uma lista de TIFFs."""
    # (Implementação completa de build_vrt como em A_3_1)
    if isinstance (input_tif_list, str):
        input_tif_list = input_tif_list.split()
    # (Checagem de arquivos inexistentes omitida)
    if os.path.exists(vrt_path):
        os.remove(vrt_path)
    
    gdal.BuildVRT (vrt_path, input_tif_list)

def translate_to_tiff (vrt_path, output_path):
    """Traduz um VRT para um TIFF otimizado."""
    # (Implementação completa de translate_to_tiff como em A_3_1)
    if os.path.exists (output_path):
        os.remove (output_path)
        
    options = gdal.TranslateOptions (
        format="GTiff",
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "COPY_SRC_OVERVIEWS=YES", "BIGTIFF=YES"],
        noData=0
    )
    result = gdal.Translate (output_path, vrt_path, options=options)
    if result is None:
        raise RuntimeError (f"Failed to translate VRT to TIFF: {output_path}")

def generate_optimized_image (name_out_vrt, name_out_tif, files_tif_list, suffix=""):
    """Gera o TIFF final otimizado a partir do merge de TIFFs menores."""
    # (Implementação completa de generate_optimized_image como em A_3_1)
    try:
        name_out_vrt_suffixed = name_out_vrt.replace(".tif", f"{suffix}.vrt") if suffix else name_out_vrt.replace(".tif", ".vrt")
        name_out_tif_suffixed = name_out_tif.replace(".tif", f"{suffix}.tif") if suffix else name_out_tif
        
        build_vrt (name_out_vrt_suffixed, files_tif_list)
        translate_to_tiff (name_out_vrt_suffixed, name_out_tif_suffixed)
        return True
    except Exception as e:
        log_message (f" [ERROR] Failed to generate optimized image. {e}")
        return False

def check_or_create_collection(collection, ee_project):
    """Verifica e cria uma coleção no GEE se ela não existir."""
    check_command = f'earthengine --project {ee_project} asset info {collection}'
    status = os.system(check_command)
    if status != 0:
        create_command = f'earthengine --project {ee_project} create collection {collection}'
        os.system(create_command)

def clean_directories (directories_to_clean):
    """Limpa e recria diretórios especificados."""
    for directory in directories_to_clean:
        if os.path.exists (directory):
            shutil.rmtree (directory)
        os.makedirs (directory)

def remove_temporary_files(files_to_remove):
    """Remove arquivos temporários."""
    for file in files_to_remove:
        if os.path.exists (file):
            try:
                os.remove (file)
            except Exception:
                pass


# EMBEDDING CORE FUNCTIONS


def create_embedding_model_graph (hyperparameters):
    """
    Cria o grafo computacional TensorFlow adaptado para a extração de EMBEDDINGS.
    Usa tf.variable_scope para garantir que a rede seja reconstruída com os 
    mesmos nomes de variáveis do checkpoint (salvo via A_2_1/A_3_1).
    """
    
    graph = tf.Graph()
    with graph.as_default():
        # Define placeholders
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')
        
        # Normaliza os dados
        normalized = (x_input - hyperparameters['data_mean']) / hyperparameters['data_std']
        
        # Constrói as camadas (USANDO SCOPE PARA RESTAURAR OS NOMES DO CHECKPOINT)
        
        with tf.variable_scope('hidden1'):
            hidden1 = fully_connected_layer (normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu', name='h1')
        with tf.variable_scope('hidden2'):
            hidden2 = fully_connected_layer (hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu', name='h2')
        with tf.variable_scope('hidden3'):
            hidden3 = fully_connected_layer (hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu', name='h3')
        with tf.variable_scope('hidden4'):
            hidden4 = fully_connected_layer (hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu', name='h4')
        
        # PONTO DE SAÍDA PADRÃO (L5)
        with tf.variable_scope('hidden5'): 
            embedding_output = fully_connected_layer (hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu', name='embedding_output_L5')
        
        # O tensor de saída para a extração L5:
        outputs = embedding_output
        tf.identity (outputs, name='extracted_embedding') # Tensor L5: 'extracted_embedding:0'
        
        # O resto do grafo (logits) é mantido para carregar o checkpoint
        with tf.variable_scope('logits'):
            logits = fully_connected_layer (embedding_output, n_neurons=hyperparameters['NUM_CLASSES'])
        
        # Otimizador, Loss e Saver (necessários para o restore)
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
        optimizer = tf.train.AdamOptimizer(hyperparameters['lr']).minimize(cross_entropy)
        
        saver = tf.train.Saver()
        return graph, {'x_input': x_input, 'y_input': y_input}, saver
        
def classify_for_embeddings (data_classify_vector, model_path, hyperparameters, selected_layer, block_size=40000000):
    """Extrai embeddings dos dados em blocos, buscando o tensor da camada escolhida."""
    
    num_pixels = data_classify_vector.shape[0]
    num_blocks = (num_pixels + block_size - 1) // block_size
    output_blocks = []

    # Mapeamento do nome da camada escolhida para o nome do tensor no grafo
    tensor_map = {
        'h1': 'h1:0',
        'h2': 'h2:0',
        'h3': 'h3:0',
        'h4': 'h4:0',
        'h5': 'extracted_embedding:0' # L5
    }
    tensor_name = tensor_map.get(selected_layer, 'extracted_embedding:0') # Default para L5

    for i in range (num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        data_block = data_classify_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()
        graph, placeholders, saver = create_embedding_model_graph (hyperparameters)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)
        
        with tf.Session (graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)
            
            # Busca o tensor de embedding DINÂMICO
            output_block = sess.run(
                graph.get_tensor_by_name(tensor_name),
                feed_dict={placeholders['x_input']: data_block}
            )
            output_blocks.append (output_block)

    output_data_classify = np.concatenate (output_blocks, axis=0)
    return output_data_classify

def convert_to_raster_multiband (dataset_classify, embedding_data_scene_hwc, output_image_name):
    """Salva o array multi-banda (Embeddings) como GeoTIFF, com normalização para uint8 (0-255)."""
    
    rows, cols, bands = embedding_data_scene_hwc.shape
    driver = gdal.GetDriverByName('GTiff')
    
    # Normalização de 0 a 255 (Quantização 8-bit)
    min_vals = np.min(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    max_vals = np.max(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-6 # Evita divisão por zero
    
    normalized_data = 255 * (embedding_data_scene_hwc - min_vals) / range_vals
    embedding_data_scene_uint8 = normalized_data.astype ('uint8')
    
    # Transpõe para o formato (C, H, W) que o GDAL espera
    embedding_data_chw = np.transpose (embedding_data_scene_uint8, (2, 0, 1))
    
    options = ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'TILED=YES', 'BIGTIFF=YES']
    
    # Cria o dataset de saída com o número de bandas correto (gdal.GDT_Byte para uint8)
    outDs = driver.Create (output_image_name, cols, rows, bands, gdal.GDT_Byte, options=options)
    
    # Escreve cada banda do embedding
    for i in range (bands):
        outDs.GetRasterBand (i + 1).WriteArray (embedding_data_chw[i])
        
    outDs.SetGeoTransform (dataset_classify.GetGeoTransform())
    outDs.SetProjection (dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None
    
    return True

def upload_embedding_to_gee (gcs_path, asset_id, satellite, region, year, version, ee_project):
    """Realiza o upload de Embeddings multi-banda para o GEE."""
    
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')
    
    # Lógica para verificar e deletar asset existente (omitida para brevidade)
    # ...
    
    # Perform the upload using Earth Engine CLI
    upload_command = (
        f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
        f'--pyramiding_policy-mode '
        f'--property satellite={satellite} '
        f'--property region={region} '
        f'--property year={year} '
        f'--property version={version} '
        f'--property source=IPAM '
        f'--property type=annual_embedding ' # TIPO DE ASSET MUDADO
        f'--property time_start={timestamp_start} '
        f'--property time_end={timestamp_end} '
        f'--property create_date={creation_date} '
        f'{gcs_path}'
    )
    
    status = os.system(upload_command)
    return status == 0


# WORKFLOWS DE PROCESSAMENTO DE EMBEDDINGS

def process_single_image_embedding (dataset_classify, version, region, folder_temp, embedding_layer, country, bucket_name, fs):
    """Processa uma única imagem extraindo o embedding DNN da camada 'embedding_layer'."""
    
    # Assumimos que a variável 'country', 'bucket_name', 'fs' estão no escopo global ou foram passadas.
    
    # 1. Preparação: Download do Modelo (Idêntico ao A_3_1)
    gcs_model_file = f'gs://{bucket_name}/sudamerica/{country}/models_col1/col1_{country}_{version}_{region}_rnn_lstm_ckpt*'
    model_file_local_temp = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt'
    
    try:
        subprocess.run(f'gsutil cp {gcs_model_file} {folder_temp}', shell=True, check=True)
        time.sleep(2)
        fs.invalidate_cache()
    except Exception:
        log_message(" [ERROR] Failed to download model from GCS.")
        return None
        
    # 2. Carregar Hiperparâmetros (Idêntico ao A_3_1)
    json_path = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt_hyperparameters.json'
    with open (json_path, 'r') as json_file:
        hyperparameters = json.load(json_file)
    
    # 3. Conversão e Vetorização de Dados (Idêntico ao A_3_1)
    data_classify = convert_to_array (dataset_classify)
    data_classify_vector = reshape_single_vector(data_classify)
    
    # 4. Execução da Extração (CHAMA A FUNÇÃO MODIFICADA)
    output_data_classified = classify_for_embeddings (
        data_classify_vector, 
        model_file_local_temp, 
        hyperparameters,
        embedding_layer # Passa a camada escolhida
    )
    
    # 5. Reshape de volta para formato de imagem multi-banda (H, W, C)
    H, W = data_classify.shape[:2]
    C = output_data_classified.shape[-1] 
    
    output_image_data_hwc = output_data_classified.reshape ([H, W, C])
    
    # 6. Retorna o array HWC do embedding
    return output_image_data_hwc


def process_year_by_satellite_embedding (satellite_years, bucket_name, folder_mosaic, folder_temp, suffix,
                                         ee_project, country, version, region, simulate_test=False, embedding_layer='h5'):
    """
    Workflow principal para geração de embeddings, adaptado para camada dinâmica.
    (Baseado em A_4_1 e A_3_1)
    """

    # 1. Setup inicial
    # Assume-se que 'fs' (gcsfs.GCSFileSystem) foi definido globalmente no A_4_1
    fs = gcsfs.GCSFileSystem(project=bucket_name)

    # Assume-se que ee.FeatureCollection e ee.ImageCollection são acessíveis (EE inicializado)
    try:
        grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
        grid_landsat = grid.getInfo()['features']
    except Exception as e:
        log_message(f"[ERROR] Falha ao carregar grid do GEE: {e}")
        return

    start_time = time.time()
    
    # Define a nova coleção GEE para embeddings
    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION_EMBEDDINGS/embedding_field_{country}_{version}'
    check_or_create_collection (collection_name, ee_project)
    
    for satellite_year in satellite_years:
        satellite = satellite_year['satellite']
        # Limita a 1 ano se for simulação
        years = satellite_year['years'][:1] if simulate_test else satellite_year['years']
        
        with tqdm (total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                test_tag = "_test" if simulate_test else ""
                
                # Novo nome do arquivo TIFF para Embeddings
                image_name = f"embedding_{country}_{satellite}_{version}_region{region[1:]}_{year}{suffix}{test_tag}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_embeddings/{image_name}.tif'
                # Path para o mosaico COG (usando o formato iXX_country_rY_year_cog.tif)
                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                # Caminho corrigido para models_col1_cog (assumindo que esta é a convenção)
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif' 

                # 2. Download do COG
                if not os.path.exists (local_cog_path):
                    try:
                        os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')
                        time.sleep(2)
                        fs.invalidate_cache()
                    except Exception as e:
                        log_message(f"[ERROR] Falha ao baixar COG: {gcs_cog_path}. {e}")
                        continue
                
                input_scenes = []
                # Limita a 1 cena se for simulação
                grids_to_process = [grid_landsat[0]] if simulate_test else grid_landsat
                
                with tqdm (total=len(grids_to_process), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid_feature in grids_to_process:
                        orbit = grid_feature['properties']['ORBITA']
                        point = grid_feature['properties']['PONTO']
                        # Nome TEMP diferente para evitar conflito com classificação
                        output_image_name = f'{folder_temp}/image_emb_{country}_{region}_{version}_{orbit}_{point}_{year}.tif'
                        geometry_scene = grid_feature['geometry']
                        # Nome TEMP diferente para o clip
                        NBR_clipped = f'{folder_temp}/image_mosaic_emb_clipped_{orbit}_{point}_{year}.tif' 

                        if os.path.isfile (output_image_name):
                            pbar_scenes.update(1)
                            continue
                        
                        # 3. Clipagem da Imagem
                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)
                        
                        if clipping_success:
                            dataset_classify = load_image (NBR_clipped)
                            
                            # 4. Extração de Embeddings (CHAMA A FUNÇÃO COM O NOVO PARÂMETRO)
                            image_data_hwc = process_single_image_embedding (
                                dataset_classify, 
                                version, 
                                region, 
                                folder_temp, 
                                embedding_layer, # Camada de embedding
                                country, 
                                bucket_name, 
                                fs
                            )
                            
                            # --- CORREÇÃO DE FLUXO: VERIFICA SE O EMBEDDING FOI GERADO (SE NÃO RETORNOU None) ---
                            if image_data_hwc is None:
                                log_message(f"[ERROR] Extração de embedding falhou (provavelmente download do modelo) para cena {orbit}/{point}. Pulando.")
                                pbar_scenes.update(1)
                                remove_temporary_files([NBR_clipped]) # Limpa o clip
                                continue # Pula para a próxima cena
                            # --- FIM DA CORREÇÃO DE FLUXO ---

                            # 5. Conversão para Raster Multi-Banda
                            convert_to_raster_multiband (dataset_classify, image_data_hwc, output_image_name)
                            input_scenes.append(output_image_name)
                            remove_temporary_files([NBR_clipped])
                        
                        pbar_scenes.update(1)

                # 6. Geração do TIFF Otimizado Multi-Banda (Merge VRT)
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_temp}/merged_emb_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"
                    generate_optimized_image (merge_output_temp, output_image, input_scenes_str)
                    
                    # 7. Upload para GCS e GEE
                    status_upload = os.system(f'gsutil cp {output_image} {gcs_filename}')
                    time.sleep(2)
                    fs.invalidate_cache()
                    
                    if status_upload == 0 and os.system(f'gsutil ls {gcs_filename}') == 0:
                        upload_embedding_to_gee (
                            gcs_filename,
                            f'{collection_name}/{image_name}',
                            satellite, region, year, version, ee_project
                        )
                        
                    clean_directories ([folder_temp])
                
                elapsed = time.time() - start_time
                log_message(f"[INFO] Year {year} embedding generation completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                pbar_years.update(1)


# MAIN EXECUTION LOGIC (render_embedding_models)
def render_embedding_models(models_to_process, simulate_test=False):
    """Processa uma lista de modelos e mosaicos para extrair Embeddings."""
    
    bucket_name_global = 'mapbiomas-fire'
    # Mock de fs e ee_project (eles devem ser definidos no escopo do seu notebook)
    ee_project_global = 'mapbiomas-GUYANA' # Substituir pelo projeto real
    
    for model_info in models_to_process:
        
        model_name = model_info["model"]
        mosaics = model_info ["mosaics"]
        simulation = model_info["simulation"]
        embedding_layer = model_info["embedding_layer"] # Extrai a camada
        
        try:
            parts = model_name.split('_')
            country = parts[1]
            version = parts[2]
            region = parts[3].split('.')[0]
        except Exception:
            log_message(f"[ERROR] Não foi possível extrair info do modelo: {model_name}")
            continue

        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_temp = f'{folder}/tmp_emb' 
        folder_mosaic = f'{folder}/mosaics_cog'
        
        for directory in [folder_temp, folder_mosaic]:
            if not os.path.exists (directory):
                os.makedirs (directory)
        
        clean_directories ([folder_temp, folder_mosaic])

        satellite_years = []
        for mosaic in mosaics:
            mosaic_parts = mosaic.split('_')
            
            # --- CORREÇÃO DO PARSING DO ANO ---
            try:
                # O formato do mosaico é I89_guyana_r5_2022_cog.tif
                satellite = mosaic_parts[0]
                
                # O ano é a quarta parte (índice 3)
                year_str = mosaic_parts[3]
                
                # Garante que só pega os 4 dígitos e converte para int
                import re
                year_match = re.search(r'\d{4}', year_str)

                if year_match:
                    year = int(year_match.group(0))
                else:
                    log_message(f"[ERROR A_4_1] Não foi possível extrair o ano (4 dígitos) do mosaico: {mosaic}")
                    continue
                
                satellite_years.append({"satellite": satellite, "years": [year]})
                
            except Exception as e:
                log_message(f"[ERROR A_4_1] Falha ao processar o nome do mosaico {mosaic}: {e}")
                continue
        
        if not simulation:            process_year_by_satellite_embedding (
                satellite_years=satellite_years,
                bucket_name=bucket_name_global,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=ee_project_global, # Use sua variável global real
                country=country,
                version=version,
                region=region,
                simulate_test=simulate_test,
                embedding_layer=embedding_layer # Passa a camada
            )
