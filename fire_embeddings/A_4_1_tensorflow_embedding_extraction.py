# A_4_1_tensorflow_embedding_extraction.py
# last_update: '2025/06/02'
# MapBiomas Fire Classification Algorithms Step A_4_1 - Functions for TensorFlow Embedding Extraction

# ====================================
# 📦 INSTALL AND IMPORT LIBRARIES
# ====================================

import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Usar somente a versão compatível 1.x [1, 2]

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
from scipy import ndimage # Para filter_spatial, embora não seja usado na saída final de embeddings [5]

# Assumimos que 'log_message', 'bucket_name', 'ee_project' e 'fs' são definidos globalmente em A_0_x.

# ====================================
# 🧰 SUPPORT FUNCTIONS (Definições Mínimas de Utils)
# ====================================

# Funções Utilizadas (Importadas ou redefinidas de A_3_1)
def fully_connected_layer(input, n_neurons, activation=None):
    # Implementação baseada em A_3_1 [6-11]
    input_size = input.get_shape().as_list()[12]
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))), name='W')
    b = tf.Variable(tf.zeros([n_neurons]), name='b')
    layer = tf.matmul(input, W) + b
    if activation == 'relu':
        layer = tf.nn.relu(layer)
    return layer

# Funções utilitárias de I/O e GEE (A_3_1) - Apenas cabeçalhos, assumindo corpo idêntico ao A_3_1
def load_image(image_path):
    # ... (body as in A_3_1 [13, 14])
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Error loading image: {image_path}. Check the path.")
    return dataset

def convert_to_array(dataset):
    # ... (body as in A_3_1 [13-16])
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    stacked_data = np.stack(bands_data, axis=2)
    return stacked_data

def reshape_single_vector(data_classify):
    # ... (body as in A_3_1 [15-18])
    return data_classify.reshape([data_classify.shape * data_classify.shape[12], data_classify.shape[19]])

def clip_image_by_grid(geom, image, output, buffer_distance_meters=100, max_attempts=5, retry_delay=5):
    # ... (body as in A_3_1 [20-27])
    # Assumimos o corpo completo de A_3_1, incluindo reproject_geometry
    pass

def generate_optimized_image(name_out_vrt, name_out_tif, files_tif_list, suffix=""):
    # ... (body as in A_3_1 [28-33])
    # Assumimos o corpo completo de A_3_1, incluindo build_vrt e translate_to_tiff
    pass

def check_or_create_collection(collection, ee_project):
    # ... (body as in A_3_1 [34, 35])
    pass

def clean_directories(directories_to_clean):
    # ... (body as in A_3_1 [30, 33])
    pass
def remove_temporary_files(files_to_remove):
    # ... (body as in A_3_1 [6, 36-38])
    pass


# ====================================
# 🆕 EMBEDDING CORE FUNCTIONS (BIÓPSIA)
# ====================================

def create_embedding_model_graph(hyperparameters):
    """
    Cria o grafo computacional TensorFlow adaptado para a extração de EMBEDDINGS.
    A saída é a última camada oculta (NUM_N_L5), antes dos logits.
    """
    graph = tf.Graph()
    with graph.as_default():
        # Define placeholders
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')

        # Normaliza os dados
        normalized = (x_input - hyperparameters['data_mean']) / hyperparameters['data_std']

        # Constrói as camadas (igual ao A_3_1) [39-41]
        hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu', name='h1')
        hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu', name='h2')
        hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu', name='h3')
        hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu', name='h4')
        
        # PONTO DE BIÓPSIA: A camada oculta final é a saída.
        embedding_output = fully_connected_layer(hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu', name='embedding_output') 
        
        # Definimos o tensor de saída para a extração
        outputs = embedding_output 
        tf.identity(outputs, name='extracted_embedding') # Nomeamos o tensor

        # O resto do grafo (logits, loss, optimizer) é mantido para fins de carregamento do checkpoint
        logits = fully_connected_layer(embedding_output, n_neurons=hyperparameters['NUM_CLASSES']) 
        
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
        optimizer = tf.train.AdamOptimizer(hyperparameters['lr']).minimize(cross_entropy)
        
        saver = tf.train.Saver()
        return graph, {'x_input': x_input, 'y_input': y_input}, saver

def classify_for_embeddings(data_classify_vector, model_path, hyperparameters, block_size=40000000):
    """
    Extrai embeddings dos dados em blocos, buscando o tensor 'extracted_embedding:0'.
    """
    # log_message(f"[INFO] Starting EMBEDDING extraction with model at path: {model_path}")

    num_pixels = data_classify_vector.shape
    num_blocks = (num_pixels + block_size - 1) // block_size
    output_blocks = []

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        # log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")
        data_block = data_classify_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()
        
        # Cria o grafo de embedding
        graph, placeholders, saver = create_embedding_model_graph(hyperparameters) 

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)

            # Busca o tensor de embedding, não a classe predita
            output_block = sess.run(
                graph.get_tensor_by_name('extracted_embedding:0'), 
                feed_dict={placeholders['x_input']: data_block}
            )
            
            output_blocks.append(output_block)
            
    output_data_classify = np.concatenate(output_blocks, axis=0)
    # log_message(f"[INFO] Embedding extraction completed. Shape: {output_data_classify.shape}")
    return output_data_classify

def convert_to_raster_multiband(dataset_classify, embedding_data_scene_hwc, output_image_name):
    """
    Salva o array multi-banda (Embeddings) como GeoTIFF, com normalização para uint8 (0-255).
    Baseado na prática de quantização [4, 42].
    """
    # log_message(f"[INFO] Converting array to multi-band GeoTIFF raster: {output_image_name}")

    rows, cols, bands = embedding_data_scene_hwc.shape 
    driver = gdal.GetDriverByName('GTiff')
    
    # Normalização de 0 a 255 (Quantização 8-bit)
    min_vals = np.min(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    max_vals = np.max(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    
    # Calcula a faixa (range) e evita divisão por zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-6 
    
    # Normaliza para 0-255 e converte para uint8 (8-bit)
    normalized_data = 255 * (embedding_data_scene_hwc - min_vals) / range_vals
    embedding_data_scene_uint8 = normalized_data.astype('uint8')
    
    # Transpõe para o formato (C, H, W) que o GDAL espera
    embedding_data_chw = np.transpose(embedding_data_scene_uint8, (2, 0, 1))

    options = [
        'COMPRESS=DEFLATE',
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]

    # Cria o dataset de saída com o número de bandas correto
    # Tipo de dado GDT_Byte é para uint8 (0-255)
    outDs = driver.Create(output_image_name, cols, rows, bands, gdal.GDT_Byte, options=options) 

    # Escreve cada banda do embedding
    for i in range(bands):
        outDs.GetRasterBand(i + 1).WriteArray(embedding_data_chw[i])

    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None 

    # log_message(f"[INFO] Multi-band raster (Embeddings) saved: {output_image_name}")
    return True

def upload_embedding_to_gee(gcs_path, asset_id, satellite, region, year, version):
    """
    Realiza o upload de Embeddings multi-banda para o GEE.
    Adapta o 'type' para 'annual_embedding'.
    """
    timestamp_start = int(datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.now().strftime('%Y-%m-%d')
    
    # (Lógica para verificar e deletar asset existente - omitida para brevidade, mas idêntica ao A_3_1)

    # Perform the upload using Earth Engine CLI
    upload_command = (
        f'earthengine --project {ee_project} upload image --asset_id={asset_id} '
        f'--pyramiding_policy=mode '
        f'--property satellite={satellite} '
        f'--property region={region} '
        f'--property year={year} '
        f'--property version={version} '
        f'--property source=IPAM '
        f'--property type=annual_embedding ' # <--- TIPO DE ASSET MUDADO
        f'--property time_start={timestamp_start} '
        f'--property time_end={timestamp_end} '
        f'--property create_date={creation_date} '
        f'{gcs_path}'
    )

    # log_message(f"[INFO] Starting upload of EMBEDDING to GEE: {asset_id}")
    status = os.system(upload_command)

    if status == 0:
        # log_message(f"[INFO] Upload completed successfully: {asset_id}")
        return True
    else:
        # log_message(f"[ERROR] Upload failed for GEE asset: {asset_id}")
        return False


# ====================================
# 📈 WORKFLOWS DE PROCESSAMENTO DE EMBEDDINGS
# ====================================

def process_single_image_embedding(dataset_classify, version, region, folder_temp):
    """
    Processa uma única imagem extraindo o embedding DNN e retornando o array HWC.
    (Adaptado de process_single_image em A_3_1 [43-49])
    """
    # Variáveis globais assumidas: country, bucket_name, fs
    
    # 1. Preparação: Download do Modelo (Idêntico ao A_3_1)
    gcs_model_file = f'gs://{bucket_name}/sudamerica/{country}/models_col1/col1_{country}_{version}_{region}_rnn_lstm_ckpt*'
    model_file_local_temp = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt'
    
    # (Lógica de download gsutil - omitida para brevidade)
    try:
        subprocess.run(f'gsutil cp {gcs_model_file} {folder_temp}', shell=True, check=True)
        time.sleep(2)
        fs.invalidate_cache()
    except:
        # log_message(f"[ERROR] Failed to download model from GCS.")
        return None

    # 2. Carregar Hiperparâmetros (Idêntico ao A_3_1)
    json_path = f'{folder_temp}/col1_{country}_{version}_{region}_rnn_lstm_ckpt_hyperparameters.json'
    with open(json_path, 'r') as json_file:
        hyperparameters = json.load(json_file)
    
    # 3. Conversão e Vetorização de Dados (Idêntico ao A_3_1)
    data_classify = convert_to_array(dataset_classify)
    data_classify_vector = reshape_single_vector(data_classify)
    
    # 4. Execução da Extração
    output_data_classified = classify_for_embeddings(data_classify_vector, model_file_local_temp, hyperparameters)

    # 5. Reshape de volta para formato de imagem multi-banda (H, W, C)
    H, W, _ = data_classify.shape
    # O número de canais (C) é a dimensão do embedding (NUM_N_L5)
    C = hyperparameters['NUM_N_L5'] 
    
    # output_image_data agora é multi-banda (H, W, C)
    output_image_data_hwc = output_data_classified.reshape([H, W, C])

    # 6. Retorna o array HWC do embedding (SEM filtro espacial)
    return output_image_data_hwc

def process_year_by_satellite_embedding(satellite_years, bucket_name, folder_mosaic, folder_temp, suffix,
                                        ee_project, country, version, region, simulate_test=False):
    """
    Workflow principal para geração de embeddings, adaptando o nome da coleção GEE.
    (Adaptado de process_year_by_satellite em A_3_1 [49-57])
    """
    
    # Define a nova coleção GEE para embeddings
    collection_name = f'projects/{ee_project}/assets/FIRE/COLLECTION1/CLASSIFICATION_EMBEDDINGS/embedding_field_{country}_{version}'
    check_or_create_collection(collection_name, ee_project)
    
    # (Restante da lógica de loop de anos e satélites, download de COG, e iteração por grid_landsat - idêntica a A_3_1)
    # ... (O download do grid e a iteração inicial)
    
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-{region}')
    grid_landsat = grid.getInfo()['features']
    start_time = time.time()
    
    for satellite_year in satellite_years: 
        satellite = satellite_year['satellite']
        years = satellite_year['years'][:1 if simulate_test else None] 
        
        with tqdm(total=len(years), desc=f'Processing years for satellite {satellite.upper()}') as pbar_years:
            for year in years:
                test_tag = "_test" if simulate_test else ""
                
                # Novo nome do arquivo TIFF para Embeddings
                image_name = f"embedding_{country}_{satellite}_{version}_region{region[1:]}_{year}{suffix}{test_tag}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_embeddings/{image_name}.tif' # Pasta de saída dedicada
                
                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_{region}_{year}_cog.tif'
                
                if not os.path.exists(local_cog_path):
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')
                    time.sleep(2)
                    fs.invalidate_cache()

                input_scenes = []
                grids_to_process = [grid_landsat] if simulate_test else grid_landsat
                
                with tqdm(total=len(grids_to_process), desc=f'Processing scenes for year {year}') as pbar_scenes:
                    for grid_feature in grids_to_process:
                        orbit = grid_feature['properties']['ORBITA']
                        point = grid_feature['properties']['PONTO']
                        output_image_name = f'{folder_temp}/image_emb_{country}_{region}_{version}_{orbit}_{point}_{year}.tif' # Nome TEMP diferente
                        geometry_scene = grid_feature['geometry']
                        NBR_clipped = f'{folder_temp}/image_mosaic_emb_clipped_{orbit}_{point}_{year}.tif' # Nome TEMP diferente

                        if os.path.isfile(output_image_name):
                            pbar_scenes.update(1)
                            continue

                        # 1. Clipagem da Imagem
                        clipping_success = clip_image_by_grid(geometry_scene, local_cog_path, NBR_clipped)

                        if clipping_success:
                            dataset_classify = load_image(NBR_clipped)
                            
                            # 2. Extração de Embeddings (NOVO)
                            image_data_hwc = process_single_image_embedding(dataset_classify, version, region, folder_temp)
                            
                            # 3. Conversão para Raster Multi-Banda (NOVO)
                            convert_to_raster_multiband(dataset_classify, image_data_hwc, output_image_name)

                            input_scenes.append(output_image_name)
                            remove_temporary_files([NBR_clipped])
                        
                        # ... (Atualização da barra de progresso)

                # 4. Geração do TIFF Otimizado Multi-Banda (Merge VRT)
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_temp}/merged_emb_temp_{year}.tif"
                    output_image = f"{folder_temp}/{image_name}.tif"

                    generate_optimized_image(merge_output_temp, output_image, input_scenes_str)

                    # 5. Upload para GCS e GEE
                    status_upload = os.system(f'gsutil cp {output_image} {gcs_filename}')
                    time.sleep(2)
                    fs.invalidate_cache()
                    
                    if status_upload == 0 and os.system(f'gsutil ls {gcs_filename}') == 0:
                        upload_embedding_to_gee( # <--- NOVO UPLOAD
                            gcs_filename,
                            f'{collection_name}/{image_name}',
                            satellite, region, year, version
                        )
                
                clean_directories([folder_temp])
                elapsed = time.time() - start_time
                # log_message(f"[INFO] Year {year} embedding generation completed. Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                pbar_years.update(1)

# ====================================
# 🚀 MAIN EXECUTION LOGIC (render_embedding_models)
# ====================================

def render_embedding_models(models_to_process, simulate_test=False):
    """
    Processa uma lista de modelos e mosaicos para extrair Embeddings.
    (Adaptado de render_classify_models em A_3_1 [57-61])
    """
    # log_message(f"[INFO] [render_embedding_models] STARTING EMBEDDING EXTRACTION {models_to_process}")
    bucket_name = 'mapbiomas-fire'
    
    for model_info in models_to_process:
        # Extrai info do modelo (assumimos que 'country' é global/ambiente)
        model_name = model_info["model"]
        mosaics = model_info["mosaics"]
        simulation = model_info["simulation"]

        parts = model_name.split('_')
        country = parts[12] # Assumindo 'col1_country_vX_rY.meta'
        version = parts[19]
        region = parts[62].split('.') 

        # Define diretórios locais (Idêntico a A_3_1)
        folder = f'/content/mapbiomas-fire/sudamerica/{country}'
        folder_temp = f'{folder}/tmp_emb' # Diretório temporário dedicado para embeddings
        folder_mosaic = f'{folder}/mosaics_cog'

        # Garante que os diretórios existam
        for directory in [folder_temp, folder_mosaic]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        clean_directories([folder_temp, folder_mosaic])

        # Prepara a lista de satélite/ano (Idêntico a A_3_1)
        satellite_years = []
        for mosaic in mosaics:
            mosaic_parts = mosaic.split('_')
            satellite = mosaic_parts
            year = int(mosaic_parts[62])
            satellite_years.append({"satellite": satellite, "years": [year]})

        if simulation:
            # log_message(f"[SIMULATION] Would generate embeddings for model: {model_name}")
            pass
        else:
            # Chama a função de processamento de embeddings
            process_year_by_satellite_embedding(
                satellite_years=satellite_years,
                bucket_name=bucket_name,
                folder_mosaic=folder_mosaic,
                folder_temp=folder_temp,
                suffix='',
                ee_project=f'mapbiomas-{country}',
                country=country,
                version=version,
                region=region,
                simulate_test=simulate_test
            )
    # log_message(f"[INFO] [render_embedding_models] FINISH EMBEDDING EXTRACTION")
