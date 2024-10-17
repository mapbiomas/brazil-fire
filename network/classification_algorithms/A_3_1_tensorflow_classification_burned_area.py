import os
import numpy as np
import tensorflow as tf
from scipy import ndimage
from osgeo import gdal
import rasterio
from rasterio.mask import mask
import ee  # Para integração com o Google Earth Engine
from tqdm import tqdm  # Para barras de progresso
import time
import math
from shapely.geometry import shape, box, mapping
import shutil  # Para operações de arquivos e pastas

# 6.1 Funções para a predição de área queimada utilizando o modelo disponível
# ---------------------------
# Funções para Processamento
# ---------------------------

# Função para transformar os dados classificados em um vetor único de pixels
def reshape_single_vector(data_classify):
    """
    Transforma os dados de uma imagem classificada em um vetor 1D de pixels.

    :param data_classify: Imagem classificada (array 2D)
    :return: Vetor 1D contendo os pixels da imagem
    """
    return data_classify.reshape([data_classify.shape[0] * data_classify.shape[1], data_classify.shape[2]])

# Função para realizar a classificação utilizando o modelo TensorFlow
def classify(data_classify_vector):
    """
    Executa a classificação dos dados utilizando um modelo de rede neural.
    O processamento é feito em blocos para evitar problemas de memória.

    :param data_classify_vector: Vetor com os dados de entrada (pixels)
    :return: Dados classificados
    """
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)  # Limita a fração de memória da GPU utilizada
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # Restaura o modelo treinado
        saver.restore(sess, f'{folder_model}/col1_{country}_v{version}_r{region}_rnn_lstm_ckpt')

        # Classificar os dados em blocos de tamanho máximo de 4.000.000 pixels
        output_data_classify0 = outputs.eval({x_input: data_classify_vector[:4000000, bi]})
        output_data_classify1 = outputs.eval({x_input: data_classify_vector[4000000:8000000, bi]})
        output_data_classify2 = outputs.eval({x_input: data_classify_vector[8000000:12000000, bi]})
        output_data_classify3 = outputs.eval({x_input: data_classify_vector[12000000:, bi]})

        # Concatenar todos os blocos classificados
        output_data_classify = np.concatenate([
            output_data_classify0,
            output_data_classify1,
            output_data_classify2,
            output_data_classify3
        ])

    # Limpa a sessão do TensorFlow para liberar memória
    tf.keras.backend.clear_session()
    return output_data_classify

# Função para remodelar os dados classificados de volta para o formato de imagem
def reshape_image_output(output_data_classified, data_classify):
    """
    Reorganiza os dados classificados em um formato de imagem 2D.

    :param output_data_classified: Vetor com os dados classificados
    :param data_classify: Imagem original
    :return: Imagem classificada no formato 2D
    """
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])

# Função para aplicar filtro espacial na imagem classificada
def filter_spatial(output_image_data):
    """
    Aplica um filtro espacial para remover regiões pequenas da imagem classificada.

    :param output_image_data: Imagem classificada em formato binário
    :return: Imagem com filtragem espacial aplicada
    """
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))  # Remove pequenas regiões brancas
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))  # Remove pequenos buracos pretos
    return close_image

# Função para converter um array NumPy de volta em um raster GeoTIFF
def convert_to_raster(dataset_classify, image_data_scene, output_image_name):
    """
    Converte um array NumPy em um arquivo raster GeoTIFF, aplicando a transformação geoespacial correta.

    :param dataset_classify: Dataset GDAL com os metadados da imagem original
    :param image_data_scene: Dados da imagem classificados (NumPy array)
    :param output_image_name: Nome do arquivo de saída
    """
    cols, rows = dataset_classify.RasterXSize, dataset_classify.RasterYSize
    driver = gdal.GetDriverByName('GTiff')
    outDs = driver.Create(output_image_name, cols, rows, 1, gdal.GDT_Float32)
    outDs.GetRasterBand(1).WriteArray(image_data_scene)

    # Aplica a GeoTransform e a Projeção do dataset original
    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None  # Libera o dataset de saída da memória

# Função para processar a classificação da imagem
def render_classify(dataset_classify):
    """
    Processa uma imagem classificada, aplicando filtragem espacial e gerando o resultado.

    :param dataset_classify: Dataset GDAL da imagem a ser classificada
    :return: Imagem classificada após o filtro espacial
    """
    data_classify = convert_to_array(dataset_classify)
    data_classify_vector = reshape_single_vector(data_classify)
    output_data_classified = classify(data_classify_vector)
    output_image_data = reshape_image_output(output_data_classified, data_classify)
    return filter_spatial(output_image_data)

# Função para ler a grade Landsat a partir do Google Earth Engine
def read_grid_landsat():
    """
    Lê a grade Landsat de uma coleção auxiliar no Google Earth Engine.

    :return: GeoInformações sobre as regiões da grade Landsat
    """
    grid = ee.FeatureCollection(f'projects/mapbiomas-{country}/assets/FIRE/AUXILIARY_DATA/GRID_REGIONS/grid-{country}-r{region}')
    return grid.getInfo()['features']

# Função para converter metros em graus com base na latitude
def meters_to_degrees(meters, latitude):
    """
    Converte uma distância em metros para graus, levando em consideração a latitude.

    :param meters: Distância em metros
    :param latitude: Latitude em graus
    :return: Distância em graus
    """
    return meters / (111320 * abs(math.cos(math.radians(latitude))))

# Função para expandir a geometria com um buffer em metros
def expand_geometry(geometry, buffer_distance_meters=50):
    """
    Expande a geometria aplicando um buffer em metros.

    :param geometry: Geometria original
    :param buffer_distance_meters: Distância do buffer em metros
    :return: Geometria expandida
    """
    geom = shape(geometry)
    centroid_lat = geom.centroid.y
    buffer_distance_degrees = meters_to_degrees(buffer_distance_meters, centroid_lat)
    expanded_geom = geom.buffer(buffer_distance_degrees)
    return mapping(expanded_geom)

# Função para verificar se há uma interseção significativa entre geometria e a imagem
def has_significant_intersection(geom, image_bounds, min_intersection_area=0.01):
    """
    Verifica se há uma interseção significativa entre a geometria e os limites da imagem.

    :param geom: Geometria da cena
    :param image_bounds: Limites da imagem
    :param min_intersection_area: Área mínima para considerar uma interseção
    :return: True se a interseção for significativa, False caso contrário
    """
    geom_shape = shape(geom)
    image_shape = box(*image_bounds)
    intersection = geom_shape.intersection(image_shape)
    return intersection.area >= min_intersection_area

# Função para recortar a imagem pela geometria fornecida
def clip_image_by_grid(geom, image, output, buffer_distance_meters=100):
    """
    Recorta a imagem com base em uma geometria e salva o resultado.

    :param geom: Geometria de recorte
    :param image: Caminho para a imagem de entrada
    :param output: Caminho para a imagem de saída
    :param buffer_distance_meters: Distância do buffer para a geometria
    """
    with rasterio.open(image) as src:
        expanded_geom = expand_geometry(geom, buffer_distance_meters)

        try:
            # Verificar se há interseção suficiente para recortar a imagem
            if has_significant_intersection(expanded_geom, src.bounds):
                out_image, out_transform = mask(src, [expanded_geom], crop=True, nodata=np.nan, filled=True)
            else:
                print(f'Skipping image: {image} - Insufficient overlap with raster.')
                return
        except ValueError as e:
            print(f'Skipping image: {image} - {str(e)}')
            return

    # Atualiza os metadados e salva o arquivo raster de saída
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})

    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)

# Função para construir o VRT e traduzir usando gdal_translate
def gerar_imagem_otimizada(name_out_vrt, name_out_tif, files_tif_str):
    """
    Gera um VRT a partir de múltiplos arquivos TIFF e converte para um TIFF otimizado e comprimido usando gdal_translate.

    Parâmetros:
    name_out_vrt: Caminho para o arquivo de saída VRT.
    name_out_tif: Caminho para o arquivo TIFF de saída otimizado.
    files_tif_str: String contendo os caminhos dos arquivos TIFF que serão processados.
    """
    # Primeiro: criar o VRT a partir de várias cenas
    print(f'[INFO] Construindo o VRT: {name_out_vrt}')
    os.system(f'gdalbuildvrt {name_out_vrt} {files_tif_str}')

    # Segundo: traduzir o VRT para um TIFF otimizado e comprimido
    print(f'[INFO] Traduzindo para TIFF otimizado: {name_out_tif}')
    os.system(f'gdal_translate -a_nodata 0 -co TILED=YES -co compress=DEFLATE -co PREDICTOR=2 -co COPY_SRC_OVERVIEWS=YES -co BIGTIFF=YES {name_out_vrt} {name_out_tif}')

    print(f'[INFO] Tradução finalizada. Imagem otimizada salva em {name_out_tif}')

import os
import shutil
import datetime

# Função para limpar as pastas antes do início do processamento
def limpar_pastas(pastas_para_limpar):
    for pasta in pastas_para_limpar:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)  # Remove o diretório e todo o seu conteúdo
            os.makedirs(pasta)  # Recria o diretório vazio
            print(f'[INFO] Pasta limpa: {pasta}')
        else:
            os.makedirs(pasta)
            print(f'[INFO] Pasta criada: {pasta}')

# Função para construir o VRT e traduzir usando gdal_translate
def gerar_imagem_otimizada(name_out_vrt, name_out_tif, files_tif_str):
    os.system(f'gdalbuildvrt {name_out_vrt} {files_tif_str}')
    os.system(f'gdal_translate -a_nodata 0 -co compress=DEFLATE {name_out_vrt} {name_out_tif}')

# Função para verificar e criar a coleção no GEE
def verificar_ou_criar_colecao(colecao, eeProject):
    check_command = f'earthengine --project {eeProject} asset info {colecao}'
    status = os.system(check_command)

    if status != 0:
        print(f'[INFO] Criando nova coleção no GEE: {colecao}')
        create_command = f'earthengine --project {eeProject} create collection {colecao}'
        os.system(create_command)
    else:
        print(f'[INFO] Coleção já existe: {colecao}')

# Função para realizar o upload de um arquivo para o GEE com metadados e verificar se o asset já existe
def upload_para_gee(gcs_path, asset_id, satellite, region, year, version, eeProject):
    timestamp_start = int(datetime.datetime(year, 1, 1).timestamp() * 1000)
    timestamp_end = int(datetime.datetime(year, 12, 31).timestamp() * 1000)
    creation_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Verificar se o asset já existe no GEE
    check_asset_command = f'earthengine --project {eeProject} asset info {asset_id}'
    asset_status = os.system(check_asset_command)

    if asset_status == 0:
        print(f'[INFO] Asset já existe, pulando upload: {asset_id}')
    else:
        upload_command = (
            f'earthengine --project {eeProject} upload image --asset_id={asset_id} '
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

        print(f'[INFO] Iniciando upload para o GEE: {asset_id}')
        status = os.system(upload_command)

        if status == 0:
            print(f'[INFO] Upload concluído com sucesso: {asset_id}')
        else:
            print(f'[ERRO] Falha no upload para o GEE: {asset_id}')
            print(f'[ERRO] Status do comando: {status}')

# Função para remover arquivos temporários
def remover_arquivos_temporarios(arquivos_para_remover):
    for arquivo in arquivos_para_remover:
        if os.path.exists(arquivo):
            try:
                os.remove(arquivo)
                print(f'[INFO] Arquivo temporário removido: {arquivo}')
            except Exception as e:
                print(f'[ERRO] Falha ao remover arquivo: {arquivo}. Detalhes: {str(e)}')

# Função principal de processamento
def processar_ano_por_satelite(satellite_years, bucket_name, folder_mosaic, folder_images, sulfix, eeProject, country, version):
    grid_landsat = read_grid_landsat()  # Carregar a grade Landsat
    start_time = time.time()

    collection_name = f'projects/{eeProject}/assets/FIRE/COLLECTION1/CLASSIFICATION/burned_area_{country}_v{version}'
    verificar_ou_criar_colecao(collection_name, eeProject)  # Verificar ou criar a coleção no GEE

    # Loop principal para processar satélites e anos
    for satellite_year in satellite_years:
        satellite = satellite_year['satellite']
        print(f"\n{'='*60}")
        print(f'[INFO] Iniciando o processamento para o satélite: {satellite.upper()}')
        print(f"{'='*60}")

        years = satellite_year['years']

        # Barra de progresso para o processamento dos anos
        with tqdm(total=len(years), desc=f'Processando anos para o satélite {satellite.upper()}') as pbar_anos:
            for year in years:
                print(f"\n{'-'*60}")
                print(f'[INFO] Iniciando o processamento do ano {year} para o satélite {satellite.upper()}...')
                print(f"{'-'*60}")

                image_name = f"burned_area_{country}_{satellite}_v{version}_region{region}_{year}{sulfix}"
                gcs_filename = f'gs://{bucket_name}/sudamerica/{country}/result_classified/{image_name}.tif'

                local_cog_path = f'{folder_mosaic}/{satellite}_{country}_r{region}_{year}_cog.tif'
                gcs_cog_path = f'gs://{bucket_name}/sudamerica/{country}/mosaics_col1_cog/{satellite}_{country}_r{region}_{year}_cog.tif'

                if os.path.exists(local_cog_path):
                    print(f'[INFO] Arquivo COG já copiado localmente: {local_cog_path}')
                else:
                    print(f'[INFO] Copiando arquivo COG do GCS para o diretório local...')
                    os.system(f'gsutil cp {gcs_cog_path} {local_cog_path}')

                input_scenes = []
                total_scenes_done = 0

                # Barra de progresso para o processamento das cenas dentro de um ano
                with tqdm(total=len(grid_landsat), desc=f'Processando cenas para o ano {year}') as pbar_cenas:
                    for grid in grid_landsat:
                        orbit = grid['properties']['ORBITA']
                        point = grid['properties']['PONTO']
                        output_image_name = f'{folder_images}/image_col3_{country}_r{region}_v{version}_{orbit}_{point}_{year}.tif'

                        if os.path.isfile(output_image_name):
                            print(f'[INFO] Arquivo já existente: {output_image_name}. Pulando...')
                            pbar_cenas.update(1)  # Atualiza a barra de progresso
                            continue

                        geometry_cena = grid['geometry']
                        NBR_clipped = f'{folder_images}/image_mosaic_col3_{country}_r{region}_v{version}_{orbit}_{point}_clipped_{year}.tif'

                        try:
                            clip_image_by_grid(geometry_cena, local_cog_path, NBR_clipped)
                            dataset_classify = load_image(NBR_clipped)
                            image_data = render_classify(dataset_classify)  # Aqui deve chamar o modelo
                            convert_to_raster(dataset_classify, image_data, output_image_name)
                            input_scenes.append(output_image_name)
                            total_scenes_done += 1

                            print(f'[PROGRESSO] {total_scenes_done}/{len(grid_landsat)} cenas processadas.')
                            pbar_cenas.update(1)  # Atualiza a barra de progresso
                        except Exception as e:
                            print(f'[ERRO] Falha ao processar cena {orbit}/{point}. Detalhes: {str(e)}')
                            pbar_cenas.update(1)  # Mesmo em caso de erro, a barra de progresso é atualizada
                            continue

                # Realizar o merge das cenas
                if input_scenes:
                    input_scenes_str = " ".join(input_scenes)
                    merge_output_temp = f"{folder_images}/merged_temp_{year}.tif"
                    output_image = f"{folder_images}/{image_name}.tif"

                    try:
                        gerar_imagem_otimizada(merge_output_temp, output_image, input_scenes_str)
                        print(f'[INFO] Merge de todas as {len(input_scenes)} cenas concluído para o ano {year}.')

                        # Realizar o upload para o GCS
                        upload_status = os.system(f'gsutil cp {output_image} {gcs_filename}')
                        if upload_status == 0:
                            print(f'[INFO] Upload para o GCS bem-sucedido: {gcs_filename}')
                        else:
                            print(f'[ERRO] Falha ao fazer o upload do arquivo: {output_image}')
                            continue

                        # Upload para o Google Earth Engine (GEE) dentro da coleção
                        outputAssetID = f'{collection_name}/{image_name}'
                        print(f'[INFO] Fazendo upload para o GEE: {outputAssetID}')
                        upload_para_gee(gcs_filename, outputAssetID, satellite, region, year, version, eeProject)

                    except Exception as e:
                        print(f'[ERRO] Falha ao realizar o merge das cenas para o ano {year}. Detalhes: {str(e)}')
                        continue

                # Limpar arquivos temporários após o subloop de cada ano
                arquivos_temporarios = [local_cog_path, merge_output_temp] + input_scenes
                remover_arquivos_temporarios(arquivos_temporarios)

                # Limpar a pasta tmp1 após o processamento de cada ano
                limpar_pastas([folder_images])

                elapsed_time = time.time() - start_time
                print(f'[INFO] Tempo total gasto até agora para o ano {year}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

                # Atualiza a barra de progresso do processamento do ano
                pbar_anos.update(1)

        print(f"\n{'='*60}")
        print(f'[INFO] Processamento do satélite {satellite.upper()} concluído.')
        print(f"{'='*60}")

    print('[INFO] Processamento completo.')

def render_classify(satellite_years, model_path, country, region, version):
    # Definir bucket e projeto GEE
    bucketName = 'mapbiomas-fire'
    eeProject = f'mapbiomas-{country}'
    
    # Definir diretórios e garantir que existam
    folder = f'/content/mapbiomas-fire/sudamerica/{country}'
    folder_model = f'{folder}/models_col1'
    folder_images = f'{folder}/tmp1'  # Diretório para armazenamento temporário de imagens
    folder_mosaic = f'{folder}/mosaics_cog'  # Diretório para arquivos COG (Cloud-Optimized GeoTIFF)

    # Garantir que as pastas existam
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)
    
    # Limpar pastas de imagens e mosaicos
    limpar_pastas([folder_images, folder_mosaic])

    # Chamar a função principal de processamento
    processar_ano_por_satelite(
        satellite_years=satellite_years,
        bucket_name=bucketName,
        folder_mosaic=folder_mosaic,
        folder_images=folder_images,
        sulfix='',  # Se precisar adicionar algum sufixo especial
        eeProject=eeProject,
        country=country,
        version=version
    )



