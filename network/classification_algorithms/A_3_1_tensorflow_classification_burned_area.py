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

def render_classify(dataset_classify, model_path, country, region, version, year):
    """
    Processa a classificação de uma imagem com base no modelo e gera a saída final.

    :param dataset_classify: Dataset GDAL da imagem a ser classificada
    :param model_path: Caminho do modelo a ser usado para classificação
    :param country: País (ex: "guyana")
    :param region: Região (ex: "region1")
    :param version: Versão do modelo (ex: "v1")
    :param year: Ano da imagem a ser classificada
    :return: Imagem classificada após a aplicação do modelo e filtragem espacial
    """
    # 1. Converter a imagem para um array numpy
    data_classify = convert_to_array(dataset_classify)
    
    # 2. Transformar a imagem em um vetor 1D de pixels
    data_classify_vector = reshape_single_vector(data_classify)
    
    # 3. Carregar e usar o modelo TensorFlow para realizar a classificação
    output_data_classified = classify(data_classify_vector, model_path, country, region, version)
    
    # 4. Remodelar o vetor classificado de volta para o formato de imagem
    output_image_data = reshape_image_output(output_data_classified, data_classify)
    
    # 5. Aplicar um filtro espacial para remover ruídos e regiões pequenas
    output_image_filtered = filter_spatial(output_image_data)
    
    return output_image_filtered

def classify(data_classify_vector, model_path, country, region, version):
    """
    Realiza a classificação dos dados utilizando um modelo TensorFlow.
    
    :param data_classify_vector: Vetor 1D com os dados de entrada (pixels da imagem)
    :param model_path: Caminho para o modelo TensorFlow salvo
    :param country: País
    :param region: Região (ex: "region1")
    :param version: Versão do modelo
    :return: Vetor de saída com os dados classificados
    """
    # Limitar a fração de memória da GPU usada (ajustar conforme a sua GPU)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.50)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    
    # Carregar o gráfico e o modelo salvo
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph(f'{model_path}/col1_{country}_v{version}_r{region}_rnn_lstm_ckpt.meta')
        saver.restore(sess, tf.compat.v1.train.latest_checkpoint(model_path))
        
        graph = tf.compat.v1.get_default_graph()
        x_input = graph.get_tensor_by_name('x_input:0')  # Ajustar conforme a arquitetura do modelo
        outputs = graph.get_tensor_by_name('output_layer:0')  # Ajustar conforme a arquitetura do modelo
        
        # Dividir os dados em blocos para evitar problemas de memória
        block_size = 4000000  # Ajuste o tamanho do bloco conforme necessário
        output_data_classified = []
        
        for i in range(0, len(data_classify_vector), block_size):
            block = data_classify_vector[i:i + block_size]
            output_block = sess.run(outputs, feed_dict={x_input: block})
            output_data_classified.append(output_block)
        
        # Concatenar os blocos classificados
        output_data_classified = np.concatenate(output_data_classified, axis=0)
    
    # Limpar a sessão do TensorFlow
    tf.keras.backend.clear_session()
    
    return output_data_classified

def convert_to_array(dataset_classify):
    """
    Converte o dataset GDAL da imagem classificada em um array NumPy.
    
    :param dataset_classify: Dataset GDAL da imagem
    :return: Array NumPy contendo os dados da imagem
    """
    cols = dataset_classify.RasterXSize
    rows = dataset_classify.RasterYSize
    bands = dataset_classify.RasterCount
    
    array_data = np.zeros((rows, cols, bands), dtype=np.float32)
    for i in range(1, bands + 1):
        array_data[:, :, i - 1] = dataset_classify.GetRasterBand(i).ReadAsArray()
    
    return array_data

def reshape_image_output(output_data_classified, data_classify):
    """
    Reorganiza os dados classificados em um formato de imagem 2D.
    
    :param output_data_classified: Vetor com os dados classificados
    :param data_classify: Imagem original (array 2D ou 3D)
    :return: Imagem classificada no formato 2D
    """
    return output_data_classified.reshape([data_classify.shape[0], data_classify.shape[1]])


def filter_spatial(output_image_data):
    """
    Aplica um filtro espacial para remover pequenas áreas na imagem classificada.
    
    :param output_image_data: Imagem classificada (binária)
    :return: Imagem filtrada
    """
    binary_image = output_image_data > 0
    open_image = ndimage.binary_opening(binary_image, structure=np.ones((4, 4)))  # Remove pequenos clusters
    close_image = ndimage.binary_closing(open_image, structure=np.ones((8, 8)))  # Preenche pequenos buracos
    return close_image
