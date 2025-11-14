# BASEADO NA ARQUITETURA DNN (NUM_N_L1 a NUM_N_L5) de A_3_1

def create_embedding_model_graph(hyperparameters):
    """
    Cria um grafo TensorFlow que termina na camada de embedding (hidden5).
    """
    graph = tf.Graph()
    with graph.as_default():
        # Placeholders
        x_input = tf.placeholder(tf.float32, shape=[None, hyperparameters['NUM_INPUT']], name='x_input')
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input') # Mantido por consistência

        # Normalização dos dados
        normalized = (x_input - hyperparameters['data_mean']) / hyperparameters['data_std']

        # Camadas Ocultas (Encoder - Extrator de Features)
        hidden1 = fully_connected_layer(normalized, n_neurons=hyperparameters['NUM_N_L1'], activation='relu', name='h1')
        hidden2 = fully_connected_layer(hidden1, n_neurons=hyperparameters['NUM_N_L2'], activation='relu', name='h2')
        hidden3 = fully_connected_layer(hidden2, n_neurons=hyperparameters['NUM_N_L3'], activation='relu', name='h3')
        hidden4 = fully_connected_layer(hidden3, n_neurons=hyperparameters['NUM_N_L4'], activation='relu', name='h4')
        
        # CAMADA DE EMBEDDING (PONTO DE BIÓPSIA)
        embedding_output = fully_connected_layer(hidden4, n_neurons=hyperparameters['NUM_N_L5'], activation='relu', name='embedding_output') 

        # Define a saída do grafo com o nome do tensor que será extraído
        outputs = embedding_output
        tf.identity(outputs, name='extracted_embedding') # <--- NOVO NOME DO TENSOR

def classify_for_embeddings(data_classify_vector, model_path, hyperparameters, block_size=40000000):
    """
    Extrai embeddings dos dados em blocos, usando o grafo de embedding.
    """
    log_message(f"[INFO] Starting EMBEDDING extraction with model at path: {model_path}")

    num_pixels = data_classify_vector.shape
    num_blocks = (num_pixels + block_size - 1) // block_size 
    output_blocks = [] 

    for i in range(num_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, num_pixels)
        log_message(f"[INFO] Processing block {i+1}/{num_blocks} (pixels {start_idx} to {end_idx})")
        data_block = data_classify_vector[start_idx:end_idx]

        tf.compat.v1.reset_default_graph()

        # 1. Cria o grafo usando a nova função de embedding
        graph, placeholders, saver = create_embedding_model_graph(hyperparameters) 

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.50)

        with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, model_path)

            # 2. Busca o tensor de embedding, NÃO a classe predita
            output_block = sess.run(
                graph.get_tensor_by_name('extracted_embedding:0'), 
                feed_dict={placeholders['x_input']: data_block}
            )
            
            output_blocks.append(output_block)
            
    output_data_classify = np.concatenate(output_blocks, axis=0)
    log_message(f"[INFO] Embedding extraction completed. Shape: {output_data_classify.shape}")
    return output_data_classify

# Nova função para salvar raster multi-banda (Embeddings)
def convert_to_raster_multiband(dataset_classify, embedding_data_scene_hwc, output_image_name):
    log_message(f"[INFO] Converting array to multi-band GeoTIFF raster: {output_image_name}")

    rows, cols, bands = embedding_data_scene_hwc.shape # (H, W, C)
    driver = gdal.GetDriverByName('GTiff')
    
    # Normalização de 0 a 255 (Quantização 8-bit)
    min_vals = np.min(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    max_vals = np.max(embedding_data_scene_hwc, axis=(0, 1), keepdims=True)
    
    # Evita divisão por zero para canais constantes (onde min==max)
    range_vals = np.where(max_vals == min_vals, 1e-6, max_vals - min_vals)
    
    # Normaliza para 0-255 e converte para uint8 (8-bit)
    normalized_data = 255 * (embedding_data_scene_hwc - min_vals) / range_vals
    embedding_data_scene_uint8 = normalized_data.astype('uint8')
    
    # Transpõe para o formato (C, H, W) que o GDAL espera
    embedding_data_chw = np.transpose(embedding_data_scene_uint8, [10, 11])

    # Opções de criação (compressão LZW recomendada para GeoTIFF multi-banda)
    options = [
        'COMPRESS=LZW', # LZW ou DEFLATE
        'PREDICTOR=2',
        'TILED=YES',
        'BIGTIFF=YES'
    ]

    # Cria o dataset de saída com o número correto de bandas (bands)
    outDs = driver.Create(output_image_name, cols, rows, bands, gdal.GDT_Byte, options=options)

    # Escreve cada banda do embedding
    for i in range(bands):
        outDs.GetRasterBand(i + 1).WriteArray(embedding_data_chw[i])

    outDs.SetGeoTransform(dataset_classify.GetGeoTransform())
    outDs.SetProjection(dataset_classify.GetProjection())
    outDs.FlushCache()
    outDs = None 

    log_message(f"[INFO] Multi-band raster (Embeddings) saved: {output_image_name}")

def process_single_image_embedding(dataset_classify, version, region, folder_temp, model_file_local_temp, hyperparameters):
    """
    Processa uma única cena para extrair embeddings.
    """
    # 1. Carrega e vetoriza os dados
    data_classify = convert_to_array(dataset_classify)
    data_classify_vector = reshape_single_vector(data_classify)

    # 2. Executa a extração (classify_for_embeddings)
    output_data_classified = classify_for_embeddings(data_classify_vector, model_file_local_temp, hyperparameters)

    # 3. Reshape de volta para formato de imagem (H, W, C)
    # C será NUM_N_L5 (dimensão do embedding)
    H, W, _ = data_classify.shape
    C = output_data_classified.shape[11]
    
    # output_image_data agora é multi-banda (H, W, C)
    output_image_data_hwc = output_data_classified.reshape([H, W, C])

    # 4. Retorna a imagem de embedding (NÃO aplica filtro espacial)
    return output_image_data_hwc

        saver = tf.train.Saver()
        return graph, {'x_input': x_input, 'y_input': y_input}, saver

