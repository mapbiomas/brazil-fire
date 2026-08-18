#!/usr/bin/env python
# coding: utf-8

version = 2
biome = 'caatinga' #['pantanal', 'pampa',  'caatinga', 'cerrado', 'mata_atlantica'] 
#region = '6'
folder = '../../dados'# pasta principal onde fica os dados
folder_amostras = f'../../../mnt/Files-Geo/Arquivos/amostras_col4/{biome}_r{region}'
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_col5'
images_train_test = [
    #f'train_test_fire_nbr_{biome}_r{region}_l5_v{version}_*.tif',
    #f'train_test_fire_nbr_{biome}_r{region}_l57_v{version}_*.tif',
    f'train_test_fire_nbr_{biome}_r{region}_l78_v{version}_*.tif',
    f'train_test_fire_nbr_{biome}_r{region}_l89_v{version}_*.tif',
    f'train_test_fire_nbr_{biome}_r{region}_l89_v3_*.tif',
    #f'train_test_fire_nbr_{biome}_r{region}_l89_v2_*.tif',

]

def load_image(image):
    return gdal.Open(image, gdal.GA_ReadOnly)
    
def convert_to_array(dataset):
    nbr = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(nbr, 2)

# ### IMAGENS DE TREINO E TESTE ###
all_data_train_test_vector = []
for index, images in enumerate(images_train_test):
    images_name = glob.glob(f'{folder_amostras}/{images}')

    for image in images_name:
        #file = '{0}'.format(image)
        dataset_train_test = load_image(image)
        data_train_test = convert_to_array(dataset_train_test)

        # Reshape nos dados de treino e teste
        vector = data_train_test.reshape([data_train_test.shape[0] * data_train_test.shape[1], data_train_test.shape[2]])
        dataclean = vector[~np.isnan(vector).any(axis=1)]
        all_data_train_test_vector.append(dataclean)

# ### PREPARAR DADOS PARA O MODELO ###

# Para selecionar apenas os dados válidos e embaralhar - dados de treino e teste: 
def filter_valid_data_and_shuffle(data_train_test_vector): 
    #data = data_train_test_vector[~np.isnan(data_train_test_vector).any(axis=1)]
    np.random.shuffle(data_train_test_vector)
    return data_train_test_vector

# Concatena os dados de treino e teste
data_train_test_vector = np.concatenate(all_data_train_test_vector)

# seleciona apenas os dados válidos e embaralha 
valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)

bi = [0,1,2,3] # index of nbr
li = 4 # index of label

# SEPARA OS DADOS ENTRE TREINO E VALIDAÇÃO
TRAIN_FRACTION = 0.7

training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)
training_data= valid_data_train_test[0:training_size,:]
validation_data = valid_data_train_test[training_size:-1,:]

# Compute per-band means and standard deviations of the input nbr
data_mean = training_data[:,bi].mean(0)
data_std = training_data[:,bi].std(0)

# HIPER-PARÂMETROS

lr = 0.001 # learning rate
BATCH_SIZE = 1000
N_ITER = 7000
NUM_INPUT = len(bi)
NUM_N_L1 = 7
NUM_N_L2 = 14
NUM_N_L3 = 7
NUM_N_L4 = 14
NUM_N_L5 = 7
"""NUM_N_L6 = 14
NUM_N_L7 = 7
NUM_N_L8 = 28
NUM_N_L9 = 7
NUM_N_L10 = 14
NUM_N_L11 = 128
NUM_N_L12 = 64
NUM_N_L13 = 128
NUM_N_L14 = 64
NUM_N_L15 = 128"""

NUM_CLASSES = 2

# CONSTRUÇÃO DO MODELO

def fully_connected_layer(input, n_neurons, activation=None):
    
    # variáveis da camada
    input_size = input.get_shape().as_list()[1]
    W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))))
    b = tf.Variable(tf.zeros([n_neurons]))

    # operação linear
    layer = tf.matmul(input, W) + b
    
    # aplica a não linearidade
    if activation == 'relu':
        layer = tf.nn.relu(layer)
    
    return layer


graph = tf.Graph() # cria um grafo
with graph.as_default(): # abre o grafo para adicionar os nós
    
    # camdas de inputs
    x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT])
    y_input = tf.placeholder(tf.int64, shape=[None])

    normalized = (x_input - data_mean) / data_std
    hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
    hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
    hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
    hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
    hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')
    """hidden6 = fully_connected_layer(hidden5, n_neurons=NUM_N_L6, activation='relu')
    hidden7 = fully_connected_layer(hidden6, n_neurons=NUM_N_L7, activation='relu')
    hidden8 = fully_connected_layer(hidden7, n_neurons=NUM_N_L8, activation='relu')
    hidden9 = fully_connected_layer(hidden8, n_neurons=NUM_N_L9, activation='relu')
    hidden10 = fully_connected_layer(hidden9, n_neurons=NUM_N_L10, activation='relu')
    hidden11 = fully_connected_layer(hidden10, n_neurons=NUM_N_L11, activation='relu')
    hidden12 = fully_connected_layer(hidden11, n_neurons=NUM_N_L12, activation='relu')
    hidden13 = fully_connected_layer(hidden12, n_neurons=NUM_N_L13, activation='relu')
    hidden14 = fully_connected_layer(hidden13, n_neurons=NUM_N_L14, activation='relu')
    hidden15 = fully_connected_layer(hidden14, n_neurons=NUM_N_L15, activation='relu')"""
    
    logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input, name='error'))
    
    # otimizador
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    outputs = tf.argmax(logits, 1)

    correct_prediction = tf.equal(outputs, y_input)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # inicializador
    init = tf.global_variables_initializer()
    
    # para salvar o modelo de treino
    saver = tf.train.Saver()

# EXECUÇÃO DO MODELO
start_time = time.time()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
  
    validation_dict = {
        x_input: validation_data[:,bi],
        y_input: validation_data[:,li]
    }

    for i in range(N_ITER + 1):
        batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]
        # cria o feed_dict de treino
        feed_dict = {
          x_input: batch[:,bi], 
          y_input: batch[:,li]
        }
        # executa a interação de treino
        optimizer.run(feed_dict=feed_dict)

        if i % 100 == 0:
            # calcula a acurácia
            acc = accuracy.eval(validation_dict) * 100
            
            # salva as variávies do modelo
            model_path = f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt'
            saver.save(sess, model_path)
            print('Accuracy %.2f%% at step %s' %(acc, i))

end_time = time.time()
training_time = end_time - start_time
print(colored('Spent time: {0}'.format(time.strftime("%H:%M:%S", time.gmtime(training_time))), 'yellow'))
print(colored(f'Modelo salvo em: {model_path}','green'))     