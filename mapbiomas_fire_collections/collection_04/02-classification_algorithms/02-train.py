#!/usr/bin/env python
# coding: utf-8

"""
MapBiomas Fire – Neural Network Training
---------------------------------------

This script trains a neural network model for burned area detection
using NBR-based samples extracted from Landsat imagery.

The workflow includes:

1. Loading training and validation samples
2. Converting raster samples into tabular vectors
3. Filtering and shuffling valid observations
4. Splitting data into training and validation sets
5. Training a fully connected neural network (TensorFlow 1.x)
6. Saving model checkpoints during training

Inputs
------
Raster sample files (.tif) containing:
    - NBR bands (features)
    - Label band (burned / unburned)

Outputs
-------
TensorFlow checkpoint model files saved to:

    folder_modelo/

Notes
-----
This script uses TensorFlow 1.x compatibility mode.
"""

# =====================================================================
# Configuration
# =====================================================================

version = 1

# Available biomes:
# ['pantanal', 'pampa', 'caatinga', 'cerrado', 'mata_atlantica']
biome = 'pantanal'

region = '1' # Available regions: # ['1', '2', '3', '4', '5', '6', '7']

# Root data directory
folder = '../../dados'

# Folder containing training samples
folder_amostras = f'../../../mnt/Files-Geo/Arquivos/amostras_col4/{biome}_r{region}'

# Folder where trained models will be saved
folder_modelo = '../../../mnt/Files-Geo/Arquivos/modelos_col4'


# Training / validation image patterns
images_train_test = [

    # f'train_test_fire_nbr_{biome}_r{region}_l5_v{version}_*.tif',
    # f'train_test_fire_nbr_{biome}_r{region}_l57_v{version}_*.tif',

    f'train_test_fire_nbr_{biome}_r{region}_l78_v{version}_*.tif',
    f'train_test_fire_nbr_{biome}_r{region}_l89_v{version}_*.tif',

    # f'train_test_fire_nbr_{biome}_r{region}_l78_v1_*.tif',
    # f'train_test_fire_nbr_{biome}_r{region}_l89_v2_*.tif',
]


# =====================================================================
# Helper Functions
# =====================================================================

def load_image(image):
    """
    Load raster image using GDAL.
    """
    return gdal.Open(image, gdal.GA_ReadOnly)


def convert_to_array(dataset):
    """
    Convert GDAL dataset to a NumPy array.

    Returns
    -------
    array : numpy.ndarray
        Shape = (rows, cols, bands)
    """
    nbr = [dataset.GetRasterBand(i + 1).ReadAsArray()
           for i in range(dataset.RasterCount)]

    return np.stack(nbr, 2)


# =====================================================================
# Load Training and Validation Samples
# =====================================================================

all_data_train_test_vector = []

for index, images in enumerate(images_train_test):

    images_name = glob.glob(f'{folder_amostras}/{images}')

    for image in images_name:

        dataset_train_test = load_image(image)
        data_train_test = convert_to_array(dataset_train_test)

        # Reshape raster to pixel vector
        vector = data_train_test.reshape(
            [data_train_test.shape[0] * data_train_test.shape[1],
             data_train_test.shape[2]]
        )

        # Remove pixels containing NaN values
        dataclean = vector[~np.isnan(vector).any(axis=1)]

        all_data_train_test_vector.append(dataclean)


# =====================================================================
# Prepare Data for Training
# =====================================================================

def filter_valid_data_and_shuffle(data_train_test_vector):
    """
    Shuffle training and validation samples.
    """
    np.random.shuffle(data_train_test_vector)
    return data_train_test_vector


# Concatenate all samples
data_train_test_vector = np.concatenate(all_data_train_test_vector)

# Shuffle dataset
valid_data_train_test = filter_valid_data_and_shuffle(data_train_test_vector)


# Feature indices (NBR bands)
bi = [0, 1, 2, 3]

# Label index
li = 4


# =====================================================================
# Train / Validation Split
# =====================================================================

TRAIN_FRACTION = 0.7

training_size = int(valid_data_train_test.shape[0] * TRAIN_FRACTION)

training_data = valid_data_train_test[0:training_size, :]
validation_data = valid_data_train_test[training_size:-1, :]


# Compute normalization statistics
data_mean = training_data[:, bi].mean(0)
data_std = training_data[:, bi].std(0)


# =====================================================================
# Hyperparameters
# =====================================================================

lr = 0.001
BATCH_SIZE = 1000
N_ITER = 7000

NUM_INPUT = len(bi)

NUM_N_L1 = 7
NUM_N_L2 = 14
NUM_N_L3 = 7
NUM_N_L4 = 14
NUM_N_L5 = 7

NUM_CLASSES = 2


# =====================================================================
# Neural Network Architecture
# =====================================================================

def fully_connected_layer(input, n_neurons, activation=None):
    """
    Build a fully connected neural network layer.
    """

    input_size = input.get_shape().as_list()[1]

    W = tf.Variable(
        tf.truncated_normal(
            [input_size, n_neurons],
            stddev=1.0 / math.sqrt(float(input_size))
        )
    )

    b = tf.Variable(tf.zeros([n_neurons]))

    layer = tf.matmul(input, W) + b

    if activation == 'relu':
        layer = tf.nn.relu(layer)

    return layer


# =====================================================================
# TensorFlow Graph Definition
# =====================================================================

graph = tf.Graph()

with graph.as_default():

    # Inputs
    x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT])
    y_input = tf.placeholder(tf.int64, shape=[None])

    # Normalize inputs
    normalized = (x_input - data_mean) / data_std

    # Hidden layers
    hidden1 = fully_connected_layer(normalized, NUM_N_L1, 'relu')
    hidden2 = fully_connected_layer(hidden1, NUM_N_L2, 'relu')
    hidden3 = fully_connected_layer(hidden2, NUM_N_L3, 'relu')
    hidden4 = fully_connected_layer(hidden3, NUM_N_L4, 'relu')
    hidden5 = fully_connected_layer(hidden4, NUM_N_L5, 'relu')

    # Output layer
    logits = fully_connected_layer(hidden5, NUM_CLASSES)

    # Loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y_input,
            name='error'
        )
    )

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # Predictions
    outputs = tf.argmax(logits, 1)

    correct_prediction = tf.equal(outputs, y_input)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialization
    init = tf.global_variables_initializer()

    # Model saver
    saver = tf.train.Saver()


# =====================================================================
# Model Training
# =====================================================================

start_time = time.time()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(graph=graph,
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    sess.run(init)

    validation_dict = {
        x_input: validation_data[:, bi],
        y_input: validation_data[:, li]
    }

    for i in range(N_ITER + 1):

        batch = training_data[
            np.random.choice(training_size, BATCH_SIZE, False), :
        ]

        feed_dict = {
            x_input: batch[:, bi],
            y_input: batch[:, li]
        }

        optimizer.run(feed_dict=feed_dict)

        if i % 100 == 0:

            acc = accuracy.eval(validation_dict) * 100

            model_path = f'{folder_modelo}/col3_{biome}_r{region}_v{version}_rnn_lstm_ckpt'

            saver.save(sess, model_path)

            print('Accuracy %.2f%% at step %s' % (acc, i))


# =====================================================================
# Training Summary
# =====================================================================

end_time = time.time()
training_time = end_time - start_time

print(colored(
    'Spent time: {0}'.format(time.strftime("%H:%M:%S",
    time.gmtime(training_time))), 'yellow'))

print(colored(
    f'Model saved at: {model_path}', 'green'))