"""Pure-NumPy forward pass for the M4 feed-forward DNN.

Inference must not depend on TensorFlow graph/session state: M5 runs inside
worker threads, and Keras ``predict_on_batch`` creates tensors in a different
default graph per thread. The saved ``weights.npz`` is just Dense layers, so
NumPy is enough (and thread-safe).
"""

import numpy as np


def forward_embeddings(saved_vars, layers, x_normalized):
    """Return the last hidden layer (ReLU) for already-normalized inputs."""
    layer = np.asarray(x_normalized, dtype=np.float32)
    for index in range(len(layers)):
        kernel = saved_vars[f'fc_{index}/kernel:0']
        bias = saved_vars[f'fc_{index}/bias:0']
        layer = np.maximum(0, np.dot(layer, kernel) + bias)
    return layer


def forward_predict(saved_vars, layers, x_normalized):
    """Return one probability per row for already-normalized inputs."""
    layer = forward_embeddings(saved_vars, layers, x_normalized)
    kernel = saved_vars['output/kernel:0']
    bias = saved_vars['output/bias:0']
    logits = np.dot(layer, kernel) + bias
    probs = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
    return probs.flatten()


def predict_chunked(saved_vars, layers, x_normalized, chunk=65536):
    """Predict in bounded chunks to keep peak memory low on large tiles."""
    if len(x_normalized) <= chunk:
        return forward_predict(saved_vars, layers, x_normalized)
    parts = [
        forward_predict(saved_vars, layers, x_normalized[start:start + chunk])
        for start in range(0, len(x_normalized), chunk)
    ]
    return np.concatenate(parts, axis=0)
