"""Offline tests for the NumPy DNN forward pass used by M5."""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

try:
    import numpy as np
    from M_dnn import forward_embeddings, forward_predict, predict_chunked
    HAS_NUMPY = True
except ImportError:
    np = None
    forward_embeddings = forward_predict = predict_chunked = None
    HAS_NUMPY = False


def _sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -20, 20)))


@unittest.skipUnless(HAS_NUMPY, "NumPy is not installed in this environment")
class ForwardPassTests(unittest.TestCase):
    def test_matches_manual_dense_math(self):
        saved = {
            'fc_0/kernel:0': np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32),
            'fc_0/bias:0': np.array([0.0, 0.1], dtype=np.float32),
            'output/kernel:0': np.array([[2.0], [-1.0]], dtype=np.float32),
            'output/bias:0': np.array([0.3], dtype=np.float32),
        }
        x = np.array([[1.0, 2.0], [-1.0, 0.5]], dtype=np.float32)

        hidden = np.maximum(0, np.dot(x, saved['fc_0/kernel:0']) + saved['fc_0/bias:0'])
        expected = _sigmoid(
            np.dot(hidden, saved['output/kernel:0']) + saved['output/bias:0']
        ).flatten()

        np.testing.assert_allclose(
            forward_embeddings(saved, [2], x), hidden, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            forward_predict(saved, [2], x), expected, rtol=1e-6, atol=1e-6
        )

    def test_chunked_matches_single_call(self):
        rng = np.random.default_rng(11)
        saved = {
            'fc_0/kernel:0': rng.normal(size=(4, 3)).astype(np.float32),
            'fc_0/bias:0': rng.normal(size=(3,)).astype(np.float32),
            'output/kernel:0': rng.normal(size=(3, 1)).astype(np.float32),
            'output/bias:0': rng.normal(size=(1,)).astype(np.float32),
        }
        x = rng.random((1000, 4), dtype=np.float32)

        np.testing.assert_allclose(
            predict_chunked(saved, [3], x, chunk=100),
            forward_predict(saved, [3], x),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_large_block_shape_from_m5(self):
        rng = np.random.default_rng(7)
        saved = {
            'fc_0/kernel:0': rng.normal(size=(4, 7)).astype(np.float32),
            'fc_0/bias:0': rng.normal(size=(7,)).astype(np.float32),
            'output/kernel:0': rng.normal(size=(7, 1)).astype(np.float32),
            'output/bias:0': rng.normal(size=(1,)).astype(np.float32),
        }
        x = rng.random((65536, 4), dtype=np.float32)

        probs = forward_predict(saved, [7], x)
        self.assertEqual(probs.shape, (65536,))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())


if __name__ == '__main__':
    unittest.main()
