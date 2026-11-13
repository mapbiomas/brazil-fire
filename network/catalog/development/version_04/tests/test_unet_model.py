"""Tests for the U-Net model, dataset and inference tiling (need TensorFlow; skipped otherwise)."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "unet"))

try:
    import tensorflow as tf
    HAVE_TF = True
except ImportError:
    HAVE_TF = False

try:
    import rasterio  # noqa: F401
    HAVE_RIO = True
except ImportError:
    HAVE_RIO = False


@unittest.skipUnless(HAVE_TF, "tensorflow not installed")
class TestModel(unittest.TestCase):
    def setUp(self):
        from unet_config import make_config
        self.cfg = make_config(BATCH=2, FILTERS=[4, 8], BOTTOM_FILTERS=8, CHIP=16,
                               INFER_TILE=16, OVERLAP=8)
        self.stats = {"clip_low": [-1.0] * 11, "clip_high": [2.0] * 11,
                      "mean": [0.5] * 11, "std": [0.25] * 11}

    def test_unet_shape_any_size(self):
        from unet_model import build_unet
        m = build_unet(11, (4, 8), 8)
        for size in (16, 32, 48):
            out = m(np.zeros((1, size, size, 11), np.float32))
            self.assertEqual(tuple(out.shape), (1, size, size, 1))

    def test_loss_ignores_255(self):
        from unet_model import masked_bce_dice
        y = np.array([[1, 0], [255, 255]], np.float32).reshape(1, 2, 2, 1)
        p1 = np.array([[0.9, 0.1], [0.5, 0.5]], np.float32).reshape(1, 2, 2, 1)
        p2 = np.array([[0.9, 0.1], [0.0, 1.0]], np.float32).reshape(1, 2, 2, 1)
        self.assertAlmostEqual(float(masked_bce_dice(y, p1)), float(masked_bce_dice(y, p2)), places=5)

    def test_metric_masked(self):
        from unet_model import masked_metric_class
        M = masked_metric_class()
        y = np.array([1, 1, 0, 0, 255, 255], np.float32).reshape(1, 1, 6, 1)
        p = np.array([0.9, 0.2, 0.8, 0.1, 0.9, 0.9], np.float32).reshape(1, 1, 6, 1)
        iou, f1 = M("iou"), M("f1")
        iou.update_state(y, p)
        f1.update_state(y, p)
        self.assertAlmostEqual(float(iou.result()), 1 / 3, places=5)
        self.assertAlmostEqual(float(f1.result()), 0.5, places=5)

    def test_tf_normalize_matches_numpy(self):
        from unet_features import normalize
        from unet_model import tf_normalize
        x = np.random.default_rng(0).normal(0.5, 1, (4, 4, 11)).astype(np.float32)
        x[0, 0, :] = np.nan
        x[1, 1, 3] = np.inf
        np.testing.assert_allclose(tf_normalize(tf.constant(x), self.stats).numpy(),
                                   normalize(x, self.stats), atol=1e-6)

    def test_chip_roundtrip_and_dataset(self):
        from unet_chips import decode_chip_numpy, encode_chip
        from unet_model import make_dataset
        rng = np.random.default_rng(1)
        x = rng.random((16, 16, 11)).astype(np.float32)
        x[0, 0] = np.nan
        y = rng.choice([0, 1, 255], (16, 16)).astype(np.uint8)
        rec = encode_chip(x, y, 2024, "train", 1.5, -2.5)
        x2, y2, meta = decode_chip_numpy(rec)
        np.testing.assert_array_equal(np.isnan(x2), np.isnan(x))
        np.testing.assert_array_equal(y2, y)
        self.assertEqual(meta["split"], "train")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "2024_train_000.tfrecord.gz")
            with tf.io.TFRecordWriter(path, options="GZIP") as w:
                for _ in range(3):
                    w.write(rec)
            batches = list(make_dataset([path], self.stats, self.cfg, training=True))
            xb, yb = batches[0]
            self.assertEqual(tuple(xb.shape), (2, 16, 16, 11))
            self.assertEqual(tuple(yb.shape), (2, 16, 16, 1))
            self.assertTrue(np.all(np.isfinite(xb.numpy())))
            # augmentation moves x and y together: label counts are preserved
            self.assertEqual(sorted(np.unique(yb.numpy()).tolist()), sorted(np.unique(y).astype(float).tolist()))

    def test_augment_aligned(self):
        from unet_model import _augment
        x = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
        y = x.copy()
        for _ in range(10):
            xa, ya = _augment(tf.constant(x), tf.constant(y))
            np.testing.assert_array_equal(xa.numpy(), ya.numpy())

    def test_short_training_runs(self):
        from unet_model import build_unet, masked_bce_dice, masked_metric_class
        M = masked_metric_class()
        m = build_unet(11, (4, 8), 8)
        m.compile(optimizer="adam", loss=masked_bce_dice, metrics=[M("iou")])
        x = np.random.rand(4, 16, 16, 11).astype(np.float32)
        y = (x[..., :1] > 0.5).astype(np.float32)
        y[:, :2] = 255
        h = m.fit(x, y, epochs=2, verbose=0)
        self.assertIn("iou_fire", h.history)


@unittest.skipUnless(HAVE_RIO, "rasterio not installed")
class TestTiling(unittest.TestCase):
    def test_core_tiles_cover_once(self):
        from rasterio.windows import Window
        from unet_predict import core_tiles
        win = Window(7, 3, 50, 37)
        cover = np.zeros((60, 60), int)
        for r, c, h, w in core_tiles(win, 16):
            cover[r:r + h, c:c + w] += 1
        self.assertTrue(np.all(cover[3:40, 7:57] == 1))
        self.assertEqual(cover.sum(), 50 * 37)

    def test_to_output_m5_rule(self):
        from unet_predict import NODATA, to_output
        prob = np.array([[0.0, 0.0004, 0.0006, 0.9]])
        valid = np.array([[True, True, True, False]])
        doy = np.array([[10, 20, 30, 40]])
        p, d = to_output(prob, valid, doy)
        self.assertEqual(p.tolist(), [[0, 0, 1, NODATA]])
        self.assertEqual(d.tolist(), [[NODATA, NODATA, 30, NODATA]])


if __name__ == "__main__":
    unittest.main()
