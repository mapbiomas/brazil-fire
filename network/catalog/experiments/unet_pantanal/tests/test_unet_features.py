"""Tests for the U-Net feature/metric helpers (NumPy only)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import FULL_BANDS as DEFAULT_BANDS  # noqa: E402
from unet_config import FULL_INDICES as DEFAULT_INDICES  # noqa: E402
from unet_config import make_config  # noqa: E402
from unet_features import (PixelReservoir, add_indices, best_threshold,  # noqa: E402
                           compute_norm_stats, metrics_at, normalize,
                           prob_histograms, scale_bands)


class TestConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = make_config()
        self.assertEqual(cfg["SCALE"], 100.0)
        self.assertEqual(cfg["BANDS"], ["green", "swir2"])
        self.assertEqual(cfg["INDICES"], [])
        self.assertIsNone(cfg["VALID_BAND"])  # dayOfYear not exported yet

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            make_config(NOPE=1)

    def test_bad_chip(self):
        with self.assertRaises(ValueError):
            make_config(CHIP=100)

    def test_bad_split(self):
        with self.assertRaises(ValueError):
            make_config(SPLIT=(0.5, 0.2, 0.2))


class TestIndices(unittest.TestCase):
    def setUp(self):
        # blue, red, nir, swir1, swir2
        self.px = np.array([[0.05, 0.08, 0.30, 0.20, 0.10]], dtype=np.float32)

    def test_formulas(self):
        out = add_indices(self.px, DEFAULT_BANDS, DEFAULT_INDICES)[0]
        b, r, n, s1, s2 = self.px[0]
        e = 1e-6
        expected = [
            (n - s2) / (n + s2 + e),
            (s1 - s2) / (s1 + s2 + e),
            10 * s2 - 9.8 * s1 + 2,
            (n - r) / (n + r + e),
            1 / ((0.1 - r) ** 2 + (0.06 - n) ** 2 + e),
            2.5 * (n - r) / (n + 6 * r - 7.5 * b + 1),
        ]
        np.testing.assert_allclose(out[5:], expected, rtol=1e-5)
        np.testing.assert_allclose(out[:5], self.px[0])

    def test_shape_and_nan(self):
        x = np.random.rand(4, 4, 5).astype(np.float32)
        x[0, 0] = np.nan
        out = add_indices(x, DEFAULT_BANDS, DEFAULT_INDICES)
        self.assertEqual(out.shape, (4, 4, 11))
        self.assertTrue(np.all(np.isnan(out[0, 0])))

    def test_scale_bands(self):
        raw = np.full((5, 2, 2), 50, dtype=np.uint8)
        valid = np.array([[True, False], [True, True]])
        x = scale_bands(raw, 100.0, valid)
        self.assertEqual(x.shape, (2, 2, 5))
        self.assertAlmostEqual(float(x[0, 0, 0]), 0.5)
        self.assertTrue(np.all(np.isnan(x[0, 1])))


class TestNormalization(unittest.TestCase):
    def test_stats_and_normalize(self):
        rng = np.random.default_rng(0)
        px = rng.normal(5, 2, size=(10000, 2))
        st = compute_norm_stats(px, ["a", "b"], (1, 99))
        x = np.array([[[5.0, 5.0], [np.nan, 1e9]]], dtype=np.float32)
        out = normalize(x, st)
        self.assertEqual(out[0, 1, 0], 0.0)  # nodata -> 0
        self.assertAlmostEqual(float(out[0, 1, 1]), (st["clip_high"][1] - st["mean"][1]) / st["std"][1], places=4)
        self.assertLess(abs(float(out[0, 0, 0])), 0.1)

    def test_reservoir_bounded_and_uniform(self):
        res = PixelReservoir(1000, 1, seed=1)
        for k in range(10):
            res.add(np.full((500, 1), k, dtype=np.float32))
        self.assertEqual(len(res.data), 1000)
        self.assertEqual(res.seen, 5000)
        counts = np.bincount(res.data[:, 0].astype(int), minlength=10)
        self.assertTrue(np.all(counts > 50))  # every chunk represented

    def test_reservoir_skips_nan(self):
        res = PixelReservoir(10, 2)
        res.add(np.array([[1, 2], [np.nan, 1]], dtype=np.float32))
        self.assertEqual(len(res.data), 1)


class TestMetrics(unittest.TestCase):
    def test_metrics_ignore_255(self):
        prob = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.99])
        lab = np.array([1, 1, 1, 0, 0, 255])
        hf, hn = prob_histograms(prob, lab)
        m = metrics_at(hf, hn, 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["iou"], 0.5)
        self.assertAlmostEqual(m["omission"], 1 / 3)

    def test_best_threshold(self):
        prob = np.array([0.35, 0.4, 0.3, 0.1])
        lab = np.array([1, 1, 0, 0])
        hf, hn = prob_histograms(prob, lab)
        best = best_threshold(hf, hn)
        self.assertEqual(best["f1"], 1.0)
        self.assertTrue(0.3 < best["threshold"] <= 0.35)


if __name__ == "__main__":
    unittest.main()
