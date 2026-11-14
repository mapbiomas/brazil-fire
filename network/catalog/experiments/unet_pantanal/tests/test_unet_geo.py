"""Tests for the U-Net geometry helpers (need shapely + rasterio; skipped otherwise)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import rasterio  # noqa: F401
    import shapely  # noqa: F401
    import geopandas  # noqa: F401
    HAVE_GEO = True
except ImportError:
    HAVE_GEO = False


@unittest.skipUnless(HAVE_GEO, "shapely/rasterio/geopandas not installed")
class TestGeo(unittest.TestCase):
    def test_block_grid_split_and_box(self):
        from unet_geo import BlockGrid
        g = BlockGrid((0, 0, 100, 100), 10, 10)
        pts = [(x + 5, y + 5) for x in range(0, 100, 10) for y in range(0, 100, 10)]
        g.assign(pts, (0.7, 0.15, 0.15), seed=42)
        c = g.counts()
        self.assertEqual(sum(c.values()), 100)
        self.assertEqual(c["train"], 70)
        self.assertEqual(c["val"], 15)
        # a box inside one block has one split; across a line, up to 4 blocks
        self.assertEqual(len(g.blocks_of_box(1, 1, 9, 9)), 1)
        self.assertEqual(len(g.blocks_of_box(5, 5, 15, 15)), 4)
        self.assertEqual(len(g.blocks_of_box(0, 0, 10, 10)), 1)  # edge on the line
        # same seed -> same assignment
        g2 = BlockGrid((0, 0, 100, 100), 10, 10).assign(pts, (0.7, 0.15, 0.15), seed=42)
        self.assertEqual(g.split, g2.split)

    def test_unused_blocks(self):
        from unet_geo import BlockGrid
        g = BlockGrid((0, 0, 100, 100), 10, 10).assign([(5, 5)], (0.7, 0.15, 0.15), 1)
        self.assertEqual(g.split_of(95, 95), "unused")

    def test_block_size_geographic(self):
        from unet_geo import block_size_in_crs
        dx, dy = block_size_in_crs(25, "EPSG:4326", -18.0)
        self.assertAlmostEqual(dy, 25 / 111.32, places=6)
        self.assertGreater(dx, dy)
        self.assertEqual(block_size_in_crs(25, "EPSG:5880", -18.0), (25000.0, 25000.0))

    def test_points_inside_polygon(self):
        from shapely.geometry import Polygon
        from unet_geo import sample_points_in_polygon
        poly = Polygon([(0, 0), (10, 0), (0, 10)])
        pts = sample_points_in_polygon(poly, 4, np.random.default_rng(0))
        self.assertEqual(len(pts), 4)
        self.assertTrue(all(poly.contains(p) for p in pts))
        sliver = Polygon([(0, 0), (10, 10), (10, 10 + 1e-9)])  # tiny area, large bbox
        pts = sample_points_in_polygon(sliver, 4, np.random.default_rng(0), max_tries=1)
        self.assertEqual(len(pts), 1)  # falls back to a representative point

    def test_balance(self):
        import pandas as pd
        from unet_geo import balance_by_class
        df = pd.DataFrame({"cls": [1] * 10 + [0] * 4 + [1] * 3 + [0] * 5,
                           "split": ["train"] * 14 + ["test"] * 8})
        out = balance_by_class(df, "cls", "split", 0)
        counts = out.groupby(["split", "cls"]).size().to_dict()
        self.assertEqual(counts, {("test", 0): 3, ("test", 1): 3, ("train", 0): 4, ("train", 1): 4})

    def test_chip_window_centered(self):
        from rasterio.transform import from_origin
        from unet_geo import chip_window, window_box
        tf = from_origin(0, 100, 1, 1)  # 1 unit pixels, top-left (0, 100)
        w = chip_window(tf, 50.5, 49.5, 8)
        self.assertEqual((w.col_off, w.row_off, w.width, w.height), (46, 46, 8, 8))
        self.assertEqual(window_box(tf, w), (46, 46, 54, 54))

    def test_labels_fire_wins(self):
        from rasterio.transform import from_origin
        from shapely.geometry import box
        from unet_geo import chip_quality, rasterize_labels
        tf = from_origin(0, 10, 1, 1)
        y, overlap = rasterize_labels([box(0, 5, 5, 10)], [box(3, 3, 10, 10)], (10, 10), tf)
        self.assertEqual(y[0, 0], 1)        # fire only
        self.assertEqual(y[0, 9], 0)        # not fire only
        self.assertEqual(y[9, 0], 255)      # no polygon
        self.assertEqual(y[2, 4], 1)        # overlap -> fire
        self.assertEqual(overlap, 2 * 5)    # x 3-5, y 5-10 -> 2 cols x 5 rows
        valid = np.ones((10, 10), bool)
        valid[:, :6] = False
        keep, lab, nod = chip_quality(y, valid, 0.01, 0.5)
        self.assertFalse(keep)
        self.assertAlmostEqual(nod, 0.6)


class TestBlockClasses(unittest.TestCase):
    """Block split keeps fire and not-fire in every split when it can (numpy only)."""

    def setUp(self):
        try:
            from unet_geo import BlockGrid  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_all_splits_get_both_classes(self):
        from unet_geo import BlockGrid
        g = BlockGrid((0, 0, 100, 100), 10, 10)
        # 20 blocks along a row: fire in every block, not-fire in every second block
        pts = [(5 + 10 * i, 5, 1) for i in range(10)] + [(5 + 10 * i, 15, 0) for i in range(0, 10, 2)]
        g.assign([(x, y) for x, y, _ in pts], (0.7, 0.15, 0.15), seed=1, classes_xy=pts)
        self.assertTrue(g.stratified)
        for s in ("train", "val", "test"):
            cls = {c for (x, y, c) in pts if g.split_of(x, y) == s}
            self.assertEqual(cls, {0, 1}, s)

    def test_falls_back_when_impossible(self):
        from unet_geo import BlockGrid
        g = BlockGrid((0, 0, 100, 100), 10, 10)
        pts = [(5 + 10 * i, 5, 1) for i in range(6)] + [(5, 15, 0)]  # one single not-fire block
        g.assign([(x, y) for x, y, _ in pts], (0.7, 0.15, 0.15), seed=1, classes_xy=pts, tries=50)
        self.assertFalse(g.stratified)
        self.assertEqual(sum(g.counts().values()), 7)


if __name__ == "__main__":
    unittest.main()
