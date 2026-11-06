"""Offline tests for the experimental export grid."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_export_grid import tiles_for_bounds


class ExportGridTests(unittest.TestCase):
    def test_grid_is_anchored_and_does_not_duplicate_boundary(self):
        tiles = list(tiles_for_bounds((-20, -10, 20, 10), 1, 10))
        self.assertEqual(len(tiles), 8)
        self.assertEqual(len({tile.id for tile in tiles}), 8)
        self.assertEqual(tiles[0].id, 'c-2_r-1')
        self.assertEqual((tiles[-1].xmin, tiles[-1].xmax), (10, 20))

    def test_partial_bounds_keep_stable_ids(self):
        full = {t.id for t in tiles_for_bounds((-20, -10, 20, 10), 1, 10)}
        partial = {t.id for t in tiles_for_bounds((-19, -9, 9, 9), 1, 10)}
        self.assertTrue(partial < full)

    def test_invalid_arguments(self):
        for bounds, scale, size in [((0, 0, 0, 1), 1, 10), ((0, 0, 1, 1), 0, 10),
                                    ((0, 0, 1, 1), 1, True), ((0, 0, 1, 1), 1, 1.5)]:
            with self.subTest(bounds=bounds, scale=scale, size=size):
                with self.assertRaises(ValueError):
                    list(tiles_for_bounds(bounds, scale, size))


if __name__ == '__main__':
    unittest.main()
