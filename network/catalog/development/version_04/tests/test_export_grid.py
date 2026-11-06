"""Offline tests for the experimental export grid."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_export_grid import tiles_for_bounds, filter_intersecting_tiles


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

    def test_country_filter_keeps_intersecting_and_near_boundary_tiles(self):
        from shapely.geometry import mapping, box
        tiles = list(tiles_for_bounds((0, 0, 10, 10), 1, 1))
        selected = filter_intersecting_tiles(tiles, mapping(box(0.2, 0.2, 0.8, 0.8)))
        ids = {tile.id for tile in selected}
        self.assertIn('c+0_r+0', ids)
        self.assertIn('c+1_r+0', ids)
        self.assertNotIn('c+9_r+9', ids)
        self.assertLess(len(selected), len(tiles))

    def test_invalid_country_is_not_silently_ignored(self):
        from shapely.geometry import mapping, Polygon
        tiles = list(tiles_for_bounds((0, 0, 2, 2), 1, 1))
        bow_tie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        with self.assertRaises(ValueError):
            filter_intersecting_tiles(tiles, mapping(bow_tie))

    def test_geometry_collection_is_supported(self):
        from shapely.geometry import mapping, box, GeometryCollection
        tiles = list(tiles_for_bounds((0, 0, 10, 10), 1, 1))
        region = GeometryCollection([box(0.2, 0.2, 0.8, 0.8), box(5.2, 5.2, 5.8, 5.8)])
        ids = {tile.id for tile in filter_intersecting_tiles(tiles, mapping(region))}
        self.assertIn('c+0_r+0', ids)
        self.assertIn('c+5_r+5', ids)
        self.assertNotIn('c+9_r+9', ids)

    def test_invalid_arguments(self):
        for bounds, scale, size in [((0, 0, 0, 1), 1, 10), ((0, 0, 1, 1), 0, 10),
                                    ((0, 0, 1, 1), 1, True), ((0, 0, 1, 1), 1, 1.5)]:
            with self.subTest(bounds=bounds, scale=scale, size=size):
                with self.assertRaises(ValueError):
                    list(tiles_for_bounds(bounds, scale, size))


if __name__ == '__main__':
    unittest.main()
