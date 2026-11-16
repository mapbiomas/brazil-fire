"""Regression tests for region-scoped M5 inference."""

import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from M5_spatial import classify_region_block, clip_geometry_to_cell, rasterize_region_mask


class RegionBlockTests(unittest.TestCase):
    def test_predicts_only_valid_pixels_inside_requested_region(self):
        stack = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        data_valid = np.array([[True, True], [False, True]])
        region_mask = np.array([[True, False], [True, True]])
        day_of_year = np.array([[50, 60], [70, 250]], dtype=np.int16)
        normalize = Mock(side_effect=lambda values, _: values)
        predict = Mock(return_value=np.array([0.0, 0.8], dtype=np.float32))

        result = classify_region_block(
            stack, data_valid, region_mask, day_of_year,
            norm_stats={}, normalize_fn=normalize, predict_fn=predict,
        )

        np.testing.assert_array_equal(normalize.call_args.args[0], [[1.0], [4.0]])
        np.testing.assert_array_equal(
            result['probability'],
            np.array([[0, -9999], [-9999, 800]], dtype=np.int16),
        )
        np.testing.assert_array_equal(
            result['day_of_year'],
            np.array([[-9999, -9999], [-9999, 250]], dtype=np.int16),
        )
        self.assertEqual(result['valid_pixels'], 2)
        self.assertEqual(result['burned_pixels'], 1)
        self.assertAlmostEqual(result['confidence_sum'], 0.8)

    def test_empty_region_does_not_call_model(self):
        predict = Mock(side_effect=AssertionError('model must not run'))
        result = classify_region_block(
            np.ones((2, 2, 1), dtype=np.float32),
            np.ones((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=bool),
            np.ones((2, 2), dtype=np.int16),
            norm_stats={}, normalize_fn=lambda values, _: values,
            predict_fn=predict,
        )

        predict.assert_not_called()
        self.assertEqual(result['valid_pixels'], 0)
        self.assertTrue((result['probability'] == -9999).all())
        self.assertTrue((result['day_of_year'] == -9999).all())

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'same height and width'):
            classify_region_block(
                np.ones((2, 2, 1)), np.ones((2, 2), dtype=bool),
                np.ones((1, 2), dtype=bool), None,
                norm_stats={}, normalize_fn=lambda values, _: values,
                predict_fn=lambda values: np.zeros(len(values)),
            )


class GeometryMaskTests(unittest.TestCase):
    def test_rasterize_uses_pixel_center_region_mask(self):
        expected = np.array([[True, False], [False, True]])
        rasterio = types.ModuleType('rasterio')
        features = types.ModuleType('rasterio.features')
        features.geometry_mask = Mock(return_value=expected)
        rasterio.features = features
        with patch.dict(sys.modules, {
            'rasterio': rasterio,
            'rasterio.features': features,
        }):
            actual = rasterize_region_mask(
                [{'type': 'Polygon', 'coordinates': []}],
                out_shape=(2, 2), transform='affine',
            )
        np.testing.assert_array_equal(actual, expected)
        features.geometry_mask.assert_called_once_with(
            [{'type': 'Polygon', 'coordinates': []}],
            out_shape=(2, 2), transform='affine', invert=True,
            all_touched=False,
        )

    @unittest.skipUnless(
        importlib.util.find_spec('shapely') and importlib.util.find_spec('pyproj'),
        'Shapely and pyproj are needed for geometry projection tests',
    )
    def test_geometry_is_intersection_of_region_and_cell(self):
        region = {
            'type': 'Polygon',
            'coordinates': [[[2, 1], [5, 1], [5, 3], [2, 3], [2, 1]]],
        }
        cell = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
        }

        geometries, bounds = clip_geometry_to_cell(region, cell, 'EPSG:4326')

        self.assertEqual(bounds, (2.0, 1.0, 4.0, 3.0))
        self.assertEqual(geometries[0]['type'], 'Polygon')

    @unittest.skipUnless(
        importlib.util.find_spec('shapely') and importlib.util.find_spec('pyproj'),
        'Shapely and pyproj are needed for geometry projection tests',
    )
    def test_intersection_bounds_are_projected_to_source_raster_crs(self):
        region = {
            'type': 'Polygon',
            'coordinates': [[[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]]],
        }
        cell = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
        }

        _, bounds = clip_geometry_to_cell(region, cell, 'EPSG:3857')

        self.assertAlmostEqual(bounds[0], 111319.49, delta=1)
        self.assertAlmostEqual(bounds[1], 111325.14, delta=1)
        self.assertAlmostEqual(bounds[2], 333958.47, delta=1)
        self.assertAlmostEqual(bounds[3], 334111.17, delta=1)

    @unittest.skipUnless(
        importlib.util.find_spec('shapely') and importlib.util.find_spec('pyproj'),
        'Shapely and pyproj are needed for geometry projection tests',
    )
    def test_touching_boundary_without_area_is_not_classified(self):
        region = {
            'type': 'Polygon',
            'coordinates': [[[4, 1], [5, 1], [5, 3], [4, 3], [4, 1]]],
        }
        cell = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
        }

        geometries, bounds = clip_geometry_to_cell(region, cell, 'EPSG:4326')

        self.assertEqual(geometries, [])
        self.assertIsNone(bounds)


if __name__ == '__main__':
    unittest.main()
