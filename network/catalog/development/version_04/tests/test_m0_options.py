"""Offline tests for the mosaic-method configuration contract.

The notebook always states the active methods; these tests lock that in and
cover the per-sensor/period filtering used by M1 and M2.
"""

import os
import sys
import unittest
from unittest.mock import patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M_mosaics
from M0_auth_config import set_global_opts


def options(**overrides):
    values = {
        'country': 'testland',
        'gcs_bucket': 'bucket',
        'gcs_library_images_prefix': 'sudamerica/testland/CATALOG_01',
        'gcs_campaigns_prefix': 'sudamerica/testland/CATALOG_01',
        'gee_project': 'mapbiomas-testland',
        'gee_library_images_prefix': 'projects/mapbiomas-testland/assets/FIRE/CATALOG_01',
        'gee_campaigns_prefix': 'projects/mapbiomas-testland/assets/FIRE/CATALOG_01',
        'campaign': 'MONITOR_01',
        'asset_regions': 'regions',
        'sensor': 'sentinel2',
        'periodicity': 'monthly',
        'mosaic_methods': ['minnbr'],
        'language': 'en',
    }
    values.update(overrides)
    return values


class MosaicMethodConfigTests(unittest.TestCase):
    def test_mosaic_methods_is_required(self):
        values = options()
        values.pop('mosaic_methods')
        with self.assertRaises(ValueError) as context:
            set_global_opts(**values)
        self.assertIn('mosaic_methods is required', str(context.exception))

    def test_config_keeps_exactly_what_the_notebook_passed(self):
        set_global_opts(**options(mosaic_methods=['minnbr', 'minnbr_buffer']))
        self.assertEqual(
            M0_auth_config.CONFIG['mosaic_methods'],
            ['minnbr', 'minnbr_buffer'],
        )
        self.assertEqual(M0_auth_config.GLOBAL_OPTS['MOSAIC_METHOD'], 'minnbr')

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            set_global_opts(**options(mosaic_methods=['does_not_exist']))

    def test_method_must_support_the_selected_sensor_and_period(self):
        with patch.object(
            M_mosaics, 'available_methods', return_value=['minnbr']
        ):
            with self.assertRaises(ValueError) as context:
                set_global_opts(**options(mosaic_methods=['minnbr_buffer']))
        self.assertIn('not supported', str(context.exception))

    def test_hls_and_modis_are_declared_for_every_method(self):
        expected = ['median', 'minnbr', 'minnbr_buffer', 'minndvi']
        self.assertEqual(M_mosaics.available_methods('hls', 'monthly'), expected)
        self.assertEqual(M_mosaics.available_methods('modis', 'yearly'), expected)

    def test_methods_for_filters_by_pair(self):
        with patch.dict(
            M_mosaics.MOSAIC_METHODS,
            {
                'minnbr': {'sensors': ['sentinel2'], 'periods': ['monthly']},
                'median': {'sensors': ['landsat'], 'periods': ['monthly']},
            },
            clear=True,
        ):
            self.assertEqual(
                M_mosaics.methods_for(
                    ['minnbr', 'median'], sensor='landsat', period='monthly'
                ),
                ['median'],
            )

    def test_methods_for_accepts_a_single_string(self):
        self.assertEqual(
            M_mosaics.methods_for('minnbr', sensor='hls', period='monthly'),
            ['minnbr'],
        )

    def test_methods_for_unknown_pair_returns_the_configured_methods(self):
        self.assertEqual(
            M_mosaics.methods_for(
                ['minnbr', 'median'], sensor='unknown', period='monthly'
            ),
            ['minnbr', 'median'],
        )


if __name__ == '__main__':
    unittest.main()
