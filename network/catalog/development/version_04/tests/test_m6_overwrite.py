"""Offline tests for M6 replacing existing GEE image assets in place."""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from M6_image_upload import build_image_upload_command


class ImageUploadOverwriteTests(unittest.TestCase):
    def test_upload_targets_same_asset_with_force_and_preserves_metadata(self):
        command = build_image_upload_command(
            project='mapbiomas-testland',
            asset_id='projects/mapbiomas-testland/assets/region_2025_08',
            gcs_uri='gs://bucket/region.tif',
            metadata={
                'time_start': 1754006400000,
                'time_end': 1756684800000,
                'model_id': 'model_a',
                'region': 'region',
                'period': '2025_08',
                'year': '2025',
                'month': '08',
                'campaign': 'MONITOR_01',
                'country': 'testland',
                'version': 'version_04',
                'type': 'monthly',
                'periodicity': 'monthly',
            },
        )

        self.assertEqual(command[0:5], [
            'earthengine', '--project=mapbiomas-testland', 'upload', 'image', '--force',
        ])
        self.assertIn('--asset_id=projects/mapbiomas-testland/assets/region_2025_08', command)
        self.assertIn('--time_start', command)
        self.assertIn('1754006400000', command)
        self.assertIn('--property', command)
        self.assertIn('model_id=model_a', command)
        self.assertIn('region=region', command)
        self.assertEqual(command[-1], 'gs://bucket/region.tif')
        self.assertNotIn('delete', command)


if __name__ == '__main__':
    unittest.main()
