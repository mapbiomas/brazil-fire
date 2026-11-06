"""Golden tests: annual/monthly paths and names must not change."""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from M0_auth_config import (
    get_asset_mosaic_collection,
    monthly_chunk_path,
    monthly_cog_path,
    mosaic_name,
    mosaic_name_for_period,
    period_chunk_path,
    period_cog_path,
    set_global_opts,
    yearly_chunk_path,
    yearly_cog_path,
)


class PeriodPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_opts(
            country='testland',
            gcs_bucket='bucket',
            gcs_library_images_prefix='sudamerica/testland/CATALOG_01',
            gcs_campaigns_prefix='sudamerica/testland/CATALOG_01',
            gee_project='mapbiomas-testland',
            gee_library_images_prefix='projects/mapbiomas-testland/assets/FIRE/CATALOG_01',
            gee_campaigns_prefix='projects/mapbiomas-testland/assets/FIRE/CATALOG_01',
            campaign='COLLECTION_02',
            asset_regions='projects/mapbiomas-testland/assets/FIRE/CATALOG_01/COLLECTION_02/AUXILIARY/regions',
            sensor=['landsat'],
            periodicity=['yearly'],
            mosaic_methods=['minnbr'],
            language='en',
        )

    def test_monthly_paths_match_legacy(self):
        self.assertEqual(
            monthly_chunk_path(2025, 6),
            'sudamerica/testland/CATALOG_01/LIBRARY_IMAGES/LANDSAT/MONTHLY/MINNBR/2025_06/CHUNKS',
        )
        self.assertEqual(
            period_chunk_path('2025_06'),
            monthly_chunk_path(2025, 6),
        )
        self.assertEqual(
            monthly_cog_path(2025, 6),
            'sudamerica/testland/CATALOG_01/LIBRARY_IMAGES/LANDSAT/MONTHLY/MINNBR/2025_06/COG',
        )
        self.assertEqual(period_cog_path('2025_06'), monthly_cog_path(2025, 6))

    def test_yearly_paths_match_legacy(self):
        self.assertEqual(
            yearly_chunk_path(2025),
            'sudamerica/testland/CATALOG_01/LIBRARY_IMAGES/LANDSAT/ANNUALLY/MINNBR/2025/CHUNKS',
        )
        self.assertEqual(period_chunk_path('2025'), yearly_chunk_path(2025))
        self.assertEqual(
            yearly_cog_path(2025),
            'sudamerica/testland/CATALOG_01/LIBRARY_IMAGES/LANDSAT/ANNUALLY/MINNBR/2025/COG',
        )
        self.assertEqual(period_cog_path('2025'), yearly_cog_path(2025))

    def test_names_match_legacy(self):
        self.assertEqual(
            mosaic_name(2025, 6, 'monthly', band='red'),
            'image_testland_fire_landsat_minnbr_red_2025_06',
        )
        self.assertEqual(
            mosaic_name_for_period('2025_06', band='red'),
            mosaic_name(2025, 6, 'monthly', band='red'),
        )
        self.assertEqual(
            mosaic_name_for_period('2025'),
            'image_testland_fire_landsat_minnbr_2025',
        )

    def test_future_periods_are_configuration_only(self):
        self.assertTrue(
            period_chunk_path('2025_Q1').endswith('/LANDSAT/QUARTERLY/MINNBR/2025_Q1/CHUNKS')
        )
        self.assertTrue(
            period_cog_path('2025_S1').endswith('/LANDSAT/SEMIANNUAL/MINNBR/2025_S1/COG')
        )
        self.assertEqual(
            mosaic_name_for_period('2025_Q1', band='red'),
            'image_testland_fire_landsat_minnbr_red_2025_Q1',
        )

    def test_asset_collections_use_registry_folders(self):
        self.assertTrue(
            get_asset_mosaic_collection(sensor='landsat', periodicity='yearly', mosaic='minnbr')
            .endswith('/LANDSAT/YEARLY/MINNBR')
        )
        self.assertTrue(
            get_asset_mosaic_collection(sensor='landsat', periodicity='monthly', mosaic='minnbr')
            .endswith('/LANDSAT/MONTHLY/MINNBR')
        )
        self.assertTrue(
            get_asset_mosaic_collection(
                sensor='landsat', period_key='2025_Q1', mosaic='minnbr'
            ).endswith('/LANDSAT/QUARTERLY/MINNBR')
        )


if __name__ == '__main__':
    unittest.main()
