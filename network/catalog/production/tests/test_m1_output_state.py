"""Offline tests for the M1 cell status contract.

M2 removes the source chunks after a COG is built, so a GCS row must stay OK
when either the chunks or the final COG exists. OK rows link to the COG, the
chunks folder or the GEE asset.
"""

import datetime
import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M1_export_ui
from M0_auth_config import set_global_opts
from M1_export_ui import (
    ExportDispatcherUI,
    output_link_url,
    resolve_output_state,
)


COUNTRY = 'testland'
MOSAIC = 'minnbr'
SENSOR = 'landsat'
PERIOD = 'yearly'
PERIOD_KEY = '2025'
BAND = 'blue'
MOSAIC_NAME = f'image_{COUNTRY}_fire_{SENSOR}_{MOSAIC}_{BAND}_{PERIOD_KEY}'
COG_DIR = (
    f'sudamerica/{COUNTRY}/CATALOG_01/LIBRARY_IMAGES/'
    f'LANDSAT/ANNUALLY/MINNBR/{PERIOD_KEY}/COG'
)
CHUNK_DIR = (
    f'sudamerica/{COUNTRY}/CATALOG_01/LIBRARY_IMAGES/'
    f'LANDSAT/ANNUALLY/MINNBR/{PERIOD_KEY}/CHUNKS'
)
ASSET_COLLECTION = (
    f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01/'
    f'LIBRARY_IMAGES/LANDSAT/YEARLY/MINNBR/{BAND}'
)


class OutputStateTests(unittest.TestCase):
    def test_gcs_chunks_only_is_ok(self):
        self.assertEqual(
            resolve_output_state('GCS', True, False, False, False),
            ('OK', True),
        )

    def test_gcs_cog_only_is_ok(self):
        self.assertEqual(
            resolve_output_state('GCS', False, True, False, False),
            ('OK', True),
        )

    def test_gcs_with_nothing_is_miss(self):
        self.assertEqual(
            resolve_output_state('GCS', False, False, False, False),
            ('MISS', False),
        )

    def test_asset_row_only_depends_on_the_asset(self):
        self.assertEqual(
            resolve_output_state('ASSET', False, False, True, False),
            ('OK', True),
        )
        self.assertEqual(
            resolve_output_state('ASSET', True, True, False, False),
            ('MISS', False),
        )

    def test_running_task_wins_over_existing_outputs(self):
        self.assertEqual(
            resolve_output_state('GCS', True, True, True, True),
            ('RUN', True),
        )


class OutputLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_opts(
            country=COUNTRY,
            gcs_bucket='bucket',
            gcs_library_images_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gcs_campaigns_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gee_project=f'mapbiomas-{COUNTRY}',
            gee_library_images_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            gee_campaigns_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            campaign='COLLECTION_02',
            asset_regions='regions',
            sensor=[SENSOR],
            periodicity=[PERIOD],
            mosaic_methods=[MOSAIC],
            language='en',
        )
        # M1 imported CONFIG before the first set_global_opts() bound it.
        M1_export_ui.CONFIG = M0_auth_config.CONFIG

    def link(self, row_type, has_chunks, has_cog, has_asset):
        return output_link_url(
            row_type, has_chunks, has_cog, has_asset,
            sensor=SENSOR, period=PERIOD, mosaic=MOSAIC,
            period_key=PERIOD_KEY, band=BAND, mosaic_name=MOSAIC_NAME,
        )

    def test_cog_row_links_to_the_cog_file(self):
        self.assertEqual(
            self.link('GCS', False, True, False),
            f'https://storage.googleapis.com/bucket/{COG_DIR}/{MOSAIC_NAME}_cog.tif',
        )

    def test_chunks_only_row_links_to_the_chunks_folder(self):
        self.assertEqual(
            self.link('GCS', True, False, False),
            f'https://console.cloud.google.com/storage/browser/bucket/{CHUNK_DIR}',
        )

    def test_cog_link_wins_when_chunks_and_cog_coexist(self):
        self.assertEqual(
            self.link('GCS', True, True, False),
            f'https://storage.googleapis.com/bucket/{COG_DIR}/{MOSAIC_NAME}_cog.tif',
        )

    def test_missing_outputs_have_no_link(self):
        self.assertIsNone(self.link('GCS', False, False, False))

    def test_asset_row_links_to_the_gee_asset(self):
        self.assertEqual(
            self.link('ASSET', False, False, True),
            f'https://code.earthengine.google.com/?asset={ASSET_COLLECTION}/{MOSAIC_NAME}',
        )

    def test_asset_row_without_asset_has_no_link(self):
        self.assertIsNone(self.link('ASSET', False, True, False))


class CacheFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.ui = ExportDispatcherUI.__new__(ExportDispatcherUI)

    def test_missing_timestamp_is_stale(self):
        self.ui.state = {}
        self.assertTrue(self.ui._cache_is_stale())

    def test_old_timestamp_is_stale(self):
        stamp = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=301)
        self.ui.state = {'updated_at': stamp.isoformat()}
        self.assertTrue(self.ui._cache_is_stale())

    def test_recent_timestamp_is_fresh(self):
        stamp = datetime.datetime.now(datetime.timezone.utc)
        self.ui.state = {'updated_at': stamp.isoformat()}
        self.assertFalse(self.ui._cache_is_stale())


if __name__ == '__main__':
    unittest.main()
