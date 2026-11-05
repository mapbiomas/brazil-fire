"""Offline tests for the GCS cache contract consumed by M1/M2/M4/M5.

The Peru cache is a single state.json. It derives both gcs_chunks and cogs_*
from one GCS scan. After M2 deletes the chunks, gcs_chunks must be empty while
cogs_* keeps the COG.
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M_cache
import M_gcs
from M_cache import CacheManager


COUNTRY = 'testland'
BUCKET = 'bucket'
CAMPAIGN = 'MONITOR_01'
PREFIX = f'sudamerica/{COUNTRY}/CATALOG_01'
LIBRARY = f'{PREFIX}/LIBRARY_IMAGES'

CHUNK = (
    f'{BUCKET}/{LIBRARY}/SENTINEL2/MONTHLY/MINNBR/2025_06/CHUNKS/'
    f'image_{COUNTRY}_fire_sentinel2_minnbr_blue_2025_06'
    '_0000000000-0000000000.tif'
)
COG = (
    f'{BUCKET}/{LIBRARY}/SENTINEL2/MONTHLY/MINNBR/2025_06/COG/'
    f'image_{COUNTRY}_fire_sentinel2_minnbr_blue_2025_06_cog.tif'
)

CONFIG = {
    'country': COUNTRY,
    'bucket': BUCKET,
    'campaign': CAMPAIGN,
    'gcs_library_images_prefix': PREFIX,
    'gcs_library_images': LIBRARY,
    'gcs_campaigns_prefix': PREFIX,
    'gcs_cache': f'{PREFIX}/.CACHE',
    'gee_library_images_prefix': f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
    'gee_campaigns_prefix': f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
    'mosaic_methods': ['minnbr'],
}


class FakeFS:
    def __init__(self, paths):
        self.paths = list(paths)

    def find(self, prefix):
        return list(self.paths)

    def exists(self, path):
        return False


def scan_gcs(paths, callback=None):
    old_cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        with patch.object(M_cache, 'CONFIG', CONFIG), \
             patch.object(M0_auth_config, 'CONFIG', CONFIG), \
             patch.object(M_cache, '_get_fs', return_value=FakeFS(paths)), \
             patch.object(M_gcs, 'write_json', lambda *args, **kwargs: None):
            CacheManager.reset()
            state = CacheManager.build_cache_from_gcs()
            return callback(state) if callback else state
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        CacheManager.reset()


class GcsCacheContractTests(unittest.TestCase):
    def test_chunk_scan_fills_gcs_chunks(self):
        state = scan_gcs([CHUNK])
        self.assertEqual(
            state['gcs_chunks'],
            {f'image_{COUNTRY}_fire_sentinel2_minnbr_2025_06': ['blue']},
        )
        self.assertEqual(state['cogs_monthly'], [])

    def test_cog_only_scan_is_the_post_m2_state(self):
        state = scan_gcs([COG])
        self.assertEqual(state['gcs_chunks'], {})
        self.assertIn(
            f'image_{COUNTRY}_fire_sentinel2_minnbr_blue_2025_06',
            state['cogs_monthly'],
        )

    def test_scan_keeps_both_outputs_when_they_coexist(self):
        state = scan_gcs([CHUNK, COG])
        self.assertEqual(
            state['gcs_chunks'],
            {f'image_{COUNTRY}_fire_sentinel2_minnbr_2025_06': ['blue']},
        )
        self.assertIn(
            f'image_{COUNTRY}_fire_sentinel2_minnbr_blue_2025_06',
            state['cogs_monthly'],
        )

    def test_cogs_for_reads_the_expected_periodicity_key(self):
        monthly = scan_gcs(
            [COG], callback=lambda state: CacheManager.cogs_for('monthly')
        )
        self.assertIn(
            f'image_{COUNTRY}_fire_sentinel2_minnbr_blue_2025_06',
            monthly,
        )
        self.assertEqual(
            scan_gcs([COG], callback=lambda state: CacheManager.cogs_for('yearly')),
            [],
        )


class StaleCachePersistenceTests(unittest.TestCase):
    def test_mark_stale_persists_across_reload(self):
        old_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        writes = []
        try:
            os.chdir(tmp)
            with patch.object(M_cache, 'CONFIG', CONFIG), \
                 patch.object(M0_auth_config, 'CONFIG', CONFIG), \
                 patch.object(M_cache, '_get_fs', return_value=FakeFS([])), \
                 patch.object(
                     M_gcs,
                     'write_json',
                     side_effect=lambda path, data: writes.append(copy.deepcopy(data)),
                 ):
                CacheManager.reset()
                state = CacheManager.get_state()
                CacheManager.save(state)
                self.assertIsNotNone(writes[-1]['updated_at'])

                CacheManager.mark_stale()

                with open(CacheManager.CACHE_FILE, encoding='utf-8') as handle:
                    local = json.load(handle)
                self.assertIsNone(local['updated_at'])
                self.assertIsNone(writes[-1]['updated_at'])

                CacheManager.reset()
                reloaded = CacheManager.load(force=True)
                self.assertIsNone(reloaded['updated_at'])
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(tmp, ignore_errors=True)
            CacheManager.reset()


if __name__ == '__main__':
    unittest.main()
