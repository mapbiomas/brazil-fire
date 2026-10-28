"""Offline tests for M2 COG assembly.

Covers the output dtype contract and the source-chunk cleanup that runs after
a successful COG upload. GDAL/GCS calls are mocked; no network is used.
"""

import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M2_mosaic_logic as logic
import M_gcs
from M0_auth_config import set_global_opts
from M2_mosaic_logic import BAND_DATATYPES, assemble_country_mosaic


COUNTRY = 'testland'
BUCKET = 'bucket'
CHUNK_PREFIX = (
    f'sudamerica/{COUNTRY}/CATALOG_01/LIBRARY_IMAGES/'
    'SENTINEL2/MONTHLY/MINNBR/2025_06/CHUNKS'
)


def shard(band):
    return (
        f'gs://{BUCKET}/{CHUNK_PREFIX}/'
        f'image_{COUNTRY}_fire_sentinel2_minnbr_{band}_2025_06'
        '_0000000000-0000000000.tif'
    )


def fake_download(remote, local):
    with open(local, 'wb') as handle:
        handle.write(b'x')


@contextmanager
def assembly_env(shards, upload_error=None, rm_error=None):
    tmp = tempfile.mkdtemp()
    upload_mock = MagicMock(side_effect=upload_error)
    rm_mock = MagicMock(side_effect=rm_error)
    try:
        with patch.object(logic, 'list_gcs_files', return_value=shards), \
             patch.object(logic, 'download', side_effect=fake_download), \
             patch.object(logic, 'run_cmd', return_value=None) as run_cmd_mock, \
             patch.object(logic, 'upload', upload_mock), \
             patch.object(logic, 'rm_files', rm_mock), \
             patch.object(logic, 'check_m2_dependencies', return_value=[]), \
             patch.object(logic, 'get_temp_dir', return_value=tmp):
            yield upload_mock, rm_mock, run_cmd_mock
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def assemble(bands=('blue',)):
    return assemble_country_mosaic(
        year=2025,
        month=6,
        period='monthly',
        sensor='sentinel2',
        mosaic_method='minnbr',
        bands=list(bands),
        period_key='2025_06',
        logger=None,
    )


def translate_dtype(run_cmd_mock):
    calls = [
        call for call in run_cmd_mock.call_args_list
        if call.args and call.args[0] and call.args[0][0] == 'gdal_translate'
    ]
    assert calls, 'gdal_translate was not called'
    args = calls[0].args[0]
    return args[args.index('-ot') + 1]


class AssemblyDtypeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_opts(
            country=COUNTRY,
            gcs_bucket=BUCKET,
            gcs_library_images_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gcs_campaigns_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gee_project=f'mapbiomas-{COUNTRY}',
            gee_library_images_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            gee_campaigns_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            campaign='MONITOR_01',
            asset_regions='regions',
            sensor=['sentinel2'],
            periodicity=['monthly'],
            mosaic_methods=['minnbr'],
            language='en',
        )
        # M2 imported CONFIG before the first set_global_opts() bound it.
        logic.CONFIG = M0_auth_config.CONFIG

    def test_spectral_bands_are_byte(self):
        for band in ('blue', 'green', 'red', 'nir', 'swir1', 'swir2'):
            self.assertEqual(BAND_DATATYPES[band], 'Byte')

    def test_day_of_year_is_int16(self):
        self.assertEqual(BAND_DATATYPES['dayOfYear'], 'Int16')

    def test_translate_uses_byte_for_spectral_band(self):
        with assembly_env([shard('blue')]) as (_, _, run_cmd_mock):
            assemble(['blue'])
        self.assertEqual(translate_dtype(run_cmd_mock), 'Byte')

    def test_translate_uses_int16_for_day_of_year(self):
        with assembly_env([shard('dayOfYear')]) as (_, _, run_cmd_mock):
            assemble(['dayOfYear'])
        self.assertEqual(translate_dtype(run_cmd_mock), 'Int16')


class AssemblyChunkCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_global_opts(
            country=COUNTRY,
            gcs_bucket=BUCKET,
            gcs_library_images_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gcs_campaigns_prefix=f'sudamerica/{COUNTRY}/CATALOG_01',
            gee_project=f'mapbiomas-{COUNTRY}',
            gee_library_images_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            gee_campaigns_prefix=f'projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01',
            campaign='MONITOR_01',
            asset_regions='regions',
            sensor=['sentinel2'],
            periodicity=['monthly'],
            mosaic_methods=['minnbr'],
            language='en',
        )
        logic.CONFIG = M0_auth_config.CONFIG

    def test_only_the_assembled_band_is_cleaned(self):
        with assembly_env([shard('blue'), shard('red')]) as (upload_mock, rm_mock, _):
            result = assemble(['blue'])

        self.assertEqual(len(result), 1)
        upload_mock.assert_called_once()
        rm_mock.assert_called_once_with([shard('blue')])

    def test_cog_is_uploaded_before_chunks_are_removed(self):
        with assembly_env([shard('blue')]) as (upload_mock, rm_mock, _):
            assemble(['blue'])

        dest = upload_mock.call_args.args[1]
        self.assertTrue(
            dest.endswith(
                '/COG/image_testland_fire_sentinel2_minnbr_blue_2025_06_cog.tif'
            ),
            dest,
        )
        rm_mock.assert_called_once_with([shard('blue')])

    def test_upload_failure_keeps_the_chunks(self):
        with assembly_env(
            [shard('blue')], upload_error=RuntimeError('upload failed')
        ) as (_, rm_mock, _):
            with self.assertRaises(RuntimeError):
                assemble(['blue'])

        rm_mock.assert_not_called()

    def test_cleanup_failure_does_not_lose_the_cog(self):
        with assembly_env(
            [shard('blue')], rm_error=RuntimeError('rm failed')
        ) as (_, rm_mock, _):
            result = assemble(['blue'])

        self.assertEqual(len(result), 1)
        rm_mock.assert_called_once()

    def test_no_chunks_means_no_assembly_and_no_cleanup(self):
        with assembly_env([]) as (upload_mock, rm_mock, _):
            result = assemble(['blue'])

        self.assertEqual(result, [])
        upload_mock.assert_not_called()
        rm_mock.assert_not_called()


class RemoveFilesTests(unittest.TestCase):
    def test_removes_objects_in_batches(self):
        with patch.object(M_gcs, '_call') as call:
            M_gcs.rm_files(['gs://bucket/a.tif', 'gs://bucket/b.tif'], batch_size=1)

        self.assertEqual(call.call_count, 2)
        call.assert_any_call(['rm', '-f', 'gs://bucket/a.tif'])
        call.assert_any_call(['rm', '-f', 'gs://bucket/b.tif'])

    def test_empty_list_is_a_noop(self):
        with patch.object(M_gcs, '_call') as call:
            M_gcs.rm_files([])
        call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
