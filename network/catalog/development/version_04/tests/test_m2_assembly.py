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
             patch.object(logic, 'mask_raster_to_country', return_value='masked.tif') as mask_mock, \
             patch.object(logic, '_validate_masked_cog'), \
             patch.object(logic, '_country_geometry_geojson', return_value={
                 'type': 'Polygon',
                 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
             }), \
             patch('M1_update_manifest.complete_gcs_item') as manifest_complete_mock, \
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

    def test_parallel_assemblies_allocate_unique_temporary_workdirs(self):
        from concurrent.futures import ThreadPoolExecutor

        real_mkdtemp = tempfile.mkdtemp
        allocated = []

        def track_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            allocated.append(path)
            return path

        with patch.object(tempfile, 'mkdtemp', side_effect=track_mkdtemp), \
             assembly_env([shard('blue'), shard('green')]) as (_, _, _):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda band: assemble((band,)), ('blue', 'green'),
                ))

        # One shared test root plus one isolated temporary root per M2 job.
        self.assertEqual(len(allocated), 3)
        self.assertEqual(len(set(allocated[1:])), 2)
        self.assertTrue(all(results))

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


class AssemblyRasterRegressionTests(unittest.TestCase):
    """Exercise M2 orchestration with local rasters, not real GCS or Earth Engine."""

    @classmethod
    def setUpClass(cls):
        AssemblyDtypeTests.setUpClass()

    def test_chunks_flow_through_mask_and_cog_before_upload_and_cleanup(self):
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.shutil import copy
        from rasterio.transform import from_origin

        chunks = [shard('blue'), shard('blue').replace('0000000000-0000000000',
                                                       '0000000000-0000000002')]
        geometry = {'type': 'Polygon',
                    'coordinates': [[[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]]]}
        events = []

        with tempfile.TemporaryDirectory() as tmp:
            sources = {}
            for i, remote in enumerate(chunks):
                path = os.path.join(tmp, f'input_{i}.tif')
                with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                    with rasterio.open(path, 'w', driver='GTiff', width=2, height=2,
                                       count=1, dtype='uint8', crs='EPSG:4326',
                                       transform=from_origin(i * 2, 2, 1, 1)) as dst:
                        dst.write(np.array([[0, 7], [8, 9]] if i == 0 else
                                           [[10, 11], [12, 13]], dtype='uint8'), 1)
                        dst.write_mask(np.array([[255, 0], [255, 255]] if i == 0 else
                                                [[255, 255], [255, 255]], dtype='uint8'))
                sources[remote] = path

            def download_chunk(remote, local):
                shutil.copyfile(sources[remote], local)

            def command(args, label=None):
                events.append(args[0])
                # Explicit zero nodata would discard valid zero-valued pixels.
                self.assertNotIn('-srcnodata', args)
                self.assertNotIn('-vrtnodata', args)
                self.assertNotIn('-a_nodata', args)
                if args[0] == 'gdalbuildvrt':
                    # The local stand-in retains dataset masks when assembling tiles.
                    datasets = [rasterio.open(path) for path in args[2:]]
                    try:
                        data = np.concatenate([ds.read() for ds in datasets], axis=2)
                        mask = np.concatenate([ds.dataset_mask() for ds in datasets], axis=1)
                        profile = datasets[0].profile.copy()
                        profile.update(driver='GTiff', height=data.shape[1],
                                       width=data.shape[2])
                        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                            with rasterio.open(args[1], 'w', **profile) as dst:
                                dst.write(data)
                                dst.write_mask(mask)
                    finally:
                        for dataset in datasets:
                            dataset.close()
                elif args[0] == 'gdal_translate':
                    self.assertEqual(args[args.index('-of') + 1], 'COG')
                    copy(args[-2], args[-1], driver='COG', COMPRESS='LZW')
                else:
                    self.fail(f'Unexpected command: {args}')

            def country_mask(source, features, block_size):
                path = os.path.join(tmp, 'country.tif')
                profile = source.profile.copy()
                profile.update(driver='GTiff', count=1, dtype='uint8')
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(rasterize([feature['geometry'] for feature in features],
                                        out_shape=source.shape,
                                        transform=source.transform, dtype='uint8'), 1)
                return path

            real_mask = logic.mask_raster_to_country

            def apply_mask(*args, **kwargs):
                events.append('mask')
                return real_mask(*args, **kwargs)

            def upload_cog(path, destination):
                events.append('upload')
                with rasterio.open(path) as cog:
                    self.assertEqual(cog.tags(ns='IMAGE_STRUCTURE').get('LAYOUT'), 'COG')
                    self.assertEqual(cog.read(1)[0, 0], 0)
                    np.testing.assert_array_equal(cog.dataset_mask() > 0,
                                                  [[True, False, True, False],
                                                   [True, True, True, False]])

            def remove(remote):
                events.append('remove')
                self.assertEqual(remote, chunks)

            with patch.object(logic, 'list_gcs_files', return_value=chunks), \
                 patch.object(logic, 'download', side_effect=download_chunk), \
                 patch.object(logic, 'run_cmd', side_effect=command), \
                 patch.object(logic, '_cached_country_mask', side_effect=country_mask), \
                 patch.object(logic, '_country_geometry_geojson', return_value=geometry), \
                 patch.object(logic, 'mask_raster_to_country', side_effect=apply_mask), \
                 patch.object(logic, 'upload', side_effect=upload_cog), \
                 patch.object(logic, 'rm_files', side_effect=remove), \
                 patch.object(logic, 'check_m2_dependencies', return_value=[]), \
                 patch.object(logic, 'get_temp_dir', return_value=tmp), \
                 patch('M1_update_manifest.complete_gcs_item'):
                result = assemble(['blue'])

            self.assertEqual(len(result), 1)
            self.assertEqual(events, ['gdalbuildvrt', 'mask', 'gdal_translate', 'upload', 'remove'])

    def test_processing_failures_keep_chunks(self):
        for failing_step in ('gdalbuildvrt', 'mask', 'gdal_translate', 'validation'):
            with self.subTest(step=failing_step), \
                 assembly_env([shard('blue')]) as (upload_mock, rm_mock, run_cmd_mock):
                def fail_command(args, label=None):
                    if args[0] == failing_step:
                        raise RuntimeError(failing_step)

                run_cmd_mock.side_effect = fail_command
                with patch.object(logic, 'mask_raster_to_country',
                                  side_effect=RuntimeError(failing_step) if failing_step == 'mask'
                                  else None), \
                     patch.object(logic, '_validate_masked_cog',
                                  side_effect=RuntimeError(failing_step) if failing_step == 'validation'
                                  else None):
                    with self.assertRaisesRegex(RuntimeError, failing_step):
                        assemble(['blue'])
                upload_mock.assert_not_called()
                rm_mock.assert_not_called()


class RemoveFilesTests(unittest.TestCase):
    def test_removes_objects_in_batches(self):
        client = MagicMock()
        bucket = MagicMock()
        client.bucket.return_value = bucket
        with patch.object(M_gcs, '_storage_client', return_value=client):
            M_gcs.rm_files(['gs://bucket/a.tif', 'gs://bucket/b.tif'], batch_size=1)

        self.assertEqual(bucket.delete_blobs.call_count, 2)

    def test_empty_list_is_a_noop(self):
        client = MagicMock()
        with patch.object(M_gcs, '_storage_client', return_value=client):
            M_gcs.rm_files([])
        client.bucket.assert_not_called()


if __name__ == '__main__':
    unittest.main()
