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
    def command(args, label=None, temp_dir=None):
        if args[0] == 'gdalbuildvrt':
            with open(args[1], 'wb') as handle:
                handle.write(b'vrt')

    def mask(source, output, geometry):
        with open(output, 'wb') as handle:
            handle.write(b'masked')

    try:
        with patch.object(logic, 'list_gcs_files', return_value=shards), \
             patch.object(logic, 'download', side_effect=fake_download), \
             patch.object(logic, 'run_cmd', side_effect=command) as run_cmd_mock, \
             patch.object(logic, 'upload', upload_mock), \
             patch.object(logic, 'rm_files', rm_mock), \
             patch.object(logic, 'mask_raster_to_country', side_effect=mask) as mask_mock, \
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

    def test_previous_band_files_are_gone_before_next_band_download(self):
        paths = {}
        with assembly_env([shard('blue'), shard('green')]) as (upload_mock, rm_mock, cmd):
            original_command = cmd.side_effect

            def command(args, label=None, temp_dir=None):
                if args[0] == 'gdal_translate':
                    paths['blue' if 'blue' in label else 'green'] = args[-2:]
                    self.assertTrue(os.path.exists(args[-2]))
                    with open(args[-1], 'wb') as stream:
                        stream.write(b'cog')
                else:
                    original_command(args, label=label, temp_dir=temp_dir)

            def download(remote, local):
                if remote == shard('green'):
                    self.assertEqual(len(paths['blue']), 2)
                    self.assertTrue(all(not os.path.exists(p) for p in paths['blue']))
                fake_download(remote, local)

            def validate(masked, cog):
                self.assertTrue(os.path.exists(masked))
                self.assertTrue(os.path.exists(cog))

            def upload(cog, destination):
                self.assertTrue(os.path.exists(cog))

            cmd.side_effect = command
            upload_mock.side_effect = upload
            with patch.object(logic, 'download', side_effect=download), \
                 patch.object(logic, '_validate_masked_cog', side_effect=validate):
                result = assemble(['blue', 'green'])
            self.assertTrue(all(not os.path.exists(p) for p in paths['green']))
            self.assertEqual(len(result), 2)
            self.assertEqual(rm_mock.call_count, 2)

    def test_partial_band_files_removed_on_failure_before_job_cleanup(self):
        for failing_step in ('mask', 'translate', 'validation', 'upload'):
            with self.subTest(step=failing_step), \
                 assembly_env([shard('blue')]) as (upload_mock, rm_mock, cmd):
                created = []
                real_rmtree = shutil.rmtree
                original_command = cmd.side_effect

                def command(args, label=None, temp_dir=None):
                    if args[0] == 'gdal_translate':
                        created.append(args[-1])
                        with open(args[-1], 'wb') as stream:
                            stream.write(b'partial cog')
                        if failing_step == 'translate':
                            raise RuntimeError(failing_step)
                    else:
                        original_command(args, label=label, temp_dir=temp_dir)

                def mask(source, output, geometry):
                    created.append(output)
                    with open(output, 'wb') as stream:
                        stream.write(b'partial mask')
                    if failing_step == 'mask':
                        raise RuntimeError(failing_step)

                def cleanup(path, *args, **kwargs):
                    # At job cleanup, the per-band finally must already have
                    # removed every partial or completed band artifact.
                    if os.path.basename(path).startswith('m2_'):
                        self.assertTrue(all(not os.path.exists(p) for p in created))
                    return real_rmtree(path, *args, **kwargs)

                cmd.side_effect = command
                upload_mock.side_effect = RuntimeError(failing_step) if failing_step == 'upload' else None
                with patch.object(logic, 'mask_raster_to_country', side_effect=mask), \
                     patch.object(logic, '_validate_masked_cog',
                                  side_effect=RuntimeError(failing_step) if failing_step == 'validation' else None), \
                     patch.object(shutil, 'rmtree', side_effect=cleanup):
                    with self.assertRaisesRegex(RuntimeError, failing_step):
                        assemble(['blue'])
                self.assertTrue(created)
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

            def command(args, label=None, temp_dir=None):
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
                    self.assertFalse(os.path.exists(vrt))
                    self.assertTrue(all(not os.path.exists(p) for p in downloaded))
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
            real_validate = logic._validate_masked_cog
            downloaded = []
            vrt = None

            def apply_mask(*args, **kwargs):
                nonlocal vrt
                vrt = args[0]
                downloaded.extend(os.path.join(os.path.dirname(vrt), 'blue', os.path.basename(c))
                                  for c in chunks)
                self.assertTrue(all(os.path.exists(p) for p in downloaded))
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

            def validate(source, output):
                real_validate(source, output)
                events.append('validate')

            def remove(remote):
                events.append('remove')
                self.assertEqual(remote, chunks)

            with patch.object(logic, 'list_gcs_files', return_value=chunks), \
                 patch.object(logic, 'download', side_effect=download_chunk), \
                 patch.object(logic, 'run_cmd', side_effect=command), \
                 patch.object(logic, '_cached_country_mask', side_effect=country_mask), \
                 patch.object(logic, '_country_geometry_geojson', return_value=geometry), \
                 patch.object(logic, 'mask_raster_to_country', side_effect=apply_mask), \
                 patch.object(logic, '_validate_masked_cog', side_effect=validate), \
                 patch.object(logic, 'upload', side_effect=upload_cog), \
                 patch.object(logic, 'rm_files', side_effect=remove), \
                 patch.object(logic, 'check_m2_dependencies', return_value=[]), \
                 patch.object(logic, 'get_temp_dir', return_value=tmp), \
                 patch('M1_update_manifest.complete_gcs_item'):
                result = assemble(['blue'])

            self.assertEqual(len(result), 1)
            self.assertEqual(events, ['gdalbuildvrt', 'mask', 'gdal_translate', 'validate',
                                      'upload', 'remove'])

    def test_processing_failures_keep_chunks(self):
        for failing_step in ('gdalbuildvrt', 'mask', 'gdal_translate', 'validation'):
            with self.subTest(step=failing_step), \
                 assembly_env([shard('blue')]) as (upload_mock, rm_mock, run_cmd_mock):
                def fail_command(args, label=None, temp_dir=None):
                    if args[0] == failing_step:
                        raise RuntimeError(failing_step)
                    if args[0] == 'gdalbuildvrt':
                        with open(args[1], 'wb') as handle:
                            handle.write(b'vrt')

                run_cmd_mock.side_effect = fail_command
                with patch.object(logic, 'mask_raster_to_country',
                                  side_effect=RuntimeError(failing_step) if failing_step == 'mask'
                                  else lambda source, output, geometry: open(output, 'wb').close()), \
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


class LocalSpaceTests(unittest.TestCase):
    def test_existing_cog_local_source_removed_only_after_mask_before_preflight(self):
        events = []
        with tempfile.TemporaryDirectory() as root, \
             patch.object(logic, 'CONFIG', {'bucket': 'bucket', 'asset_regions': 'regions'}), \
             patch.object(logic, 'period_cog_path', return_value='cogs'), \
             patch.object(logic, 'mosaic_name_for_period', return_value='test'), \
             patch('M_gcs.blob_exists', return_value=True), \
             patch.object(logic, 'get_temp_dir', return_value=root), \
             patch.object(logic, 'check_m2_dependencies', return_value=[]), \
             patch.object(logic, '_country_geometry_geojson', return_value={}), \
             patch.object(logic, 'rm_files') as remove_remote:
            def download(remote, local):
                events.append('download')
                with open(local, 'wb') as stream:
                    stream.write(b'source')

            def mask(source, output, geometry):
                self.assertTrue(os.path.exists(source))
                with open(output, 'wb') as stream:
                    stream.write(b'masked')
                events.append('mask')

            def preflight(masked, directory, logger):
                self.assertFalse(os.path.exists(os.path.join(directory, 'test_cog.tif')))
                self.assertTrue(os.path.exists(masked))
                events.append('preflight')

            with patch.object(logic, 'download', side_effect=download), \
                 patch.object(logic, 'mask_raster_to_country', side_effect=mask), \
                 patch.object(logic, '_cog_space_preflight', side_effect=preflight), \
                 patch.object(logic, 'run_cmd', side_effect=lambda *a, **kw: events.append('convert')), \
                 patch.object(logic, '_validate_masked_cog', side_effect=lambda *a: events.append('validate')), \
                 patch.object(logic, 'upload', side_effect=lambda *a: events.append('upload')):
                self.assertTrue(logic.mask_existing_country_cog(
                    '2025_01', 'red', 'sentinel2', 'minnbr', update_manifest=False))
            remove_remote.assert_not_called()
        self.assertEqual(events, ['download', 'mask', 'preflight', 'convert', 'validate', 'upload'])

    def test_subprocess_temp_environment_is_isolated_between_threads(self):
        from concurrent.futures import ThreadPoolExecutor
        import json
        import threading

        before = dict(os.environ)
        barrier = threading.Barrier(2)
        with tempfile.TemporaryDirectory() as root:
            paths = [os.path.join(root, name) for name in ('a', 'b')]
            for path in paths:
                os.mkdir(path)

            def run(path):
                barrier.wait()
                result = logic.run_cmd(
                    [sys.executable, '-c',
                     "import json,os; print(json.dumps([os.environ[k] for k in "
                     "('CPL_TMPDIR','TMPDIR','TEMP','TMP')]))"],
                    temp_dir=path,
                )
                return json.loads(result.stdout)

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(run, paths))
        self.assertEqual(results, [[path] * 4 for path in paths])
        self.assertEqual(dict(os.environ), before)

    def test_failure_does_not_change_parent_environment(self):
        before = dict(os.environ)
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(RuntimeError):
                logic.run_cmd([sys.executable, '-c', 'import sys; sys.exit(1)'], temp_dir=path)
        self.assertEqual(dict(os.environ), before)

    def test_preflight_uses_default_reserve_and_masked_size(self):
        with tempfile.TemporaryDirectory() as path:
            masked = os.path.join(path, 'masked.tif')
            with open(masked, 'wb') as stream:
                stream.write(b'data')
            threshold = 2 * 1024**3 + 4
            with patch.object(M0_auth_config, 'CONFIG', {}), \
                 patch('shutil.disk_usage', return_value=shutil._ntuple_diskusage(0, 0, threshold - 1)):
                with self.assertRaisesRegex(RuntimeError, 'COG preflight heuristic') as caught:
                    logic._cog_space_preflight(masked, path)
                self.assertEqual(type(caught.exception).__name__, 'COGSpaceEstimateExceeded')
                for value in (threshold - 1, 2 * 1024**3, 4):
                    self.assertIn(str(value), str(caught.exception))
            with patch.object(M0_auth_config, 'CONFIG', {}), \
                 patch('shutil.disk_usage', return_value=shutil._ntuple_diskusage(0, 0, threshold)):
                logic._cog_space_preflight(masked, path)

    def test_preflight_configurable_reserve_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as path:
            masked = os.path.join(path, 'masked.tif')
            with open(masked, 'wb') as stream:
                stream.write(b'data')
            with patch.object(logic, 'CONFIG', {'m2_cog_reserve_bytes': 999}), \
                 patch.object(M0_auth_config, 'CONFIG', {'m2_cog_reserve_bytes': 0}), \
                 patch('shutil.disk_usage', return_value=shutil._ntuple_diskusage(0, 0, 4)):
                logic._cog_space_preflight(masked, path)
            for value in (-1, True, 1.5, '2', None):
                with self.subTest(value=value), \
                     patch.object(M0_auth_config, 'CONFIG', {'m2_cog_reserve_bytes': value}):
                    with self.assertRaises(ValueError):
                        logic._cog_space_preflight(masked, path)

    def test_preflight_missing_size_fails_closed(self):
        logger = MagicMock()
        with tempfile.TemporaryDirectory() as path, \
             patch.object(M0_auth_config, 'CONFIG', {'m2_cog_reserve_bytes': 10}), \
             patch('shutil.disk_usage', return_value=shutil._ntuple_diskusage(0, 0, 9)):
            with self.assertRaisesRegex(RuntimeError, 'Masked TIFF size unavailable'):
                logic._cog_space_preflight(os.path.join(path, 'missing'), path, logger)

    def test_preflight_unavailable_free_space_fails_closed(self):
        logger = MagicMock()
        with patch.object(M0_auth_config, 'CONFIG', {}), \
             patch('shutil.disk_usage', side_effect=OSError('unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'free-space check unavailable'):
                logic._cog_space_preflight('masked', 'work', logger)


if __name__ == '__main__':
    unittest.main()
