"""Regression tests for the session-local country raster cache."""
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M2_mosaic_logic as m2
import M0_auth_config as m0

POLYGON = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]]}


class CountryMaskCacheTests(unittest.TestCase):
    def setUp(self):
        config_patch = patch.object(m0, 'CONFIG', {'m2_country_mask_cache_gcs_prefix': False})
        config_patch.start()
        self.addCleanup(config_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.counter = 0
        self.lock = threading.Lock()

    def source(self, name, transform=None, width=4, height=4, crs='EPSG:4326', valid=True):
        path = os.path.join(self.temp.name, name)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
            path, 'w', driver='GTiff', width=width, height=height, count=1, dtype='uint8',
            crs=crs, transform=transform or from_origin(0.01, 4, 1, 1)
        ) as dst:
            dst.write(np.ones((height, width), dtype='uint8'), 1)
            dst.write_mask(np.full((height, width), 255 if valid else 0, dtype='uint8'))
        return path

    def burn(self, args, label=None):
        with self.lock:
            self.counter += 1
        with open(args[-2], encoding='utf-8') as f:
            features = json.load(f)['features']
        with rasterio.open(args[-1], 'r+') as dst:
            dst.write(rasterize([f['geometry'] for f in features], out_shape=dst.shape,
                                transform=dst.transform, dtype='uint8'), 1)

    def test_reuse_and_grid_geometry_invalidation(self):
        import M_gcs
        sources = [self.source('a.tif'), self.source('b.tif', valid=False),
                   self.source('c.tif', transform=from_origin(1, 4, 1, 1)),
                   self.source('d.tif', width=5), self.source('e.tif', crs='EPSG:3857'),
                   self.source('f.tif', height=5)]
        with patch.object(m2, 'run_cmd', side_effect=self.burn), \
             patch.object(M_gcs, '_storage_client') as storage_client:
            for i, source in enumerate(sources):
                m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'out{i}.tif'), POLYGON)
            changed = {'type': 'Polygon', 'coordinates': [[[0, 0], [3, 0], [3, 4], [0, 4], [0, 0]]]}
            m2.mask_raster_to_country(sources[0], os.path.join(self.temp.name, 'changed.tif'), changed)
            with patch.object(m2, '_MASK_CACHE_VERSION', m2._MASK_CACHE_VERSION + 1):
                m2.mask_raster_to_country(sources[0], os.path.join(self.temp.name, 'version.tif'), POLYGON)
        storage_client.assert_not_called()
        self.assertEqual(self.counter, 7)
        with rasterio.open(os.path.join(self.temp.name, 'out1.tif')) as result:
            self.assertFalse(result.dataset_mask().any())

    def test_parallel_same_key_and_retry_after_failure(self):
        import M_gcs
        source = self.source('src.tif', transform=from_origin(0.125, 4, 1, 1))
        def apply(i):
            return m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'out{i}.tif'), POLYGON)
        with patch.object(m2, 'run_cmd', side_effect=RuntimeError('burn failed')):
            with self.assertRaisesRegex(RuntimeError, 'burn failed'):
                apply('failed')
        with patch.object(m2, 'run_cmd', side_effect=self.burn), \
             patch.object(M_gcs, '_storage_client') as storage_client:
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(apply, range(4)))
        storage_client.assert_not_called()
        self.assertEqual(self.counter, 1)

    def test_persistent_reuse_and_corruption_fallback(self):
        import M_gcs
        objects = {}

        class Blob:
            def __init__(self, name):
                self.name = name
                self.metadata = None
                self.generation = None

            def reload(self, **kwargs):
                if self.name not in objects:
                    raise FileNotFoundError(self.name)
                data, self.metadata = objects[self.name]
                self.generation = 1
                import base64
                import google_crc32c
                self.crc32c = base64.b64encode(google_crc32c.value(data).to_bytes(4, 'big')).decode()

            def download_to_filename(self, path, **kwargs):
                self_kwargs.append(kwargs)
                with open(path, 'wb') as f:
                    f.write(objects[self.name][0])

            def upload_from_filename(self, path, **kwargs):
                self.assert_create(kwargs)
                if self.name in objects:
                    raise RuntimeError('precondition failed')
                with open(path, 'rb') as f:
                    objects[self.name] = (f.read(), dict(self.metadata))
                self.generation = 1

            def assert_create(self, kwargs):
                if kwargs.get('if_generation_match') != 0:
                    raise AssertionError('upload must be create-only')

        class Client:
            def bucket(self, name):
                self_name.append(name)
                return self

            def blob(self, name):
                return Blob(name)

        self_name, self_kwargs = [], []
        source = self.source('persistent.tif', transform=from_origin(0.375, 4, 1, 1))
        def new_session():
            if m2._mask_cache_dir is not None:
                m2._mask_cache_dir.cleanup()
            m2._mask_cache_dir = None
        with patch.object(m0, 'CONFIG', {'m2_country_mask_cache_gcs_prefix': 'gs://test-bucket/dedicated'}), \
             patch.object(M_gcs, '_storage_client', return_value=Client()), \
             patch.object(m2, '_start_mask_upload', side_effect=m2._upload_country_mask), \
             patch.object(m2, 'run_cmd', side_effect=self.burn):
            new_session()
            m2.mask_raster_to_country(source, os.path.join(self.temp.name, 'p1.tif'), POLYGON)
            self.assertEqual(len(objects), 1)
            new_session()
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda i: m2.mask_raster_to_country(
                    source, os.path.join(self.temp.name, f'p2-{i}.tif'), POLYGON), range(4)))
            self.assertEqual(self.counter, 1)
            self.assertEqual(len(self_kwargs), 1)
            self.assertEqual(self_kwargs[0]['if_generation_match'], 1)
            key = next(iter(objects))
            old = objects[key]
            objects[key] = (b'corrupt', old[1])
            new_session()
            m2.mask_raster_to_country(source, os.path.join(self.temp.name, 'p3.tif'), POLYGON)
            self.assertEqual(self.counter, 2)
            self.assertEqual(objects[key][0], b'corrupt')
        self.assertTrue(all(name == 'test-bucket' for name in self_name))

    def test_unavailable_and_malformed_remote_are_nonfatal(self):
        source = self.source('offline.tif', transform=from_origin(0.625, 4, 1, 1))
        class BrokenBlob:
            def reload(self, **kwargs):
                raise OSError('network down')
        with patch.object(m2, '_remote_mask_blob', return_value=BrokenBlob()), \
             patch.object(m2, 'run_cmd', side_effect=self.burn):
            m2.mask_raster_to_country(source, os.path.join(self.temp.name, 'offline-out.tif'), POLYGON)
        self.assertEqual(self.counter, 1)
        with patch.object(m0, 'CONFIG', {'m2_country_mask_cache_gcs_prefix': 'gs://bucket'}):
            with self.assertRaises(ValueError):
                m2._remote_mask_blob('key')

    def test_competing_create_keeps_local_cache(self):
        from google.api_core.exceptions import PreconditionFailed
        source = self.source('race.tif', transform=from_origin(0.875, 4, 1, 1))
        calls = []
        class RaceBlob:
            def reload(self, **kwargs):
                raise FileNotFoundError()

            def upload_from_filename(self, path, **kwargs):
                calls.append(kwargs)
                raise PreconditionFailed('another writer created the object')
        with patch.object(m2, '_remote_mask_blob', return_value=RaceBlob()), \
             patch.object(m2, '_start_mask_upload', side_effect=m2._upload_country_mask), \
             patch.object(m2, 'run_cmd', side_effect=self.burn):
            for i in range(2):
                m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'race-{i}.tif'), POLYGON)
        self.assertEqual(self.counter, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['if_generation_match'], 0)

    def test_default_prefix_tracks_m0_reset_and_explicit_opt_out(self):
        import M_gcs
        from unittest.mock import MagicMock
        client = MagicMock()
        def config(**kwargs):
            return {'bucket': kwargs['bucket'], 'campaign': kwargs['campaign'],
                    'mosaic_methods': kwargs['mosaic_methods']}
        options = dict(country='test', gcs_bucket='first-bucket', gcs_library_images_prefix='images',
                       gcs_campaigns_prefix='campaigns', gee_project='project',
                       gee_library_images_prefix='images', gee_campaigns_prefix='campaigns',
                       campaign='campaign', asset_regions='regions', mosaic_methods=['minnbr'])
        with patch.object(m0, 'CONFIG', None), patch.object(m0, '_make_config', side_effect=config), \
             patch.object(M_gcs, '_storage_client', return_value=client):
            m0.set_global_opts(**options)
            m2._remote_mask_blob('abc')
            client.bucket.assert_called_with('first-bucket')
            client.bucket.return_value.blob.assert_called_with(
                'temp/registro_de_testes_de_mosaico/m2_country_masks/v1/abc.tif')
            m0.CONFIG['m2_country_mask_cache_gcs_prefix'] = False
            self.assertIsNone(m2._remote_mask_blob('abc'))
            options['gcs_bucket'] = 'second-bucket'
            m0.set_global_opts(**options)
            m2._remote_mask_blob('abc')
            client.bucket.assert_called_with('second-bucket')

    def test_upload_integrity_checks_server_crc_without_delete(self):
        import base64
        import google_crc32c
        source = self.source('crc.tif', transform=from_origin(1.125, 4, 1, 1))
        class Blob:
            generation = None
            def reload(self, **kwargs):
                if not hasattr(self, 'uploaded'):
                    raise FileNotFoundError()
                self.crc32c = base64.b64encode(google_crc32c.value(self.uploaded).to_bytes(4, 'big')).decode()

            def upload_from_filename(self, path, **kwargs):
                with open(path, 'rb') as f:
                    self.uploaded = f.read()
                self.kwargs = kwargs
                self.generation = 1
        blob = Blob()
        with patch.object(m2, '_remote_mask_blob', return_value=blob), \
             patch.object(m2, '_start_mask_upload', side_effect=m2._upload_country_mask), \
             patch.object(m2, 'run_cmd', side_effect=self.burn):
            m2.mask_raster_to_country(source, os.path.join(self.temp.name, 'crc-out.tif'), POLYGON)
        self.assertEqual(blob.kwargs['if_generation_match'], 0)
        self.assertTrue(hasattr(blob, 'crc32c'))
        blob.crc32c = 'incorrect'
        def bad_reload(**kwargs):
            self.assertEqual(kwargs['if_generation_match'], 1)
        blob.reload = bad_reload
        path = os.path.join(self.temp.name, 'crc-bytes')
        with open(path, 'wb') as f:
            f.write(b'bytes')
        with self.assertRaisesRegex(ValueError, 'CRC32C mismatch'):
            m2._verify_uploaded_mask(blob, path)

    def test_upload_retry_is_nonblocking_and_reuses_mask_with_concurrent_hits(self):
        source = self.source('async-retry.tif', transform=from_origin(1.375, 4, 1, 1))
        started, release, verified = threading.Event(), threading.Event(), threading.Event()
        calls = []
        class Blob:
            def reload(self, **kwargs):
                raise FileNotFoundError()
            def upload_from_filename(self, path, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    started.set()
                    release.wait(5)
                    raise OSError('temporary failure')
        blob = Blob()
        def apply(i):
            return m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'async-{i}.tif'), POLYGON)
        with patch.object(m2, '_remote_mask_blob', return_value=blob), \
             patch.object(m2, 'run_cmd', side_effect=self.burn), \
             patch.object(m2, '_verify_uploaded_mask', side_effect=lambda *a: verified.set()):
            with ThreadPoolExecutor(max_workers=5) as pool:
                try:
                    first = pool.submit(apply, 0)
                    self.assertTrue(started.wait(5))
                    first.result(timeout=1)
                    list(pool.map(apply, range(1, 5)))
                    self.assertEqual(self.counter, 1)
                finally:
                    release.set()
            self.assertTrue(verified.wait(5), 'transient failure was not retried')
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call['if_generation_match'] == 0 for call in calls))

    def test_upload_retry_is_bounded_and_does_not_overwrite(self):
        path = os.path.join(self.temp.name, 'existing-mask')
        with open(path, 'wb') as stream:
            stream.write(b'mask')
        calls = []
        class Blob:
            def upload_from_filename(self, path, **kwargs):
                calls.append(kwargs)
                raise OSError('still offline')
        with patch.object(m2, '_MASK_UPLOAD_RETRY_DELAYS', (0, 0)):
            with self.assertRaisesRegex(OSError, 'still offline'):
                m2._upload_country_mask(Blob(), path, 'key')
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call['if_generation_match'] == 0 and call['retry'] is None
                            for call in calls))

    def test_gcs_reads_and_verification_use_bounded_requests(self):
        from unittest.mock import MagicMock
        blob = MagicMock()
        blob.generation = 7
        blob.metadata = {}
        with self.assertRaisesRegex(ValueError, 'metadata'):
            m2._download_country_mask(blob, 'unused', None, 'key')
        blob.reload.assert_called_once_with(timeout=30, retry=None)

        import base64
        import google_crc32c
        path = os.path.join(self.temp.name, 'bounded-checksum')
        with open(path, 'wb') as stream:
            stream.write(b'payload')
        blob.reset_mock()
        blob.crc32c = base64.b64encode(google_crc32c.value(b'payload').to_bytes(4, 'big')).decode()
        m2._verify_uploaded_mask(blob, path)
        blob.reload.assert_called_once_with(if_generation_match=7, timeout=30, retry=None)
        with patch.object(m0, 'CONFIG', {'m2_country_mask_cache_gcs_timeout': 120}):
            blob.reset_mock()
            m2._verify_uploaded_mask(blob, path)
            blob.reload.assert_called_once_with(if_generation_match=7, timeout=120, retry=None)
        with patch.object(m0, 'CONFIG', {'m2_country_mask_cache_gcs_timeout': float('inf')}):
            with self.assertRaises(ValueError):
                m2._mask_gcs_request_timeout()

    def test_download_request_timeout_falls_back_locally(self):
        source = self.source('read-timeout.tif', transform=from_origin(1.625, 4, 1, 1))
        class Blob:
            generation = 3
            def reload(self, **kwargs):
                self.reload_kwargs = kwargs
                self.metadata = {'m2-mask-key': self.key, 'm2-mask-version': str(m2._MASK_CACHE_VERSION),
                                 'm2-mask-sha256': 'checksum'}
            def download_to_filename(self, path, **kwargs):
                self.download_kwargs = kwargs
                raise TimeoutError('read timed out')
        blob = Blob()
        def remote(key):
            blob.key = key
            return blob
        with patch.object(m2, '_remote_mask_blob', side_effect=remote), \
             patch.object(m2, 'run_cmd', side_effect=self.burn):
            m2.mask_raster_to_country(source, os.path.join(self.temp.name, 'timeout-out.tif'), POLYGON)
        self.assertEqual(self.counter, 1)
        self.assertEqual(blob.reload_kwargs, {'timeout': 30, 'retry': None})
        self.assertEqual(blob.download_kwargs, {'if_generation_match': 3, 'checksum': 'auto',
                                                'timeout': 30, 'retry': None})


if __name__ == '__main__':
    unittest.main()
