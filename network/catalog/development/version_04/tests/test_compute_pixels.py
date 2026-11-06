"""Offline checks of High Volume transport; no GEE credentials required."""
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_compute_pixels import (compute_pixels, validate_request_size, make_tile_request,
                              download_adaptive_tile, RestrictedModeError)
from M_export_grid import Tile


class ComputePixelsTests(unittest.TestCase):
    def test_memory_error_subdivides_and_reconstructs_parent_grid(self):
        import numpy as np
        import rasterio
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin

        request = make_tile_request({'expression': 'mock'}, Tile(0, 0, 0, 0, 4, 4),
                                    1, 4, 'EPSG:4326', [2])
        calls = []

        def fake_fetch(_project, _credentials, req):
            grid = req['grid']
            size = grid['dimensions']['width']
            x = grid['affineTransform']['translateX']
            y = grid['affineTransform']['translateY']
            calls.append((size, x, y))
            if size > 2:
                raise RuntimeError('HTTP 400: User memory limit exceeded.')
            with MemoryFile() as memory:
                with memory.open(driver='GTiff', width=size, height=size, count=1,
                                 dtype='int16', crs='EPSG:4326', nodata=0,
                                 transform=from_origin(x, y, 1, 1)) as ds:
                    ds.write(np.full((size, size), int(x + y * 10), dtype='int16'), 1)
                return memory.read()

        with tempfile.TemporaryDirectory() as tmp:
            path = str(pathlib.Path(tmp) / 'parent.tif')
            download_adaptive_tile('project', object(), request, path, fetch=fake_fetch,
                                   min_pixels=2, dtype='int16')
            with rasterio.open(path) as ds:
                self.assertEqual(ds.read(1).tolist(), [[40, 40, 42, 42], [40, 40, 42, 42],
                                                      [20, 20, 22, 22], [20, 20, 22, 22]])
                self.assertEqual(ds.transform, from_origin(0, 4, 1, 1))
                self.assertEqual(ds.dtypes[0], 'int16')
        self.assertEqual(len(calls), 5)

    def test_non_memory_failure_is_not_subdivided(self):
        request = make_tile_request({}, Tile(0, 0, 0, 0, 4, 4), 1, 4, 'EPSG:4326', [1])
        fetch = Mock(side_effect=RuntimeError('HTTP 403: denied'))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(RuntimeError, '403'):
            download_adaptive_tile('project', object(), request, str(pathlib.Path(tmp) / 'x.tif'),
                                   fetch=fetch, min_pixels=2)
        fetch.assert_called_once()

    def test_known_request_limit_skips_the_too_large_parent(self):
        import numpy as np
        from rasterio.io import MemoryFile
        from rasterio.transform import from_origin
        request = make_tile_request({}, Tile(0, 0, 0, 0, 4, 4), 1, 4, 'EPSG:4326', [1])
        sizes = []
        def fake_fetch(_project, _credentials, req):
            g = req['grid']
            size = g['dimensions']['width']
            sizes.append(size)
            with MemoryFile() as mem:
                with mem.open(driver='GTiff', width=size, height=size, count=1, dtype='uint8',
                              crs='EPSG:4326', transform=from_origin(
                                  g['affineTransform']['translateX'],
                                  g['affineTransform']['translateY'], 1, 1)) as ds:
                    ds.write(np.ones((size, size), dtype='uint8'), 1)
                return mem.read()
        with tempfile.TemporaryDirectory() as tmp:
            download_adaptive_tile('p', object(), request, str(pathlib.Path(tmp) / 'x.tif'),
                                   fetch=fake_fetch, min_pixels=2, request_max_pixels=2)
        self.assertEqual(sizes, [2, 2, 2, 2])

    def test_request_uses_north_up_aligned_tile(self):
        request = make_tile_request({'expression': 'mock'}, Tile(1, 2, 100, 200, 140, 240),
                                    10, 4, 'EPSG:3857', [1])
        grid = request['grid']
        self.assertEqual(grid['dimensions'], {'width': 4, 'height': 4})
        self.assertEqual(grid['affineTransform'], {'scaleX': 10, 'scaleY': -10,
                         'translateX': 100, 'translateY': 240})
        with self.assertRaises(ValueError):
            make_tile_request({}, Tile(1, 2, 100, 200, 150, 240), 10, 4, 'EPSG:3857', [1])

    def test_size_limits(self):
        validate_request_size(4096, 4096, [1, 2])
        with self.assertRaises(ValueError):
            validate_request_size(4096, 4096, [1, 2, 1])
        with self.assertRaises(ValueError):
            validate_request_size(32769, 1, [1])

    def test_retry_and_credentials(self):
        credentials = object()
        session = Mock()
        first = Mock(status_code=429)
        second = Mock(status_code=200, content=b'tile')
        session.post.side_effect = [first, second]
        factory, sleep = Mock(return_value=session), Mock()
        self.assertEqual(compute_pixels('project', credentials, {'grid': {}},
                         session_factory=factory, sleep=sleep, jitter=lambda: 0), b'tile')
        factory.assert_called_once_with(credentials)
        self.assertIn('earthengine-highvolume.googleapis.com', session.post.call_args.args[0])
        self.assertEqual(session.post.call_args.kwargs['headers']['X-Goog-User-Project'], 'project')
        sleep.assert_called_once_with(1)
        session.close.assert_called_once()

    def test_nontransient_error_is_not_retried(self):
        session = Mock()
        response = Mock(status_code=400)
        response.raise_for_status.side_effect = RuntimeError('bad request')
        session.post.return_value = response
        with self.assertRaises(RuntimeError):
            compute_pixels('project', object(), {}, session_factory=lambda _: session)
        session.post.assert_called_once()
        session.close.assert_called_once()

    def test_server_error_detail_is_reported(self):
        from requests.exceptions import HTTPError
        session = Mock()
        response = Mock(status_code=400)
        response.raise_for_status.side_effect = HTTPError('bad request')
        response.json.return_value = {'error': {'message': 'Computation timed out'}}
        session.post.return_value = response
        with self.assertRaisesRegex(RuntimeError, 'Computation timed out'):
            compute_pixels('project', object(), {}, session_factory=lambda _: session)

    def test_restricted_mode_is_signaled_without_repeated_parallel_retries(self):
        session = Mock()
        response = Mock(status_code=429)
        response.json.return_value = {'error': {'message': 'Your project is in Restricted Mode'}}
        session.post.return_value = response
        sleep = Mock()
        with self.assertRaises(RestrictedModeError):
            compute_pixels('project', object(), {}, session_factory=lambda _: session, sleep=sleep)
        session.post.assert_called_once()
        sleep.assert_not_called()
