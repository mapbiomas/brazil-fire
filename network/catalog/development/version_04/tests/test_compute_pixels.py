"""Offline checks of High Volume transport; no GEE credentials required."""
import pathlib
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_compute_pixels import compute_pixels, validate_request_size, make_tile_request
from M_export_grid import Tile


class ComputePixelsTests(unittest.TestCase):
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
