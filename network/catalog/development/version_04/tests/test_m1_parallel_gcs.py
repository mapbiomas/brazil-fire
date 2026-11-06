"""Offline interruption and resumption of the local GCS tile exporter."""
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M1_parallel_gcs as logic
import M_gcs


class ParallelResumeTests(unittest.TestCase):
    def test_restricted_mode_retries_the_tile_with_lower_concurrency(self):
        state, objects, logs, attempts = {}, set(), [], []
        geometry = MagicMock()
        geometry.bounds.return_value.coordinates.return_value.getInfo.return_value = [
            [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]
        storage = MagicMock()
        storage.list_blobs.return_value = []

        def fetch(*_args, **_kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise logic.RestrictedModeError('Restricted Mode')
            return b'tile'

        with patch.object(logic.ee.serializer, 'encode', return_value={'constant': 1}), \
             patch.object(logic.ee.data, '_get_state', return_value=MagicMock(credentials=object())), \
             patch.object(logic.L, 'ETA_RESTRICTED', 'restricted mode: concurrency one'), \
             patch.object(logic, 'compute_pixels', side_effect=fetch), \
             patch.object(logic.time, 'sleep'), \
             patch.object(M_gcs, 'blob_exists', side_effect=lambda path: path in state), \
             patch.object(M_gcs, 'read_json', side_effect=lambda path: state[path]), \
             patch.object(M_gcs, 'write_json', side_effect=lambda path, value: state.update({path: value.copy()})), \
             patch.object(M_gcs, 'upload', side_effect=lambda local, dest: objects.add(dest)), \
             patch.object(M_gcs, '_storage_client', return_value=storage):
            result = logic.export_band(MagicMock(), geometry, 'blue', 'base', 'folder/CHUNKS',
                {'bucket': 'bucket', 'gee_project': 'project'}, 111319.49079327358,
                workers=2, tile_pixels=2, logger=logs.append)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(attempts), 2)
        self.assertTrue(any('concurrency' in line.lower() or 'concurr' in line.lower() for line in logs))

    def test_repeated_failures_stop_early_and_are_reported(self):
        geometry, logs = MagicMock(), []
        geometry.bounds.return_value.coordinates.return_value.getInfo.return_value = [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]
        storage = MagicMock()
        storage.list_blobs.return_value = []
        with patch.object(logic.ee.serializer, 'encode', return_value={'constant': 1}), \
             patch.object(logic.ee.data, '_get_state', return_value=MagicMock(credentials=object())), \
             patch.object(logic, 'compute_pixels', side_effect=RuntimeError('EE timeout')) as pixels, \
             patch.object(M_gcs, 'blob_exists', return_value=False), \
             patch.object(M_gcs, 'write_json'), \
             patch.object(M_gcs, '_storage_client', return_value=storage):
            with self.assertRaises(RuntimeError):
                logic.export_band(MagicMock(), geometry, 'blue', 'base', 'folder/CHUNKS',
                                  {'bucket': 'bucket', 'gee_project': 'project'},
                                  111319.49079327358, workers=1, tile_pixels=2, logger=logs.append)
        self.assertEqual(pixels.call_count, 3)
        self.assertTrue(any('EE timeout' in line for line in logs))

    def test_interrupted_export_reuses_uploaded_tiles(self):
        state, objects, failures = {}, set(), [True]
        geometry = MagicMock()
        geometry.bounds.return_value.coordinates.return_value.getInfo.return_value = [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]
        storage = MagicMock()
        storage.list_blobs.side_effect = lambda bucket, prefix: [
            SimpleNamespace(name=path) for path in objects]

        def exists(path):
            return path in state

        def write(path, value):
            import copy
            state[path] = copy.deepcopy(value)

        def upload(local, dest):
            objects.add(dest.split('gs://bucket/', 1)[1])

        def fetch(*args, **kwargs):
            if failures[0]:
                failures[0] = False
                raise RuntimeError('temporary failure')
            return b'tile'

        credentials = MagicMock()
        creds_state = MagicMock(credentials=credentials)
        args = (MagicMock(), geometry, 'blue', 'image_test_blue_2025',
                'folder/CHUNKS', {'bucket': 'bucket', 'gee_project': 'project'},
                111319.49079327358)
        with patch.object(logic.ee.serializer, 'encode', return_value={'constant': 1}), \
             patch.object(logic.ee.data, '_get_state', return_value=creds_state), \
             patch.object(logic, 'compute_pixels', side_effect=fetch) as pixels, \
             patch.object(M_gcs, 'blob_exists', side_effect=exists), \
             patch.object(M_gcs, 'read_json', side_effect=lambda path: state[path]), \
             patch.object(M_gcs, 'write_json', side_effect=write), \
             patch.object(M_gcs, 'upload', side_effect=upload), \
             patch.object(M_gcs, '_storage_client', return_value=storage):
            with self.assertRaises(RuntimeError):
                logic.export_band(*args, workers=1, tile_pixels=2, logger=lambda _: None)
            self.assertEqual(len(objects), 3)
            result = logic.export_band(*args, workers=1, tile_pixels=2, logger=lambda _: None)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(objects), 4)
        self.assertEqual(pixels.call_count, 5)
