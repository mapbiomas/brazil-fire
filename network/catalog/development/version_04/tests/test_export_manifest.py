"""Tests for rejecting incomplete or ambiguous tile sets."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_export_manifest import completed_tiles


class ExportManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {'version': 1, 'status': 'complete', 'tiles': [
            {'id': 'a', 'object': 'gs://bucket/a.tif'},
            {'id': 'b', 'object': 'gs://bucket/b.tif'},
        ]}

    def test_only_manifest_objects_are_selected(self):
        self.assertEqual(completed_tiles(self.manifest, [
            'gs://bucket/b.tif', 'gs://bucket/old.tif', 'gs://bucket/a.tif',
        ]), ['gs://bucket/a.tif', 'gs://bucket/b.tif'])

    def test_missing_tile_is_rejected(self):
        with self.assertRaises(ValueError):
            completed_tiles(self.manifest, ['gs://bucket/a.tif'])

    def test_incomplete_or_empty_is_rejected(self):
        for change in ({'status': 'running'}, {'tiles': []}, {'version': 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                completed_tiles({**self.manifest, **change}, [])

    def test_duplicate_id_or_object_is_rejected(self):
        for tile in ({'id': 'a', 'object': 'gs://bucket/b.tif'},
                     {'id': 'b', 'object': 'gs://bucket/a.tif'}):
            with self.subTest(tile=tile), self.assertRaises(ValueError):
                completed_tiles({**self.manifest, 'tiles': [self.manifest['tiles'][0], tile]}, [])


if __name__ == '__main__':
    unittest.main()
