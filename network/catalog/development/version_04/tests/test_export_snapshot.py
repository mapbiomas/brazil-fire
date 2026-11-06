"""Snapshot tests without authenticated EE access."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_export_snapshot import pin_expression


class SnapshotTests(unittest.TestCase):
    def test_pins_loads_without_mutating_input_or_existing_version(self):
        tree = {'values': [{'functionInvocationValue': {'functionName': 'Image.load',
                 'arguments': {'id': {'constantValue': 'x'}}}},
                {'functionInvocationValue': {'functionName': 'ImageCollection.load',
                 'arguments': {'version': {'constantValue': 42}}}}]}
        pinned = pin_expression(tree, 12345)
        self.assertEqual(pinned['values'][0]['functionInvocationValue']['arguments']['version'],
                         {'constantValue': 12345})
        self.assertNotIn('version', tree['values'][0]['functionInvocationValue']['arguments'])
        self.assertEqual(pinned['values'][1]['functionInvocationValue']['arguments']['version'],
                         {'constantValue': 42})
        self.assertEqual(pin_expression(pinned, 12345), pinned)

    def test_unpinnable_source_is_rejected(self):
        for name in ('FeatureCollection.loadBigQuery', 'Image.loadGeoTIFF'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                pin_expression({'functionInvocationValue': {'functionName': name}}, 12345)

    def test_timestamp_range(self):
        for stamp in (0, -1, 10**18, True):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                pin_expression({}, stamp)
