import unittest
from datetime import datetime, timezone

from gif_factory.engine import _collection_frames


def millis(year, month=1):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)


class CollectionResolutionTests(unittest.TestCase):
    def test_single_band_annual_collection(self):
        rows = [{"index": "b", "time": millis(2024), "bands": ["value"]},
                {"index": "a", "time": millis(2023), "bands": ["value"]}]
        self.assertEqual(_collection_frames(rows), [("year_2023", "a", "value"), ("year_2024", "b", "value")])

    def test_monthly_collection(self):
        rows = [{"index": "b", "time": millis(2024, 2), "bands": ["value"]},
                {"index": "a", "time": millis(2024, 1), "bands": ["value"]}]
        self.assertEqual(_collection_frames(rows)[0][0], "year_2024_01")

    def test_ambiguous_and_multiband_fail_loudly(self):
        same = [{"index": "a", "time": millis(2024), "bands": ["value"]},
                {"index": "b", "time": millis(2024), "bands": ["value"]}]
        with self.assertRaisesRegex(ValueError, "mesmo período"):
            _collection_frames(same)
        with self.assertRaisesRegex(ValueError, "múltiplas bandas"):
            _collection_frames([{"index": "a", "time": millis(2024), "bands": ["a", "b"]}])

    def test_year_property(self):
        self.assertEqual(_collection_frames([{"index": "a", "year": 2020, "bands": ["x"]}]),
                         [("year_2020", "a", "x")])
