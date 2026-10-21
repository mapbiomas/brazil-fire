"""Offline tests for the shared year selection helper."""

import contextlib
import datetime
import io
import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from M0_auth_config import resolve_years

CURRENT_YEAR = datetime.datetime.now().year


class ResolveYearsTests(unittest.TestCase):
    def test_empty_returns_all_landsat_years(self):
        years = resolve_years([], sensor='landsat')
        self.assertEqual(years[0], CURRENT_YEAR)
        self.assertEqual(years[-1], 1985)
        self.assertEqual(len(years), CURRENT_YEAR - 1985 + 1)

    def test_none_returns_all_sentinel2_years(self):
        years = resolve_years(None, sensor='sentinel2')
        self.assertEqual(years[0], CURRENT_YEAR)
        self.assertEqual(years[-1], 2017)

    def test_explicit_list_is_sorted_and_deduplicated(self):
        self.assertEqual(
            resolve_years([2020, 2018, 2020], sensor='landsat'),
            [2020, 2018],
        )

    def test_single_int(self):
        self.assertEqual(resolve_years(2020, sensor='landsat'), [2020])

    def test_interval_separators(self):
        for value in ('1985:1987', '1985;1987', '1985-1987', ['1985:1987']):
            self.assertEqual(
                resolve_years(value, sensor='landsat'),
                [1987, 1986, 1985],
            )

    def test_out_of_range_is_ignored(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            years = resolve_years([1900, 2020], sensor='landsat')
        self.assertEqual(years, [2020])
        self.assertIn('ignored', output.getvalue())

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            resolve_years(['not-a-year'], sensor='landsat')

    def test_no_valid_years_raises(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                resolve_years(['1985:2000'], sensor='sentinel2')


if __name__ == '__main__':
    unittest.main()
