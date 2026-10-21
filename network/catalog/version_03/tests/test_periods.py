"""Offline tests for sample period normalization."""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from datetime import datetime

from M_periods import (
    ALL_PERIODICITIES,
    ENABLED_PERIODICITIES,
    FOLDER_TO_PERIODICITY,
    PERIODS,
    list_periods,
    normalize_period,
    parse_period,
    period_key,
    period_state_key,
    period_year,
)


class NormalizePeriodTests(unittest.TestCase):
    def test_integer_year(self):
        self.assertEqual(normalize_period(2025), '2025')

    def test_float_year(self):
        self.assertEqual(normalize_period(2025.0), '2025')

    def test_string_year(self):
        self.assertEqual(normalize_period('2025'), '2025')

    def test_string_with_decimal(self):
        self.assertEqual(normalize_period('2025.0'), '2025')

    def test_monthly_string(self):
        self.assertEqual(normalize_period('2025_10'), '2025_10')

    def test_six_digit_integer_is_yyyymm(self):
        self.assertEqual(normalize_period(202510), '2025_10')

    def test_six_digit_string_is_yyyymm(self):
        self.assertEqual(normalize_period('202510'), '2025_10')

    def test_missing_values(self):
        self.assertEqual(normalize_period(None), '')
        self.assertEqual(normalize_period(float('nan')), '')
        self.assertEqual(normalize_period('nan'), '')
        self.assertEqual(normalize_period('<NA>'), '')


class PeriodYearTests(unittest.TestCase):
    def test_year_from_values(self):
        self.assertEqual(period_year('2025'), 2025)
        self.assertEqual(period_year('2025_10'), 2025)
        self.assertEqual(period_year(2025), 2025)
        self.assertEqual(period_year(202510), 2025)

    def test_invalid_values(self):
        self.assertIsNone(period_year(''))
        self.assertIsNone(period_year(None))
        self.assertIsNone(period_year('abc'))


class RegistryTests(unittest.TestCase):
    def test_enabled_periods(self):
        self.assertEqual(ENABLED_PERIODICITIES, ('yearly', 'monthly'))
        self.assertIn('semestral', ALL_PERIODICITIES)
        self.assertIn('quarterly', ALL_PERIODICITIES)

    def test_folders_are_unique(self):
        folders = [definition['folder'] for definition in PERIODS.values()]
        self.assertEqual(len(folders), len(set(folders)))
        self.assertEqual(FOLDER_TO_PERIODICITY['MONTHLY'], 'monthly')
        self.assertEqual(FOLDER_TO_PERIODICITY['ANNUALLY'], 'yearly')

    def test_gee_folders_preserve_legacy_yearly(self):
        self.assertEqual(PERIODS['yearly']['gee_folder'], 'YEARLY')
        self.assertEqual(PERIODS['monthly']['gee_folder'], 'MONTHLY')

    def test_period_state_keys_keep_current_names(self):
        self.assertEqual(period_state_key('cogs', 'monthly'), 'cogs_monthly')
        self.assertEqual(period_state_key('cogs', 'yearly'), 'cogs_annually')
        self.assertEqual(period_state_key('assets', 'yearly'), 'assets_annually')


class PeriodKeyTests(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(period_key(2025, 1, 'yearly'), '2025')
        self.assertEqual(period_key(2025, 6, 'monthly'), '2025_06')
        self.assertEqual(period_key(2025, 2, 'semestral'), '2025_S2')
        self.assertEqual(period_key(2025, 3, 'quarterly'), '2025_Q3')


class ParsePeriodTests(unittest.TestCase):
    def test_yearly(self):
        period = parse_period('2025')
        self.assertEqual(period.periodicity, 'yearly')
        self.assertEqual(period.folder, 'ANNUALLY')
        self.assertEqual(period.start, datetime(2025, 1, 1))
        self.assertEqual(period.end, datetime(2026, 1, 1))

    def test_monthly(self):
        period = parse_period('2025_06')
        self.assertEqual(period.periodicity, 'monthly')
        self.assertEqual(period.folder, 'MONTHLY')
        self.assertEqual(period.start, datetime(2025, 6, 1))
        self.assertEqual(period.end, datetime(2025, 7, 1))

    def test_semester_and_quarter(self):
        semester = parse_period('2025_S2')
        self.assertEqual(semester.start, datetime(2025, 7, 1))
        self.assertEqual(semester.end, datetime(2026, 1, 1))

        quarter = parse_period('2025_Q3')
        self.assertEqual(quarter.start, datetime(2025, 7, 1))
        self.assertEqual(quarter.end, datetime(2025, 10, 1))

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            parse_period('2025_13')
        with self.assertRaises(ValueError):
            parse_period('2025_X1')
        with self.assertRaises(ValueError):
            parse_period('2025', 'monthly')


class ListPeriodsTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 20)

    def test_yearly_current_behavior(self):
        all_years = list_periods([2025, 2026], 'yearly', include_current=True, now=self.now)
        self.assertEqual([p.key for p in all_years], ['2026', '2025'])

        complete = list_periods([2025, 2026], 'yearly', include_current=False, now=self.now)
        self.assertEqual([p.key for p in complete], ['2025'])

    def test_monthly_excludes_current_month(self):
        periods = list_periods([2026], 'monthly', include_current=False, now=self.now)
        self.assertEqual(len(periods), 9)
        self.assertEqual(periods[0].key, '2026_09')
        self.assertEqual(periods[-1].key, '2026_01')

    def test_future_periods_are_parsed_but_not_enabled(self):
        semesters = list_periods([2026], 'semestral', include_current=False, now=self.now)
        self.assertEqual([p.key for p in semesters], ['2026_S1'])

        quarters = list_periods([2026], 'quarterly', include_current=False, now=self.now)
        self.assertEqual([p.key for p in quarters], ['2026_Q3', '2026_Q2', '2026_Q1'])


class FutureKeyNormalizationTests(unittest.TestCase):
    def test_semester_and_quarter_are_preserved(self):
        self.assertEqual(normalize_period('2025_S1'), '2025_S1')
        self.assertEqual(normalize_period('2025_Q1'), '2025_Q1')
        self.assertEqual(period_year('2025_S1'), 2025)
        self.assertEqual(period_year('2025_Q1'), 2025)


if __name__ == '__main__':
    unittest.main()
