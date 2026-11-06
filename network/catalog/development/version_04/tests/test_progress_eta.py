"""Offline tests for conservative remaining-time estimates."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
from M_lang import L
from M_progress_eta import format_duration, progress_message


class ProgressEtaTests(unittest.TestCase):
    def test_formats_durations_over_an_hour(self):
        self.assertEqual(format_duration(3661), '01:01:01')

    def test_resume_uses_only_newly_completed_tiles_for_rate(self):
        with patch.object(L, 'ETA_PROGRESS', '{done}/{total} {elapsed} {eta}'):
            msg = progress_message('phase', 103, 110, 30, 3, 10)
        self.assertEqual(msg, '103/110 00:00:30 00:01:10')

    def test_estimate_waits_for_samples_and_does_not_hide_failures(self):
        with patch.object(L, 'ETA_PENDING', 'pending'), patch.object(L, 'ETA_RETRY', 'retry'), \
             patch.object(L, 'ETA_PROGRESS', '{eta}'):
            self.assertEqual(progress_message('phase', 2, 10, 10, 2, 10), 'pending')
            self.assertEqual(progress_message('phase', 4, 10, 10, 4, 10, 1), 'retry')
