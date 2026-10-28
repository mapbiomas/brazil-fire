"""Offline tests for the M1 cell status contract.

M2 removes the source chunks after a COG is built, so a GCS row must stay OK
when either the chunks or the final COG exists.
"""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

from M1_export_ui import resolve_output_state


class OutputStateTests(unittest.TestCase):
    def test_gcs_chunks_only_is_ok(self):
        self.assertEqual(
            resolve_output_state('GCS', True, False, False, False),
            ('OK', True),
        )

    def test_gcs_cog_only_is_ok(self):
        self.assertEqual(
            resolve_output_state('GCS', False, True, False, False),
            ('OK', True),
        )

    def test_gcs_with_nothing_is_miss(self):
        self.assertEqual(
            resolve_output_state('GCS', False, False, False, False),
            ('MISS', False),
        )

    def test_asset_row_only_depends_on_the_asset(self):
        self.assertEqual(
            resolve_output_state('ASSET', False, False, True, False),
            ('OK', True),
        )
        self.assertEqual(
            resolve_output_state('ASSET', True, True, False, False),
            ('MISS', False),
        )

    def test_running_task_wins_over_existing_outputs(self):
        self.assertEqual(
            resolve_output_state('GCS', True, True, True, True),
            ('RUN', True),
        )


if __name__ == '__main__':
    unittest.main()
