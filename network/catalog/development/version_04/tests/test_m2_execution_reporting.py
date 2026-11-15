"""M2 scheduling and progress reporting regressions."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core')))
import M2_mosaic_ui as ui_module


class ExecutionReportingTests(unittest.TestCase):
    def test_duplicate_destination_is_scheduled_only_once(self):
        meta = {'sensor': 'sentinel2', 'year': 2025, 'month': 1,
                'period_key': '2025_01', 'band': 'red', 'mosaic': 'minnbr'}
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=dict(meta)) for i in range(2)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        with patch.object(ui_module, 'process_country_mosaic_item', return_value=['done']) as process, \
             patch.object(ui_module, 'cell_log'), \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(process.call_count, 1)

    def test_empty_normal_assembly_is_reported_as_failure(self):
        meta = {'sensor': 'sentinel2', 'year': 2025, 'month': 1,
                'period_key': '2025_01', 'band': 'red', 'mosaic': 'minnbr'}
        ui = SimpleNamespace(chk_dict={'x': SimpleNamespace(value=True, _meta=meta)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        with patch.object(ui_module, 'process_country_mosaic_item', return_value=[]), \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertTrue(any('Failed' in str(call.args[0]) for call in log.call_args_list))
        self.assertFalse(any(str(call.args[0]).startswith('OK:') for call in log.call_args_list))

    def test_thread_budget_uses_detected_cpus_without_oversubscription(self):
        choose = ui_module.choose_gdal_threads
        gib = 1024 ** 3
        self.assertEqual(choose(8, 1, 16*gib), 8)
        self.assertEqual(choose(8, 2, 16*gib), 4)
        self.assertEqual(choose(8, 1, 3*gib), 2)
        self.assertEqual(choose(2, 2, 16*gib), 1)
        self.assertEqual(choose(None, 1, None), 1)

    def test_resource_policy_is_conservative_and_scales_only_with_headroom(self):
        choose = ui_module.choose_m2_workers
        gib = 1024 ** 3
        self.assertEqual(choose(28, 8, 40*gib, 40*gib, 150*gib), 2)
        self.assertEqual(choose(28, 2, 40*gib, 40*gib, 150*gib), 1)
        self.assertEqual(choose(28, 8, 40*gib, 8*gib, 150*gib), 1)
        self.assertEqual(choose(28, 8, 40*gib, 40*gib, 20*gib), 1)
        self.assertEqual(choose(28, None, None, None, None), 1)
        self.assertEqual(choose(1, 8, 40*gib, 40*gib, 150*gib), 1)

    def test_default_runs_one_product_and_reports_start_before_work(self):
        events = []
        meta = {'sensor': 'sentinel2', 'year': 2025, 'month': 1,
                'period_key': '2025_01', 'band': 'red', 'mosaic': 'minnbr'}
        ui = SimpleNamespace(chk_dict={'x': SimpleNamespace(value=True, _meta=meta)},
                             update_manifest={'items': []}, years=[],
                             _reload_current_tab=Mock())

        def process(*args, **kwargs):
            events.append('work')
            return ['done']

        def log(message, **kwargs):
            if 'Processing' in message:
                events.append('start')

        with patch.object(ui_module, 'process_country_mosaic_item', side_effect=process), \
             patch.object(ui_module, 'cell_log', side_effect=log), \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(events, ['start', 'work'])
