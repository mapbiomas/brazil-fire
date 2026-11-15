"""M2 scheduling and progress reporting regressions."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core')))
import M2_mosaic_ui as ui_module


class ExecutionReportingTests(unittest.TestCase):
    def test_normal_chunk_grid_routes_to_assembly_without_manifest_update(self):
        ui = ui_module.MosaicAssemblerUI.__new__(ui_module.MosaicAssemblerUI)
        ui.now, ui.years, ui.bands, ui.chk_dict = SimpleNamespace(), [2025], ['red'], {}
        ui.state = {'cogs_monthly': [], 'gcs_chunks': {'mosaic': ['red']}}
        ui.update_manifest = {'version': 1, 'items': []}
        ui._reload_current_tab = Mock()
        period = SimpleNamespace(year=2025, key='2025_01', index=1, periodicity='monthly')
        with patch.object(ui_module, 'CONFIG', {'country': 'testland', 'bucket': 'test'}), \
             patch.object(ui_module, 'list_periods', return_value=[period]), \
             patch.object(ui_module, 'get_sensor_start_year', return_value=2000), \
             patch.object(ui_module, 'period_state_key', return_value='cogs_monthly'), \
             patch.object(ui_module, 'mosaic_name_for_period', side_effect=lambda *a, **k: 'mosaic'), \
             patch.object(ui_module, 'cell_log'), \
             patch.object(ui_module, 'save_manifest') as save, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}), \
             patch('M2_mosaic_logic.assemble_country_mosaic', return_value=['done']) as assemble, \
             patch('M2_mosaic_logic.mask_existing_country_cog') as update:
            ui._build_mosaic_grid('sentinel2', 'monthly', 'minnbr')
            checkbox = next(iter(ui.chk_dict.values()))
            self.assertFalse(checkbox.disabled)
            self.assertFalse(checkbox._meta['country_update'])
            self.assertTrue(checkbox._meta['has_chunks'])
            checkbox.value = True
            ui_module.start_mosaic_assembly(ui)
        assemble.assert_called_once()
        update.assert_not_called()
        save.assert_not_called()
        self.assertEqual(ui.update_manifest['items'], [])

    def test_grid_selection_completes_original_gcs_update_and_refreshes_ui(self):
        ui = ui_module.MosaicAssemblerUI.__new__(ui_module.MosaicAssemblerUI)
        ui.now = SimpleNamespace()
        ui.years = [2025]
        ui.bands = ['red']
        ui.chk_dict = {}
        ui.state = {'cogs_monthly': ['mosaic_red']}
        entry = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'band': 'red',
            'type': 'GCS', 'status': 'pending',
        }
        ui.update_manifest = {'version': 1, 'items': [entry]}
        period_info = SimpleNamespace(
            year=2025, key='2025_01', index=1, periodicity='monthly',
        )
        def rebuild():
            ui.chk_dict = {}
            ui._build_mosaic_grid('sentinel2', 'monthly', 'minnbr')

        ui._reload_current_tab = Mock(side_effect=rebuild)
        with patch.object(ui_module, 'CONFIG', {'country': 'testland', 'bucket': 'test'}), \
             patch.object(ui_module, 'list_periods', return_value=[period_info]), \
             patch.object(ui_module, 'get_sensor_start_year', return_value=2000), \
             patch.object(ui_module, 'period_state_key', return_value='cogs_monthly'), \
             patch.object(ui_module, 'mosaic_name_for_period', return_value='mosaic_red'), \
             patch.object(ui_module, 'period_cog_path', return_value='cogs'), \
             patch.object(ui_module, 'is_edit_mode', return_value=False), \
             patch.object(ui_module, 'save_manifest'), \
             patch.object(ui_module, 'cell_log'), \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs',
                          return_value={'cogs_monthly': ['mosaic_red']}), \
             patch('M2_mosaic_logic.mask_existing_country_cog', return_value=True) as mask:
            rebuild()
            checkbox = next(iter(ui.chk_dict.values()))
            self.assertFalse(checkbox.disabled)
            self.assertTrue(checkbox._meta['country_update'])
            checkbox.value = True
            ui_module.start_mosaic_assembly(ui)

        mask.assert_called_once()
        self.assertEqual(len(ui.update_manifest['items']), 1)
        self.assertIs(ui.update_manifest['items'][0], entry)
        self.assertEqual(entry['status'], 'complete')
        ui._reload_current_tab.assert_called_once()
        refreshed = next(iter(ui.chk_dict.values()))
        self.assertFalse(refreshed._meta['country_update'])
        self.assertTrue(refreshed.disabled)

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
