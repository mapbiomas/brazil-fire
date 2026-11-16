"""M2 scheduling and progress reporting regressions."""
import os
import copy
import errno
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core')))
import M2_mosaic_ui as ui_module


class ExecutionReportingTests(unittest.TestCase):
    def test_disk_full_while_submit_manifest_save_is_blocked(self):
        items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                      year=2025, month=1, period_key='2025_01', band=band,
                      type='GCS', country_update=True, has_cog=True)
                 for band in ('blue', 'green', 'red')]
        entries = [dict(item, status='pending', last_error='previous') for item in items]
        original = copy.deepcopy(entries)
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': entries}, years=[], _reload_current_tab=Mock())
        save_started = threading.Event()
        reported = threading.Event()
        event_factory = threading.Event
        class Signal:
            def __init__(self):
                self.event = event_factory()
            def set(self):
                self.event.set()
                reported.set()
            def is_set(self):
                return self.event.is_set()
        def save(data):
            if find_band_status(data, 'green') == 'masking_cog':
                save_started.set()
                self.assertTrue(reported.wait(5))
        def find_band_status(data, band):
            return next(entry['status'] for entry in data['items'] if entry['band'] == band)
        def process(item, *args, **kwargs):
            self.assertTrue(save_started.wait(5))
            raise OSError(errno.ENOSPC, 'disk full')
        with patch.object(ui_module, 'threading', SimpleNamespace(Event=Signal, Lock=threading.Lock)), \
             patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'save_manifest', side_effect=save), \
             patch.object(ui_module, 'process_country_mosaic_item', side_effect=process) as work, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(work.call_count, 1)
        self.assertEqual(entries[0]['status'], 'pending')
        self.assertEqual(entries[1:], original[1:])
        self.assertTrue(any('1 failed' in str(c.args[0]) and '2 skipped' in str(c.args[0])
                            for c in log.call_args_list))

    def test_deepcopy_failure_drains_active_worker_without_retrying_manifest(self):
        items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                      year=2025, month=1, period_key='2025_01', band=band,
                      type='GCS', country_update=True, has_cog=True)
                 for band in ('blue', 'green')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': [dict(item, status='pending') for item in items]},
                             years=[], _reload_current_tab=Mock())
        original_copy = copy.deepcopy
        calls = []
        def clone(value):
            calls.append(1)
            if len(calls) == 2:
                raise MemoryError('copy failed')
            return original_copy(value)
        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'copy', SimpleNamespace(deepcopy=clone)), \
             patch.object(ui_module, 'save_manifest') as save, \
             patch.object(ui_module, 'process_country_mosaic_item', return_value=['done']) as work, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(work.call_count, 1)
        self.assertEqual(save.call_count, 1)
        self.assertTrue(any('1 succeeded' in str(c.args[0]) and '1 skipped' in str(c.args[0])
                            and '1 manifest unconfirmed' in str(c.args[0]) for c in log.call_args_list))

    def test_disk_full_signalled_before_coordinator_replaces_success(self):
        started = []
        release_failure = threading.Event()
        items = [dict(sensor='s', year=2025, month=1, period_key='2025_01',
                      band=band, mosaic='m') for band in ('blue', 'green', 'red')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        real_wait = ui_module.wait
        def wait_with_race(futures, **kwargs):
            done, pending = real_wait(futures, **kwargs)
            release_failure.set()
            real_wait(futures)
            # Return only the original completed set, simulating the race.
            return done, pending
        def process(item, *args, **kwargs):
            started.append(item['band'])
            if item['band'] == 'green':
                self.assertTrue(release_failure.wait(5))
                raise RuntimeError('No space left on device')
            return ['done']
        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'wait', side_effect=wait_with_race), \
             patch.object(ui_module, 'process_country_mosaic_item', side_effect=process), \
             patch.object(ui_module, 'cell_log'), \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertCountEqual(started, ['blue', 'green'])

    def test_manifest_save_failure_drains_active_and_reports_unconfirmed(self):
        items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                      year=2025, month=1, period_key='2025_01', band=band,
                      type='GCS', country_update=True, has_cog=True)
                 for band in ('blue', 'green', 'red')]
        entries = [dict(item, status='pending') for item in items]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': entries}, years=[], _reload_current_tab=Mock())
        barrier = threading.Barrier(2)
        def process(item, *args, **kwargs):
            barrier.wait(timeout=5)
            return ['done']
        # Two submit saves succeed, first completion save fails.
        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'save_manifest', side_effect=[None, None, OSError('save failed')]) as save, \
             patch.object(ui_module, 'process_country_mosaic_item', side_effect=process) as work, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(work.call_count, 2)
        self.assertEqual(save.call_count, 3)
        self.assertEqual(entries[2]['status'], 'pending')
        messages = [str(c.args[0]) for c in log.call_args_list]
        self.assertTrue(any('2 succeeded' in x and '0 failed' in x and '1 skipped' in x
                            and '2 manifest unconfirmed' in x for x in messages))

    def test_submit_manifest_failure_drains_previously_started_worker(self):
        items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                      year=2025, month=1, period_key='2025_01', band=band,
                      type='GCS', country_update=True, has_cog=True)
                 for band in ('blue', 'green', 'red')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': [dict(item, status='pending') for item in items]},
                             years=[], _reload_current_tab=Mock())
        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'save_manifest', side_effect=[None, OSError('save failed')]) as save, \
             patch.object(ui_module, 'process_country_mosaic_item', return_value=['done']) as work, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(work.call_count, 1)
        self.assertEqual(save.call_count, 2)
        self.assertTrue(any('1 succeeded' in str(c.args[0]) and '2 skipped' in str(c.args[0])
                            and '1 manifest unconfirmed' in str(c.args[0]) for c in log.call_args_list))

    def test_failure_status_save_error_does_not_abandon_active_success(self):
        items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                      year=2025, month=1, period_key='2025_01', band=band,
                      type='GCS', country_update=True, has_cog=True)
                 for band in ('blue', 'green', 'red')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': [dict(item, status='pending') for item in items]},
                             years=[], _reload_current_tab=Mock())
        release = threading.Event()
        both = threading.Barrier(2)
        def process(item, *args, **kwargs):
            both.wait(timeout=5)
            if item['band'] == 'blue':
                raise RuntimeError('ordinary failure')
            self.assertTrue(release.wait(5))
            return ['done']
        def save(data):
            save.calls += 1
            if save.calls == 3:
                release.set()
                raise OSError('save failed')
        save.calls = 0
        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'save_manifest', side_effect=save), \
             patch.object(ui_module, 'process_country_mosaic_item', side_effect=process) as work, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(work.call_count, 2)
        self.assertEqual(save.calls, 3)
        self.assertEqual(ui.update_manifest['items'][0]['status'], 'masking_cog')
        self.assertTrue(any('1 succeeded' in str(c.args[0]) and '1 failed' in str(c.args[0])
                            and '1 skipped' in str(c.args[0])
                            and '2 manifest unconfirmed' in str(c.args[0]) for c in log.call_args_list))

    def test_disk_full_stops_unscheduled_and_preserves_manifest(self):
        errors = [
            RuntimeError("GDAL COG: No space left on device"),
            subprocess.CalledProcessError(1, ['gdal_translate'], stderr=b'No space left on device'),
            OSError(errno.ENOSPC, 'disk full'),
        ]
        for error in errors:
            with self.subTest(error=error):
                items = [dict(country='testland', sensor='s', period='monthly', mosaic='m',
                              year=2025, month=1, period_key='2025_01', band=band,
                              type='GCS', country_update=True, has_cog=True)
                         for band in ('blue', 'green', 'red')]
                entries = [dict(item, status='pending', last_error='old') for item in items]
                original = copy.deepcopy(entries)
                ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                               for i, item in enumerate(items)},
                                     update_manifest={'items': entries}, years=[],
                                     _reload_current_tab=Mock())
                with patch.object(ui_module, 'choose_m2_workers', return_value=1), \
                     patch.object(ui_module, 'process_country_mosaic_item', side_effect=error) as process, \
                     patch.object(ui_module, 'save_manifest'), \
                     patch.object(ui_module, 'cell_log') as log, \
                     patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
                    ui_module.start_mosaic_assembly(ui)
                self.assertEqual(process.call_count, 1)
                self.assertEqual(entries[0]['status'], 'pending')
                self.assertEqual(entries[1:], original[1:])
                self.assertTrue(any('1 failed' in str(c.args[0]) and '2 skipped' in str(c.args[0])
                                    for c in log.call_args_list))

    def test_bounded_workers_allow_active_to_finish_but_stop_after_disk_full(self):
        started = []
        both_started = threading.Barrier(2)
        release = threading.Event()
        items = [dict(sensor='s', year=2025, month=1, period_key='2025_01',
                      band=band, mosaic='m') for band in ('blue', 'green', 'red')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        def process(item, *args, **kwargs):
            started.append(item['band'])
            both_started.wait(timeout=5)
            if item['band'] == 'blue':
                raise RuntimeError('No space left on device')
            self.assertTrue(release.wait(timeout=5))
            return ['done']

        def log(message, **kwargs):
            if message.startswith('Failed'):
                release.set()

        with patch.object(ui_module, 'choose_m2_workers', return_value=2), \
             patch.object(ui_module, 'process_country_mosaic_item', side_effect=process), \
             patch.object(ui_module, 'cell_log', side_effect=log), \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertCountEqual(started, ['blue', 'green'])

    def test_ordinary_failure_does_not_stop_queue(self):
        items = [dict(sensor='s', year=2025, month=1, period_key='2025_01',
                      band=band, mosaic='m') for band in ('blue', 'green')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        with patch.object(ui_module, 'choose_m2_workers', return_value=1), \
             patch.object(ui_module, 'process_country_mosaic_item',
                          side_effect=[RuntimeError('other error'), ['done']]) as process, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(process.call_count, 2)
        self.assertTrue(any('1 failed' in str(c.args[0]) and '0 skipped' in str(c.args[0])
                            for c in log.call_args_list))

    def test_cog_space_estimate_is_item_failure_not_disk_full(self):
        import M2_mosaic_logic
        class COGSpaceEstimateExceeded(Exception):
            pass
        items = [dict(sensor='s', year=2025, month=1, period_key='2025_01',
                      band=band, mosaic='m') for band in ('blue', 'green')]
        ui = SimpleNamespace(chk_dict={str(i): SimpleNamespace(value=True, _meta=item)
                                       for i, item in enumerate(items)},
                             update_manifest={'items': []}, years=[], _reload_current_tab=Mock())
        error = COGSpaceEstimateExceeded('COG preflight estimate: no space left on device')
        with patch.object(M2_mosaic_logic, 'COGSpaceEstimateExceeded',
                          COGSpaceEstimateExceeded, create=True), \
             patch.object(ui_module, 'choose_m2_workers', return_value=1), \
             patch.object(ui_module, 'process_country_mosaic_item',
                          side_effect=[error, ['done']]) as process, \
             patch.object(ui_module, 'cell_log') as log, \
             patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}):
            ui_module.start_mosaic_assembly(ui)
        self.assertEqual(process.call_count, 2)
        self.assertTrue(any('1 succeeded' in str(c.args[0]) and '1 failed' in str(c.args[0])
                            and '0 skipped' in str(c.args[0]) for c in log.call_args_list))

    def test_pending_update_grid_routes_chunks_and_cog(self):
        for has_cog in (False, True):
            with self.subTest(has_cog=has_cog):
                ui = ui_module.MosaicAssemblerUI.__new__(ui_module.MosaicAssemblerUI)
                ui.now, ui.years, ui.bands, ui.chk_dict = SimpleNamespace(), [2025], ['red'], {}
                ui.state = {'cogs_monthly': ['mosaic'] if has_cog else [],
                            'gcs_chunks': {'mosaic': ['red']}}
                entry = {'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
                         'mosaic': 'minnbr', 'period_key': '2025_01', 'band': 'red',
                         'type': 'GCS', 'status': 'pending'}
                ui.update_manifest = {'version': 1, 'items': [entry]}
                ui._reload_current_tab = Mock()
                period = SimpleNamespace(year=2025, key='2025_01', index=1, periodicity='monthly')
                with patch.object(ui_module, 'CONFIG', {'country': 'testland', 'bucket': 'test'}), \
                     patch.object(ui_module, 'list_periods', return_value=[period]), \
                     patch.object(ui_module, 'get_sensor_start_year', return_value=2000), \
                     patch.object(ui_module, 'period_state_key', return_value='cogs_monthly'), \
                     patch.object(ui_module, 'mosaic_name_for_period', return_value='mosaic'), \
                     patch.object(ui_module, 'period_cog_path', return_value='cogs'), \
                     patch.object(ui_module, 'is_edit_mode', return_value=False), \
                     patch.object(ui_module, 'cell_log'), \
                     patch.object(ui_module, 'save_manifest'), \
                     patch.object(ui_module.CacheManager, 'build_cache_from_gcs', return_value={}), \
                     patch('M2_mosaic_logic.assemble_country_mosaic', return_value=['done']) as assemble, \
                     patch('M2_mosaic_logic.mask_existing_country_cog', return_value=True) as mask:
                    grid = ui._build_mosaic_grid('sentinel2', 'monthly', 'minnbr')
                    self.assertIn('UPDATE' if has_cog else 'READY',
                                  grid.children[1].children[2].children[1].value)
                    checkbox = next(iter(ui.chk_dict.values()))
                    self.assertFalse(checkbox.disabled)
                    self.assertTrue(checkbox._meta['country_update'])
                    checkbox.value = True
                    ui_module.start_mosaic_assembly(ui)
                self.assertEqual(assemble.call_count, 0 if has_cog else 1)
                self.assertEqual(mask.call_count, 1 if has_cog else 0)
                self.assertEqual(entry['status'], 'complete')

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
