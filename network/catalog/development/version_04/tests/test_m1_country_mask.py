"""Regression tests for country-wide M1 masks and gradual updates."""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from ipywidgets import Checkbox
from unittest.mock import patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M1_export_logic
from M1_update_manifest import (
    complete_gcs_item,
    item_key,
    mark_item_complete,
    pending_item,
)
from M1_export_ui import make_update_label
from M0_auth_config import set_global_opts
import M0_auth_config
import M1_export_ui
import M2_mosaic_logic
import M2_mosaic_ui
from M2_mosaic_logic import mask_raster_to_country


class CountryMaskTests(unittest.TestCase):
    def test_country_mask_paints_all_region_features_and_updates_mask(self):
        features = object()
        target_image = Mock(name='source_image')
        fake_mask = object()
        painted = Mock()
        painted.gt.return_value = fake_mask
        byte_image = Mock()
        byte_image.byte.return_value = byte_image
        byte_image.paint.return_value = painted

        original_image = M1_export_logic.ee.Image
        image_factory = Mock(return_value=byte_image)
        M1_export_logic.ee.Image = image_factory
        try:
            result = M1_export_logic.apply_country_mask(target_image, features)
        finally:
            M1_export_logic.ee.Image = original_image

        image_factory.assert_called_once_with(0)
        byte_image.paint.assert_called_once_with(features, 1)
        painted.gt.assert_called_once_with(0)
        target_image.updateMask.assert_called_once_with(fake_mask)
        self.assertEqual(result, target_image.updateMask.return_value)


class UpdateManifestTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            'country': 'testland',
            'sensor': 'sentinel2',
            'period': 'monthly',
            'mosaic': 'minnbr',
            'period_key': '2025_01',
            'band': 'red',
            'type': 'GCS',
            'status': 'pending',
        }

    def test_identity_is_m1_date_band_and_target_identity_without_campaign(self):
        self.assertEqual(
            item_key(self.item),
            ('testland', 'sentinel2', 'monthly', 'minnbr', '2025_01', 'red', 'gcs'),
        )
        changed = dict(self.item, campaign='SOME_CAMPAIGN')
        self.assertEqual(item_key(changed), item_key(self.item))

    def test_pending_entry_is_marked_complete_by_exact_m1_identity(self):
        manifest = {'version': 1, 'items': [dict(self.item)]}
        self.assertTrue(pending_item(manifest, self.item))
        mark_item_complete(manifest, dict(self.item, type='ASSET'))
        self.assertTrue(pending_item(manifest, self.item))
        mark_item_complete(manifest, self.item)
        self.assertFalse(pending_item(manifest, self.item))

    def test_new_masked_export_is_not_presented_as_an_update(self):
        manifest = {'version': 1, 'items': [dict(self.item, status='submitted_new')]}
        self.assertFalse(pending_item(manifest, self.item))

    def test_m2_cannot_complete_a_gcs_item_before_m1_staging_is_promoted(self):
        manifest = {'version': 1, 'items': [dict(self.item, status='pending')]}
        with patch('M1_update_manifest.load_manifest', return_value=manifest), \
             patch('M1_update_manifest.save_manifest') as save:
            complete_gcs_item('testland', 'sentinel2', 'monthly', 'minnbr', '2025_01', 'red')
        self.assertEqual(manifest['items'][0]['status'], 'pending')
        save.assert_not_called()

    def test_m2_completes_gcs_item_only_after_chunks_ready(self):
        manifest = {'version': 1, 'items': [dict(self.item, status='chunks_ready')]}
        with patch('M1_update_manifest.load_manifest', return_value=manifest), \
             patch('M1_update_manifest.save_manifest') as save:
            complete_gcs_item('testland', 'sentinel2', 'monthly', 'minnbr', '2025_01', 'red')
        self.assertEqual(manifest['items'][0]['status'], 'complete')
        save.assert_called_once_with(manifest)

    def test_update_status_is_a_label_next_to_the_existing_checkbox(self):
        checkbox = Checkbox()
        cell = make_update_label(checkbox, M1_export_ui.Lang.M1_UPDATE_LABEL)
        self.assertIs(cell.children[0], checkbox)
        self.assertIsInstance(cell.children[1], M1_export_ui.widgets.HTML)
        self.assertIn(M1_export_ui.Lang.M1_UPDATE_LABEL, cell.children[1].value)
        self.assertIn('#856404', cell.children[1].value)
        self.assertIn('mfm-update', cell._dom_classes)

    def test_m1_only_marks_existing_gee_assets_as_updates(self):
        self.assertEqual(
            M1_export_ui.resolve_output_state(
                'ASSET', False, False, True, False, update_pending=True,
            ),
            ('UPDATE', True),
        )
        self.assertEqual(
            M1_export_ui.resolve_output_state(
                'GCS', True, False, False, False, update_pending=True,
            ),
            ('OK', True),
        )

    def test_m2_pending_cog_checkbox_is_enabled_for_selection(self):
        from types import SimpleNamespace

        ui = M2_mosaic_ui.MosaicAssemblerUI.__new__(M2_mosaic_ui.MosaicAssemblerUI)
        ui.now = SimpleNamespace()
        ui.years = [2025]
        ui.bands = ['red']
        ui.chk_dict = {}
        ui.state = {'cogs_monthly': ['mosaic_red']}
        ui.update_manifest = {
            'version': 1,
            'items': [{
                'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
                'mosaic': 'minnbr', 'period_key': '2025_01', 'band': 'red',
                'type': 'GCS', 'status': 'pending',
            }],
        }
        period_info = SimpleNamespace(
            year=2025, key='2025_01', index=1, periodicity='monthly',
        )
        with patch.object(M2_mosaic_ui, 'CONFIG', {'country': 'testland'}), \
             patch.object(M2_mosaic_ui, 'list_periods', return_value=[period_info]), \
             patch.object(M2_mosaic_ui, 'get_sensor_start_year', return_value=2000), \
             patch.object(M2_mosaic_ui, 'period_state_key', return_value='cogs_monthly'), \
             patch.object(M2_mosaic_ui, 'mosaic_name_for_period', return_value='mosaic_red'), \
             patch.object(M2_mosaic_ui, 'is_edit_mode', return_value=False):
            ui._build_mosaic_grid('sentinel2', 'monthly', 'minnbr')

        checkbox = next(iter(ui.chk_dict.values()))
        self.assertFalse(checkbox.disabled)
        ui._on_select_all(None)
        self.assertTrue(checkbox.value)

    def test_update_notes_are_blue_and_explain_m1_and_m2_routes(self):
        from M_ui_components import make_update_note

        m1_note = make_update_note(M1_export_ui.Lang.M1_UPDATE_M1_NOTE)
        m2_note = make_update_note(M1_export_ui.Lang.M1_UPDATE_M2_NOTE)
        self.assertIn('GEE', m1_note.value)
        self.assertIn('COG', m2_note.value)
        status_css = M1_export_ui.PipelineStepUI.get_status_css().value
        self.assertIn('.mfm-update', status_css)
        self.assertIn('background:#fff3cd', status_css)
        self.assertIn('border:1px solid #ffeaa8', status_css)
        self.assertIn('#856404', status_css)

    def test_selected_asset_update_uses_normal_m1_export_dispatch(self):
        from types import SimpleNamespace

        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'ASSET', 'exists': True,
            'update_pending': True,
        }
        ui = M1_export_ui.ExportDispatcherUI.__new__(M1_export_ui.ExportDispatcherUI)
        ui.chk_dict = {'row': Checkbox(value=True)}
        ui.chk_dict['row']._meta = meta
        ui.update_manifest = {'version': 1, 'items': []}
        fake_feature_collection = Mock()
        with patch.object(M0_auth_config, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui.ee, 'FeatureCollection', return_value=fake_feature_collection), \
             patch('M_blocklist.refresh_blocklists', return_value={}), \
             patch('M1_export_logic.get_quality_mosaic', return_value=object()) as build_mosaic, \
             patch('M1_export_logic.export_to_asset', return_value=SimpleNamespace(id='task-1')) as export_asset, \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M1_export_ui.CacheManager.mark_stale'), \
             patch('M1_export_ui.save_manifest'):
            submitted = M1_export_ui.start_export(ui, mode='ASSET')

        self.assertEqual(submitted, 1)
        build_mosaic.assert_called_once()
        export_asset.assert_called_once()
        self.assertEqual(ui.update_manifest['items'][0]['status'], 'submitted')

    def test_start_export_routes_existing_gcs_output_to_m2_not_earth_engine(self):
        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS', 'exists': False,
        }
        ui = M1_export_ui.ExportDispatcherUI.__new__(M1_export_ui.ExportDispatcherUI)
        ui.chk_dict = {'row': Checkbox(value=True)}
        ui.chk_dict['row']._meta = meta
        ui.update_manifest = {'version': 1, 'items': []}
        with patch.object(M0_auth_config, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui.ee, 'FeatureCollection', return_value=Mock()), \
             patch('M_blocklist.refresh_blocklists', return_value={}), \
             patch('M1_export_ui.find_existing_gcs_mosaic', return_value='COG') as exists, \
             patch('M1_export_logic.get_quality_mosaic') as build_mosaic, \
             patch('M1_export_logic.export_to_gcs') as export_gcs, \
             patch('M1_export_ui.CacheManager.mark_stale'), \
             patch('M1_export_ui.save_manifest'):
            processed = M1_export_ui.start_export(ui, mode='GCS')

        self.assertEqual(processed, 1)
        exists.assert_called_once_with(meta)
        build_mosaic.assert_not_called()
        export_gcs.assert_not_called()
        self.assertEqual(ui.update_manifest['items'][0]['status'], 'pending')

    def test_start_export_uses_regular_gcs_task_when_product_does_not_exist(self):
        from types import SimpleNamespace

        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS', 'exists': False,
        }
        ui = M1_export_ui.ExportDispatcherUI.__new__(M1_export_ui.ExportDispatcherUI)
        ui.chk_dict = {'row': Checkbox(value=True)}
        ui.chk_dict['row']._meta = meta
        ui.update_manifest = {'version': 1, 'items': []}
        with patch.object(M0_auth_config, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui.ee, 'FeatureCollection', return_value=Mock()), \
             patch('M_blocklist.refresh_blocklists', return_value={}), \
             patch('M1_export_ui.find_existing_gcs_mosaic', return_value=None), \
             patch('M1_export_logic.get_quality_mosaic', return_value=object()) as build_mosaic, \
             patch('M1_export_logic.export_to_gcs', return_value=[SimpleNamespace(id='task-1')]) as export_gcs, \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M1_export_ui.CacheManager.mark_stale'), \
             patch('M1_export_ui.save_manifest'):
            processed = M1_export_ui.start_export(ui, mode='GCS')

        self.assertEqual(processed, 1)
        build_mosaic.assert_called_once()
        export_gcs.assert_called_once()
        self.assertEqual(ui.update_manifest['items'][0]['status'], 'submitted_new')

    def test_m1_grid_has_manifest_lookup_available(self):
        self.assertTrue(callable(M1_export_ui.find_item))

    def test_m1_export_logic_has_no_geometry_clip_calls(self):
        source_path = Path(CORE) / 'M1_export_logic.py'
        self.assertNotIn('.clip(', source_path.read_text(encoding='utf-8'))

    def test_manifest_uses_country_configuration_not_campaign(self):
        set_global_opts(
            country='testland', gcs_bucket='bucket',
            gcs_library_images_prefix='gs/testland',
            gcs_campaigns_prefix='gs/testland', gee_project='testland',
            gee_library_images_prefix='projects/testland/assets',
            gee_campaigns_prefix='projects/testland/assets',
            campaign='ANY_CAMPAIGN', asset_regions='regions',
            sensor=['sentinel2'], periodicity=['monthly'],
            mosaic_methods=['minnbr'], language='en',
        )
        M1_export_ui.CONFIG = M0_auth_config.CONFIG

    def test_staged_gcs_promotion_replaces_in_place_and_only_removes_after_copy(self):
        ui = M1_export_ui.ExportDispatcherUI.__new__(M1_export_ui.ExportDispatcherUI)
        item = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'band': 'red',
            'type': 'GCS', 'status': 'exporting', 'task_ids': ['task-1'],
            'staging_prefix': 'chunks/.m1_updates/run/red_',
            'target_prefix': 'gs://bucket/chunks/red_',
        }
        ui.update_manifest = {'version': 1, 'items': [item]}
        staged = ['gs://bucket/chunks/.m1_updates/run/red_-0000.tif']
        existing = [
            'gs://bucket/chunks/red_-0000.tif',
            'gs://bucket/chunks/red_-old-extra.tif',
        ]
        with patch.object(
            M1_export_ui.ee.data, 'getTaskStatus',
            return_value=[{'id': 'task-1', 'state': 'COMPLETED'}],
        ), patch('M_gcs.glob', side_effect=[staged, existing]), \
             patch('M_gcs.copy') as copy_object, \
             patch('M_gcs.rm_files') as remove_objects, \
             patch.object(M1_export_ui, 'save_manifest'):
            ui._reconcile_update_tasks()

        destination = 'gs://bucket/chunks/red_-0000.tif'
        copy_object.assert_called_once_with(staged[0], destination)
        self.assertEqual(item['status'], 'chunks_ready')
        removed_paths = [path for call in remove_objects.call_args_list for path in call.args[0]]
        self.assertNotIn(destination, removed_paths)
        self.assertIn(existing[1], removed_paths)

    def test_m2_selected_existing_cog_is_masked_directly(self):
        item = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS',
            'country_update': True, 'has_cog': True, 'has_chunks': False,
        }
        manifest = {'version': 1, 'items': [dict(item, status='pending')]}
        with patch('M2_mosaic_logic.mask_existing_country_cog', return_value=True) as mask_cog, \
             patch('M2_mosaic_logic.assemble_country_mosaic') as assemble, \
             patch('M2_mosaic_ui.save_manifest'):
            result = M2_mosaic_ui.process_country_mosaic_item(item, manifest)

        mask_cog.assert_called_once_with('2025_01', 'red', 'sentinel2', 'minnbr', logger=print)
        assemble.assert_not_called()
        self.assertTrue(result)
        self.assertEqual(manifest['items'][0]['status'], 'complete')

    def test_m2_selected_chunks_are_assembled_without_earth_engine_export(self):
        item = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS',
            'country_update': True, 'has_cog': False, 'has_chunks': True,
        }
        manifest = {'version': 1, 'items': [dict(item, status='pending')]}
        with patch('M2_mosaic_logic.mask_existing_country_cog') as mask_cog, \
             patch('M2_mosaic_logic.assemble_country_mosaic', return_value=['cog']) as assemble, \
             patch('M2_mosaic_ui.save_manifest'):
            result = M2_mosaic_ui.process_country_mosaic_item(item, manifest)

        mask_cog.assert_not_called()
        assemble.assert_called_once()
        self.assertEqual(result, ['cog'])
        self.assertEqual(manifest['items'][0]['status'], 'complete')

    def test_m2_does_not_process_manifest_update_when_data_is_absent(self):
        item = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS',
            'country_update': True, 'has_cog': False, 'has_chunks': False,
        }
        manifest = {'version': 1, 'items': [dict(item, status='pending')]}
        with patch('M2_mosaic_logic.mask_existing_country_cog') as mask_cog, \
             patch('M2_mosaic_logic.assemble_country_mosaic') as assemble, \
             patch('M2_mosaic_ui.save_manifest'):
            with self.assertRaisesRegex(RuntimeError, 'No COG or chunks'):
                M2_mosaic_ui.process_country_mosaic_item(item, manifest)

        mask_cog.assert_not_called()
        assemble.assert_not_called()
        self.assertEqual(manifest['items'][0]['status'], 'pending')


class CountryCogMaskTests(unittest.TestCase):
    def test_country_mask_keeps_grid_and_marks_exterior_invalid(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        geometry = {
            'type': 'Polygon',
            'coordinates': [[[0, 2], [2, 2], [2, 4], [0, 4], [0, 2]]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / 'source.tif'
            output_path = Path(temp_dir) / 'masked.tif'
            profile = {
                'driver': 'GTiff', 'height': 4, 'width': 4, 'count': 1,
                'dtype': 'uint8', 'crs': 'EPSG:4326',
                'transform': from_origin(0, 4, 1, 1),
            }
            with rasterio.open(source_path, 'w', **profile) as source:
                source.write(np.full((4, 4), 7, dtype='uint8'), 1)

            mask_raster_to_country(
                str(source_path), str(output_path), geometry, window_size=2,
            )

            with rasterio.open(output_path) as masked:
                self.assertEqual((masked.width, masked.height), (4, 4))
                self.assertEqual(masked.transform, profile['transform'])
                valid = masked.dataset_mask() > 0
                self.assertEqual(int(valid.sum()), 4)
                self.assertTrue((masked.read(1)[valid] == 7).all())

    def test_existing_cog_mask_replaces_same_object_only_after_new_cog_builds(self):
        with tempfile.TemporaryDirectory() as temp_dir, \
             patch.object(M2_mosaic_logic, 'CONFIG', {
                 'bucket': 'bucket', 'asset_regions': 'regions', 'country': 'testland',
             }), \
             patch.object(M2_mosaic_logic, 'period_cog_path', return_value='cogs'), \
             patch.object(M2_mosaic_logic, 'mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M_gcs.blob_exists', return_value=True), \
             patch.object(M2_mosaic_logic, 'get_temp_dir', return_value=temp_dir), \
             patch.object(M2_mosaic_logic, 'check_m2_dependencies', return_value=[]), \
             patch.object(M2_mosaic_logic, 'download') as download, \
             patch.object(M2_mosaic_logic, '_country_geometry_geojson', return_value={'type': 'Polygon'}), \
             patch.object(M2_mosaic_logic, 'mask_raster_to_country') as mask, \
             patch.object(M2_mosaic_logic, 'run_cmd') as build_cog, \
             patch.object(M2_mosaic_logic, 'upload') as upload, \
             patch('M1_update_manifest.complete_gcs_item'):
            result = M2_mosaic_logic.mask_existing_country_cog(
                '2025_01', 'red', 'sentinel2', 'minnbr',
            )

        self.assertTrue(result)
        remote = 'gs://bucket/cogs/mosaic_red_cog.tif'
        download.assert_called_once_with(remote, download.call_args.args[1])
        mask.assert_called_once()
        build_cog.assert_called_once()
        upload.assert_called_once_with(upload.call_args.args[0], 'bucket/cogs/mosaic_red_cog.tif')


if __name__ == '__main__':
    unittest.main()
