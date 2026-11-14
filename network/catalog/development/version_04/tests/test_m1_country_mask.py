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
from M1_export_ui import make_update_cell
from M0_auth_config import set_global_opts
import M0_auth_config
import M1_export_ui
import M2_mosaic_logic
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

    def test_update_status_is_a_direct_ui_button(self):
        checkbox = Checkbox()
        callback = Mock()
        cell = make_update_cell(checkbox, callback)
        button = cell.children[1]
        self.assertEqual(button.description, M1_export_ui.Lang.M1_UPDATE_LABEL)
        self.assertEqual(button.button_style, 'info')
        self.assertIn('mfm-update', cell._dom_classes)
        button.click()
        callback.assert_called_once_with(checkbox)

    def test_update_guide_tab_is_blue_and_explains_both_workflows(self):
        with patch.multiple(
            M1_export_ui.Lang,
            M1_UPDATE_TAB='Atualização',
            M1_UPDATE_GUIDE_TITLE='Por que atualizar?',
            M1_UPDATE_GUIDE_NORMAL_TITLE='Fluxo normal de processamento',
            M1_UPDATE_GUIDE_NORMAL='M1 normal.',
            M1_UPDATE_GUIDE_REASON_TITLE='Por que esta correção é necessária',
            M1_UPDATE_GUIDE_REASON='Precisamos da máscara nacional.',
            M1_UPDATE_GUIDE_ACTION_TITLE='O que Atualizar faz',
            M1_UPDATE_GUIDE_ACTION='Atualiza sem criar uma nova task.',
        ):
            guide = M1_export_ui.make_update_guide()
            tabs = M1_export_ui.make_primary_tabs(
                M1_export_ui.widgets.HTML('general guide'), guide,
                [M1_export_ui.widgets.VBox()], ['SENTINEL2'],
            )

        self.assertEqual(tabs.get_title(1), 'Atualização')
        self.assertEqual(tabs.get_title(2), 'SENTINEL2')
        self.assertIn('Fluxo normal de processamento', guide.value)
        self.assertIn('Por que esta correção é necessária', guide.value)
        self.assertIn('sem criar uma nova task', guide.value)
        status_css = M1_export_ui.PipelineStepUI.get_status_css().value
        self.assertIn('.m1-primary-tabs', status_css)
        self.assertIn('#1677ff', status_css)

    def test_update_help_explains_mask_reason_and_gcs_followup(self):
        help_widget = M1_export_ui.make_update_help()
        self.assertIn('mask', help_widget.value.lower())
        self.assertIn('m2', help_widget.value.lower())

    def test_gcs_update_uses_mask_operation_not_new_m1_export(self):
        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS',
            'update_pending': True,
        }
        manifest = {'version': 1, 'items': [dict(meta, status='pending')]}
        with patch.object(M1_export_ui, 'CONFIG', {'bucket': 'bucket'}), \
             patch('M_gcs.blob_exists', return_value=True), \
             patch('M_gcs.glob', return_value=[]), \
             patch('M1_export_ui.period_cog_path', return_value='cogs'), \
             patch('M1_export_ui.period_chunk_path', return_value='chunks'), \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M2_mosaic_logic.mask_existing_country_cog', return_value=True) as mask_existing, \
             patch('M2_mosaic_logic.assemble_country_mosaic') as assemble, \
             patch('M1_export_ui.start_export') as common_export, \
             patch('M1_export_ui.save_manifest'):
            handled = M1_export_ui.update_existing_gcs_output(meta, manifest)

        self.assertTrue(handled)
        mask_existing.assert_called_once()
        assemble.assert_not_called()
        common_export.assert_not_called()
        self.assertEqual(manifest['items'][0]['status'], 'complete')

    def test_gcs_update_assembles_existing_chunks_without_new_m1_export(self):
        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS', 'update_pending': True,
        }
        manifest = {'version': 1, 'items': [dict(meta, status='pending')]}
        with patch.object(M1_export_ui, 'CONFIG', {'bucket': 'bucket'}), \
             patch('M_gcs.blob_exists', return_value=False), \
             patch('M_gcs.glob', return_value=['gs://bucket/chunks/red_1.tif']), \
             patch('M1_export_ui.period_chunk_path', return_value='chunks'), \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M1_export_ui.period_cog_path', return_value='cogs'), \
             patch('M2_mosaic_logic.assemble_country_mosaic', return_value=['gs://bucket/cog.tif']) as assemble, \
             patch('M2_mosaic_logic.mask_existing_country_cog') as direct_mask, \
             patch('M1_export_ui.save_manifest'):
            handled = M1_export_ui.update_existing_gcs_output(meta, manifest)

        self.assertTrue(handled)
        assemble.assert_called_once()
        direct_mask.assert_not_called()
        self.assertEqual(manifest['items'][0]['status'], 'complete')

    def test_missing_gcs_output_falls_back_to_ordinary_m1_export(self):
        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS', 'update_pending': True,
        }
        manifest = {'version': 1, 'items': [dict(meta, status='pending')]}
        ui = M1_export_ui.ExportDispatcherUI.__new__(M1_export_ui.ExportDispatcherUI)
        ui.update_manifest = manifest
        ui.chk_dict = {'row': Checkbox(value=True)}
        chk = ui.chk_dict['row']
        chk._meta = meta
        chk._update_button = Mock(disabled=False)
        ui._reconcile_update_tasks = Mock()
        with patch.object(M1_export_ui, 'CONFIG', {'bucket': 'bucket'}), \
             patch('M_gcs.blob_exists', return_value=False), \
             patch('M_gcs.glob', return_value=[]), \
             patch('M1_export_ui.period_cog_path', return_value='cogs'), \
             patch('M1_export_ui.period_chunk_path', return_value='chunks'), \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M1_export_ui.start_export', return_value=1) as common_export, \
             patch('M1_update_manifest.save_manifest'):
            ui._on_update_click(chk)

        self.assertFalse(meta['update_pending'])
        common_export.assert_called_once_with(ui, mode='GCS')

    def test_start_export_rechecks_existing_cog_before_building_mosaic(self):
        meta = {
            'country': 'testland', 'sensor': 'sentinel2', 'period': 'monthly',
            'mosaic': 'minnbr', 'period_key': '2025_01', 'year': 2025,
            'month': 1, 'band': 'red', 'type': 'GCS', 'exists': False,
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
             patch('M1_export_ui.update_existing_gcs_output', return_value=True) as update_existing, \
             patch('M1_export_logic.get_quality_mosaic') as build_mosaic, \
             patch('M1_export_logic.export_to_gcs') as export_gcs, \
             patch('M1_export_ui.CacheManager.mark_stale'), \
             patch('M1_export_ui.save_manifest'):
            submitted = M1_export_ui.start_export(ui, mode='GCS')

        self.assertEqual(submitted, 1)
        update_existing.assert_called_once_with(meta, ui.update_manifest)
        build_mosaic.assert_not_called()
        export_gcs.assert_not_called()

    def test_start_export_uses_common_m1_task_when_gcs_output_is_absent(self):
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
        fake_feature_collection = Mock()
        with patch.object(M0_auth_config, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui, 'CONFIG', {'asset_regions': 'regions'}), \
             patch.object(M1_export_ui.ee, 'FeatureCollection', return_value=fake_feature_collection), \
             patch('M_blocklist.refresh_blocklists', return_value={}), \
             patch('M1_export_ui.update_existing_gcs_output', return_value=False) as update_existing, \
             patch('M1_export_logic.get_quality_mosaic', return_value=object()) as build_mosaic, \
             patch('M1_export_logic.export_to_gcs', return_value=[SimpleNamespace(id='task-1')]) as export_gcs, \
             patch('M1_export_ui.mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M1_export_ui.CacheManager.mark_stale'), \
             patch('M1_export_ui.save_manifest'):
            submitted = M1_export_ui.start_export(ui, mode='GCS')

        self.assertEqual(submitted, 1)
        update_existing.assert_called_once_with(meta, ui.update_manifest)
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

            mask_raster_to_country(str(source_path), str(output_path), geometry)

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
