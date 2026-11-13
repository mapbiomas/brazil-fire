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
        button.click()
        callback.assert_called_once_with(checkbox)

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


if __name__ == '__main__':
    unittest.main()
