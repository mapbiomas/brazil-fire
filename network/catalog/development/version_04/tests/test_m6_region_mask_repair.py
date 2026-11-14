"""Offline tests for safe discovery and GEE regional-mask repair planning."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M6_region_mask_repair as repair


CONFIG = {
    'country': 'peru',
    'bucket': 'mapbiomas-fire',
    'gcs_campaigns_prefix': 'sudamerica/peru/CATALOG_01',
    'gee_campaigns_prefix': 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01',
    'asset_regions': 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01/MONITOR_DEV/AUXILIARY/regions',
}


class ClassificationIdentityTests(unittest.TestCase):
    def test_metadata_resolves_model_region_period_and_campaign(self):
        item = repair.resolve_classification_identity(
            f"{CONFIG['gee_campaigns_prefix']}/MONITOR_DEV/"
            "LIBRARY_CLASSIFICATIONS/REGIONAL/model_x/region_alpha_2025_09",
            {
                'model_id': 'model_x', 'region': 'Region Alpha',
                'period': '2025_09', 'campaign': 'MONITOR_DEV',
            },
            ['Region Alpha'], 'MONITOR_DEV',
        )

        self.assertEqual(item, {
            'asset_id': (
                f"{CONFIG['gee_campaigns_prefix']}/MONITOR_DEV/"
                "LIBRARY_CLASSIFICATIONS/REGIONAL/model_x/region_alpha_2025_09"
            ),
            'campaign': 'MONITOR_DEV', 'model_id': 'model_x',
            'region': 'Region Alpha', 'period': '2025_09',
        })

    def test_legacy_asset_identity_uses_exact_region_prefix(self):
        asset_id = (
            f"{CONFIG['gee_campaigns_prefix']}/MONITOR_DEV/"
            "LIBRARY_CLASSIFICATIONS/REGIONAL/model_x/region_alpha_north_2025_09"
        )
        item = repair.resolve_classification_identity(
            asset_id, {}, ['Region Alpha', 'Region Alpha North'], 'MONITOR_DEV',
        )
        self.assertEqual(item['region'], 'Region Alpha North')
        self.assertEqual(item['period'], '2025_09')

    def test_unresolvable_or_wrong_campaign_assets_are_skipped(self):
        root = f"{CONFIG['gee_campaigns_prefix']}/MONITOR_DEV/LIBRARY_CLASSIFICATIONS/REGIONAL/model_x/"
        self.assertIsNone(repair.resolve_classification_identity(
            root + 'not_a_region_2025_09', {}, ['Region Alpha'], 'MONITOR_DEV',
        ))
        self.assertIsNone(repair.resolve_classification_identity(
            root + 'region_alpha_2025_09', {'campaign': 'MONITOR_01'},
            ['Region Alpha'], 'MONITOR_DEV',
        ))


class DiscoveryTests(unittest.TestCase):
    def test_discovery_lists_images_under_models_and_does_not_write(self):
        root = repair._campaign_root(CONFIG, 'MONITOR_DEV')
        model = f'{root}/model_x'
        image_id = f'{model}/Region_Alpha_2025_09'
        ee = MagicMock()
        ee.FeatureCollection.return_value.aggregate_array.return_value.distinct.return_value.getInfo.return_value = [
            'Region Alpha'
        ]
        ee.data.listAssets.side_effect = [
            {'assets': [{'name': model, 'type': 'IMAGE_COLLECTION'}]},
            {'assets': [{'name': image_id, 'type': 'IMAGE'}]},
        ]
        ee.data.getAsset.return_value = {
            'name': image_id,
            'type': 'IMAGE',
            'properties': {
                'model_id': 'model_x', 'region': 'Region Alpha',
                'period': '2025_09', 'campaign': 'MONITOR_DEV',
            },
        }

        inventory = repair.discover_gee_classifications(
            'MONITOR_DEV', config=CONFIG, ee_api=ee,
        )

        self.assertEqual(len(inventory['items']), 1)
        self.assertEqual(inventory['items'][0]['asset_id'], image_id)
        self.assertEqual(inventory['skipped'], [])
        ee.batch.Export.image.toAsset.assert_not_called()


class ExportTaskTests(unittest.TestCase):
    def test_task_masks_region_preserves_properties_projection_and_overwrites_same_id(self):
        item = {
            'asset_id': 'projects/test/assets/model_x/Region_Alpha_2025_09',
            'campaign': 'MONITOR_DEV', 'model_id': 'model_x',
            'region': 'Region Alpha', 'period': '2025_09',
        }
        ee = MagicMock()
        features = ee.FeatureCollection.return_value.filter.return_value
        features.size.return_value.getInfo.return_value = 1
        features.geometry.return_value.bounds.return_value = 'bounds-geometry'
        source = ee.Image.return_value
        source.propertyNames.return_value.getInfo.return_value = [
            'campaign', 'system:time_start', 'system:time_end',
        ]
        source.select.return_value.projection.return_value.getInfo.return_value = {
            'crs': 'EPSG:32718', 'transform': [30, 0, 0, 0, -30, 0],
        }
        ee.batch.Export.image.toAsset.return_value = 'task'

        task = repair.build_overwrite_task(item, config=CONFIG, ee_api=ee)

        self.assertEqual(task, 'task')
        ee.Filter.eq.assert_called_once_with('region_nam', 'Region Alpha')
        source.updateMask.assert_called_once()
        copied = source.updateMask.return_value.copyProperties
        copied.assert_any_call(source)
        copied.return_value.copyProperties.assert_called_once_with(
            source, ['system:time_start', 'system:time_end'],
        )
        self.assertEqual(ee.batch.Export.image.toAsset.call_args.kwargs['assetId'], item['asset_id'])
        self.assertTrue(ee.batch.Export.image.toAsset.call_args.kwargs['overwrite'])
        self.assertEqual(ee.batch.Export.image.toAsset.call_args.kwargs['region'], 'bounds-geometry')
        self.assertEqual(ee.batch.Export.image.toAsset.call_args.kwargs['crsTransform'], [30, 0, 0, 0, -30, 0])

    def test_submit_requires_explicit_overwrite_confirmation(self):
        ee = MagicMock()
        with self.assertRaises(PermissionError):
            repair.submit_gee_repairs(
                [], 'MONITOR_DEV', config=CONFIG, ee_api=ee,
            )
        ee.data.getTaskList.assert_not_called()

    def test_submit_is_capped_and_saves_each_started_task(self):
        items = [
            {
                'asset_id': f'projects/test/assets/model_x/Region_{index}_2025_09',
                'campaign': 'MONITOR_DEV', 'model_id': 'model_x',
                'region': f'Region {index}', 'period': '2025_09',
            }
            for index in range(4)
        ]
        manifest = {'version': 1, 'repair_version': repair.REPAIR_VERSION, 'items': {}}
        ee = MagicMock()
        ee.data.getTaskList.return_value = []
        task = MagicMock()
        task.id = 'task-1'
        with patch.object(repair, 'load_manifest', return_value=manifest), \
             patch.object(repair, 'save_manifest') as save, \
             patch.object(repair, 'build_overwrite_task', return_value=task) as build:
            result = repair.submit_gee_repairs(
                items, 'MONITOR_DEV', confirmation='OVERWRITE MONITOR_DEV',
                config=CONFIG, ee_api=ee, max_submissions=1,
            )

        self.assertEqual(build.call_count, 1)
        task.start.assert_called_once()
        self.assertEqual(result['items'][items[0]['asset_id']]['status'], 'submitted')
        self.assertGreaterEqual(save.call_count, 2)

    def test_submit_skips_assets_already_marked_as_repaired(self):
        item = {
            'asset_id': 'projects/test/assets/model_x/Region_Alpha_2025_09',
            'campaign': 'MONITOR_DEV', 'model_id': 'model_x',
            'region': 'Region Alpha', 'period': '2025_09', 'already_masked': True,
        }
        manifest = {'version': 1, 'repair_version': repair.REPAIR_VERSION, 'items': {}}
        ee = MagicMock()
        ee.data.getTaskList.return_value = []
        with patch.object(repair, 'load_manifest', return_value=manifest), \
             patch.object(repair, 'save_manifest'), \
             patch.object(repair, 'build_overwrite_task') as build:
            result = repair.submit_gee_repairs(
                [item], 'MONITOR_DEV', confirmation='OVERWRITE MONITOR_DEV',
                config=CONFIG, ee_api=ee,
            )

        build.assert_not_called()
        self.assertEqual(result['items'][item['asset_id']]['status'], 'complete')

    def test_submit_pauses_when_the_project_task_limit_is_already_reached(self):
        item = {
            'asset_id': 'projects/test/assets/model_x/Region_Alpha_2025_09',
            'campaign': 'MONITOR_DEV', 'model_id': 'model_x',
            'region': 'Region Alpha', 'period': '2025_09',
        }
        manifest = {'version': 1, 'repair_version': repair.REPAIR_VERSION, 'items': {}}
        ee = MagicMock()
        ee.data.getTaskList.return_value = [
            {'id': 'active-1', 'state': 'RUNNING'},
            {'id': 'active-2', 'state': 'READY'},
        ]
        with patch.object(repair, 'load_manifest', return_value=manifest), \
             patch.object(repair, 'save_manifest'), \
             patch.object(repair, 'build_overwrite_task') as build:
            repair.submit_gee_repairs(
                [item], 'MONITOR_DEV', confirmation='OVERWRITE MONITOR_DEV',
                config=CONFIG, ee_api=ee, max_active_tasks=2,
            )

        build.assert_not_called()


if __name__ == '__main__':
    unittest.main()
