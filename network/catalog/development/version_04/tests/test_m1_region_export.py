"""Offline tests for experimental regional asset submission."""
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M1_region_export as regional
import M1_export_logic as logic
import M0_auth_config as config


class RegionalAssetTests(unittest.TestCase):
    def test_stable_distinct_suffixes(self):
        self.assertEqual(regional.region_asset_suffix('Ação'), regional.region_asset_suffix('Ação'))
        self.assertNotEqual(regional.region_asset_suffix('Ação'), regional.region_asset_suffix('Acao'))

    def test_invalid_region_names_rejected(self):
        for names in ([], ['a', 'a'], ['a', None], [' a']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                regional.validate_region_names(names)

    def test_one_task_per_region_without_clip(self):
        image, collection, task = MagicMock(), MagicMock(), MagicMock()
        collection.size.return_value.getInfo.return_value = 2
        collection.aggregate_array.return_value.getInfo.return_value = ['Norte', 'Sul']
        with patch.object(regional, 'list_collection_asset_names', return_value=set()), \
             patch.object(regional.ee.data, 'getTaskList', return_value=[]), \
             patch.object(regional.ee.Filter, 'eq', side_effect=lambda *x: x), \
             patch.object(regional.ee.batch.Export.image, 'toAsset', return_value=task) as export:
            result = regional.submit_region_exports(image, collection, 'projects/p/assets/c',
                'image_test_blue_2025', 'blue', '2025', {'year': 2025}, 30, 'TEST')
        self.assertEqual(len(result['started']), 2)
        self.assertEqual(task.start.call_count, 2)
        self.assertEqual(export.call_count, 2)
        image.clip.assert_not_called()
        self.assertNotIn('overwrite', export.call_args.kwargs)
        self.assertEqual(image.select.return_value.set.call_args.kwargs, {})
        self.assertEqual(image.select.return_value.set.call_args.args[0]['expected_region_count'], 2)

    def test_missing_identifier_fails_before_submitting(self):
        collection = MagicMock()
        collection.size.return_value.getInfo.return_value = 2
        collection.aggregate_array.return_value.getInfo.return_value = ['Norte']
        with patch.object(regional.ee.batch.Export.image, 'toAsset') as export:
            with self.assertRaises(ValueError):
                regional.submit_region_exports(MagicMock(), collection, 'c', 'base',
                                               'blue', '2025', {}, 30, 'TEST')
        export.assert_not_called()

    def test_m1_wrapper_passes_temporal_metadata_and_region_collection(self):
        image = MagicMock()
        regions = MagicMock()
        date = MagicMock()
        date.millis.return_value = 123
        settings = {'country': 'colombia', 'asset_regions': 'regions'}
        with patch.object(config, 'CONFIG', settings), \
             patch.object(config, 'get_country_geometry', return_value='country'), \
             patch.object(config, 'get_asset_mosaic_collection', return_value='collection'), \
             patch.object(config, 'get_sensor_scale', return_value=30), \
             patch.object(logic, 'ensure_asset_path'), \
             patch.object(logic, 'mask_to_geometry', return_value=image), \
             patch.object(logic.ee, 'Date', return_value=date), \
             patch.object(logic.ee, 'FeatureCollection', return_value=regions), \
             patch.object(regional, 'submit_region_exports', return_value={'started': [], 'skipped': []}) as submit:
            logic.export_to_regions(image, 'base', 2025, period='yearly', band='blue',
                                    sensor='landsat', period_key='2025')
        kwargs = submit.call_args.kwargs
        self.assertEqual(kwargs['properties']['year'], 2025)
        self.assertEqual(kwargs['properties']['month'], 0)
        self.assertEqual(kwargs['period_key'], '2025')
        self.assertIs(kwargs['collection'], regions)

    def test_existing_and_active_are_skipped(self):
        collection = MagicMock()
        collection.size.return_value.getInfo.return_value = 2
        collection.aggregate_array.return_value.getInfo.return_value = ['Norte', 'Sul']
        first, second = [f'base_{regional.region_asset_suffix(n)}' for n in ['Norte', 'Sul']]
        with patch.object(regional, 'list_collection_asset_names', return_value={'c/' + first}), \
             patch.object(regional.ee.data, 'getTaskList', return_value=[
                 {'description': 'TEST_ASSET_' + second, 'state': 'RUNNING'}]), \
             patch.object(regional.ee.batch.Export.image, 'toAsset') as export:
            result = regional.submit_region_exports(MagicMock(), collection, 'c', 'base',
                                                    'blue', '2025', {}, 30, 'TEST')
        self.assertEqual(len(result['skipped']), 2)
        export.assert_not_called()
