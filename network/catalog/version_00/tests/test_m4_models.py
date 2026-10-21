"""Offline tests for M4 model listing and model-id resolution."""

import os
import sys
import unittest
from unittest.mock import patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M4_hub_manager as hub


class ResolveModelIdTests(unittest.TestCase):
    def test_string(self):
        self.assertEqual(hub.resolve_model_id('training_001_x_landsat'), 'training_001_x_landsat')

    def test_dict_prefers_id(self):
        info = {'id': 'training_001_x_landsat', 'training_id': '001'}
        self.assertEqual(hub.resolve_model_id(info), 'training_001_x_landsat')

    def test_dict_falls_back_to_training_id(self):
        self.assertEqual(hub.resolve_model_id({'training_id': 'training_001_x_landsat'}),
                         'training_001_x_landsat')

    def test_missing_identifier(self):
        self.assertIsNone(hub.resolve_model_id({}))
        self.assertIsNone(hub.resolve_model_id(None))


class ResolveModelFolderTests(unittest.TestCase):
    def test_full_id_is_kept(self):
        info = {'id': 'training_001_x_landsat', 'training_id': '001'}
        self.assertEqual(hub.resolve_model_folder(info), 'training_001_x_landsat')

    def test_numeric_training_id_builds_folder_name(self):
        info = {'training_id': '001', 'shortname': 'x', 'sensor': 'landsat'}
        self.assertEqual(hub.resolve_model_folder(info), 'training_001_x_landsat')

    def test_numeric_training_id_without_shortname_is_kept(self):
        self.assertEqual(hub.resolve_model_folder({'training_id': '001'}), '001')


class NormalizeModelsTests(unittest.TestCase):
    def test_mixed_entries_are_unique_strings(self):
        values = [
            'training_001_a_landsat',
            {'training_id': 'training_002_b_landsat'},
            {'id': 'training_003_c_landsat'},
            'training_001_a_landsat',
            None,
        ]
        self.assertEqual(hub._normalize_models(values), [
            'training_001_a_landsat',
            'training_002_b_landsat',
            'training_003_c_landsat',
        ])


class FakeFS:
    MODELS = [
        'bucket/prefix/CAMPAIGN_01/LIBRARY_MODELS/training_001_a_landsat',
        'bucket/prefix/CAMPAIGN_01/LIBRARY_MODELS/training_002_b_landsat',
    ]

    def __init__(self, metadata_for=()):
        self.metadata_for = set(metadata_for)
        self.removed = []

    def exists(self, path):
        if path.endswith('/metadata.json'):
            parent = path[:-len('/metadata.json')].rstrip('/')
            return parent in self.metadata_for
        return True

    def ls(self, path):
        return list(self.MODELS)

    def glob(self, pattern):
        base = pattern.split('/*/')[0]
        return sorted(
            f"{model}/metadata.json"
            for model in self.metadata_for
            if model.startswith(base + '/')
        )

    def rm(self, path, recursive=False):
        self.removed.append(path)


class FakeCache:
    @staticmethod
    def get_state():
        return {'trained_models': []}

    @staticmethod
    def save(*args, **kwargs):
        return None


class ListTrainedModelsTests(unittest.TestCase):
    def _run(self, fs):
        with patch.object(hub, 'CacheManager', FakeCache), \
             patch.object(hub, '_get_fs', lambda: fs), \
             patch.object(hub, 'CONFIG', {
                 'bucket': 'bucket',
                 'campaign': 'campaign',
                 'gcs_campaigns_prefix': 'prefix',
             }), \
             patch.object(M0_auth_config, '_gcs_models_base',
                          return_value='prefix/CAMPAIGN_01/LIBRARY_MODELS'), \
             patch.object(hub, '_load_m4_metadata', return_value={}), \
             patch.object(hub, '_save_m4_metadata'):
            return hub.list_trained_models(force_refresh=True)

    def test_missing_metadata_is_skipped_without_deleting(self):
        fs = FakeFS()
        self.assertEqual(self._run(fs), [])
        self.assertEqual(fs.removed, [])

    def test_only_models_with_metadata_are_listed(self):
        fs = FakeFS(metadata_for={'bucket/prefix/CAMPAIGN_01/LIBRARY_MODELS/training_001_a_landsat'})
        self.assertEqual(self._run(fs), ['training_001_a_landsat'])
        self.assertEqual(fs.removed, [])


class RegisterModelTests(unittest.TestCase):
    def test_appends_model_and_stores_metadata(self):
        state = {'trained_models': ['training_001_a_landsat']}
        saved = {}

        class FakeCache:
            @staticmethod
            def get_state():
                return state

            @staticmethod
            def save(*args, **kwargs):
                return None

        with patch.object(hub, 'CacheManager', FakeCache), \
             patch.object(hub, '_load_m4_metadata', return_value={}), \
             patch.object(hub, '_save_m4_metadata',
                          side_effect=lambda data: saved.update(data)), \
             patch.object(M0_auth_config, 'gcs_models_path',
                          return_value='prefix/CAMPAIGN_01/LIBRARY_MODELS'), \
             patch.object(hub, 'CONFIG', {'bucket': 'bucket'}):
            trained = hub.register_model('training_002_b_landsat', {'shortname': 'b'})

        self.assertEqual(trained, ['training_001_a_landsat', 'training_002_b_landsat'])
        self.assertEqual(state['trained_models'], trained)
        self.assertEqual(saved['metadata']['training_002_b_landsat'], {'shortname': 'b'})
        self.assertTrue(
            saved['meta']['training_002_b_landsat']['path'].endswith(
                '/training_002_b_landsat'
            )
        )


if __name__ == '__main__':
    unittest.main()
