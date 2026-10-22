"""Offline tests for the catalog registry (campaigns x targets)."""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M0_auth_config
import M_cache
import M5_workplan
import M_campaigns
import M_catalog
import M_gcs
import M_targets


COUNTRIES = [
    'brazil', 'bolivia', 'colombia', 'chile',
    'ecuador', 'paraguay', 'venezuela',
]


class CampaignRegistryTests(unittest.TestCase):
    def test_every_country_has_monitor_and_annual(self):
        ids = {campaign.id for campaign in M_campaigns.list_campaigns()}
        for country in COUNTRIES:
            self.assertIn(f'{country}.monitor_01', ids)
            annual = [
                campaign_id for campaign_id in ids
                if campaign_id.startswith(f'{country}.collection_')
            ]
            self.assertEqual(len(annual), 1, f'{country} needs exactly one annual campaign')

    def test_monitor_is_monthly_sentinel2(self):
        for country in COUNTRIES:
            campaign = M_campaigns.get_campaign(f'{country}.monitor_01')
            self.assertEqual(campaign.periodicity, 'monthly')
            self.assertEqual(campaign.sensors, ('sentinel2',))

    def test_annual_is_yearly_landsat(self):
        for country in COUNTRIES:
            annual = M_campaigns.list_campaigns(country=country, source='collection')
            self.assertEqual(len(annual), 1)
            self.assertEqual(annual[0].periodicity, 'yearly')
            self.assertEqual(annual[0].sensors, ('landsat',))

    def test_paths_use_the_existing_layout(self):
        campaign = M_campaigns.get_campaign('brazil.monitor_01')
        paths = M_campaigns.campaign_paths(campaign)
        self.assertEqual(
            paths['models'],
            'sudamerica/brazil/CATALOG_01/MONITOR_01/LIBRARY_MODELS',
        )
        self.assertEqual(
            paths['classifications'],
            'sudamerica/brazil/CATALOG_01/MONITOR_01/LIBRARY_CLASSIFICATIONS',
        )

    def test_cache_paths_are_split(self):
        campaign = M_campaigns.get_campaign('brazil.monitor_01')
        paths = M_campaigns.campaign_paths(campaign)
        self.assertEqual(
            paths['library_cache'],
            'sudamerica/brazil/CATALOG_01/LIBRARY_IMAGES/.CACHE',
        )
        self.assertEqual(
            paths['cache'],
            'sudamerica/brazil/CATALOG_01/MONITOR_01/.CACHE',
        )

    def test_unknown_campaign_raises(self):
        with self.assertRaises(KeyError):
            M_campaigns.get_campaign('atlantis.monitor_01')


class TargetRegistryTests(unittest.TestCase):
    def test_only_burned_area_is_enabled(self):
        self.assertEqual([target.id for target in M_targets.list_targets()], ['burned_area'])

    def test_burned_area_defaults(self):
        target = M_targets.get_target('burned_area')
        self.assertEqual(target.task, 'binary')
        self.assertEqual(target.label_column, 'fire')
        self.assertEqual(target.probability_band, 'probability')
        self.assertEqual(target.auxiliary_band, 'dayOfYear')
        self.assertEqual(target.stats_prefix, 'burned')
        self.assertEqual(target.asset_suffix, 'burned_area')

    def test_require_supported_rejects_disabled_or_non_binary(self):
        disabled = M_targets.Target(id='x', label='X', enabled=False)
        with self.assertRaises(ValueError):
            M_targets.require_supported(disabled)
        multiclass = M_targets.Target(id='y', label='Y', enabled=True, task='multiclass')
        with self.assertRaises(NotImplementedError):
            M_targets.require_supported(multiclass)


class CatalogTests(unittest.TestCase):
    def test_registry_is_consistent(self):
        self.assertEqual(M_catalog.validate_registry(), [])

    def test_matrix_covers_every_campaign(self):
        rows = M_catalog.matrix()
        campaigns = {row['campaign'] for row in rows}
        self.assertEqual(campaigns, {c.id for c in M_campaigns.list_campaigns()})
        self.assertTrue(all(row['target'] == 'burned_area' for row in rows))

    def test_describe_without_gcs(self):
        info = M_catalog.describe('chile.collection_02')
        self.assertEqual(info['campaign']['id'], 'chile.collection_02')
        self.assertEqual(info['targets'][0]['id'], 'burned_area')
        self.assertIsNone(info['data'])

    def test_classify_plan_skeleton(self):
        plan = M_catalog.classify_plan(
            'bolivia.collection_02', 'burned_area', ['2025'], model='training_001'
        )
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['campaign'], 'bolivia.collection_02')
        self.assertEqual(plan[0]['period'], '2025')


class SetCampaignTests(unittest.TestCase):
    def test_reads_config_after_set_global_opts_rebinds_it(self):
        fake = types.ModuleType('M0_auth_config')
        fake.CONFIG = None

        def fake_set_global_opts(**kwargs):
            fake.CONFIG = {
                'country': kwargs['country'],
                'campaign': kwargs['campaign'],
            }

        fake.set_global_opts = fake_set_global_opts
        with patch.dict(sys.modules, {'M0_auth_config': fake}):
            config = M_campaigns.set_campaign('brazil.collection_06')

        self.assertEqual(config['campaign_id'], 'brazil.collection_06')
        self.assertEqual(config['target_id'], 'burned_area')


class TareasPathTests(unittest.TestCase):
    def test_tareas_use_campaign_cache(self):
        fake_config = {
            'gcs_cache': 'sudamerica/brazil/CATALOG_01/.CACHE',
            'gcs_campaign_cache': 'sudamerica/brazil/CATALOG_01/MONITOR_01/.CACHE',
            'campaign': 'MONITOR_01',
        }
        with patch.object(M5_workplan, 'CONFIG', fake_config):
            self.assertEqual(
                M5_workplan.tareas_dir(),
                'sudamerica/brazil/CATALOG_01/MONITOR_01/.CACHE/tareas',
            )
            self.assertEqual(
                M5_workplan.legacy_tareas_dir(),
                'sudamerica/brazil/CATALOG_01/.CACHE/tareas',
            )


class CacheSplitTests(unittest.TestCase):
    def test_library_and_campaign_caches_are_separate(self):
        fake_config = {
            'country': 'brazil',
            'bucket': 'bucket',
            'campaign': 'MONITOR_01',
            'gcs_library_images_prefix': 'sudamerica/brazil/CATALOG_01',
            'gcs_campaigns_prefix': 'sudamerica/brazil/CATALOG_01',
            'gee_library_images_prefix': 'projects/mapbiomas-brazil/assets/FIRE/CATALOG_01',
            'gee_campaigns_prefix': 'projects/mapbiomas-brazil/assets/FIRE/CATALOG_01',
        }
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                with patch.object(M_cache, 'CONFIG', fake_config), \
                     patch.object(M0_auth_config, 'CONFIG', fake_config), \
                     patch.object(M_gcs, 'write_json', lambda *a, **k: None):
                    M_cache.CacheManager.reset()
                    state = {
                        'assets_monthly': ['asset_m'],
                        'assets_annually': ['asset_a'],
                        'gcs_chunks': {'chunk': ['red']},
                        'cogs_monthly': ['cog_m'],
                        'cogs_annually': ['cog_a'],
                        'trained_models': ['training_001'],
                        'sample_collections': ['samples_001'],
                        'classified_groups': [],
                    }
                    M_cache.CacheManager.save(state)

                    with open('library.json', encoding='utf-8') as handle:
                        library = json.load(handle)
                    with open('campaign.json', encoding='utf-8') as handle:
                        campaign = json.load(handle)

                    self.assertEqual(library['assets_annually'], ['asset_a'])
                    self.assertNotIn('trained_models', library)
                    self.assertEqual(campaign['trained_models'], ['training_001'])
                    self.assertNotIn('assets_annually', campaign)

                    M_cache.CacheManager.reset()
                    merged = M_cache.CacheManager.load(force=True)
                    self.assertEqual(merged['assets_annually'], ['asset_a'])
                    self.assertEqual(merged['trained_models'], ['training_001'])

                    M_cache.CacheManager.mark_stale()
                    with open('library.json', encoding='utf-8') as handle:
                        library = json.load(handle)
                    self.assertIsNone(library['updated_at'])
            finally:
                os.chdir(old_cwd)
                M_cache.CacheManager.reset()


if __name__ == '__main__':
    unittest.main()
