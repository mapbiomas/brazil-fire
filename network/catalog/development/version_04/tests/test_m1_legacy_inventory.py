"""Offline contract tests for legacy inventory."""
import datetime as dt
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from inventory_m1_legacy import inventory, parse_timestamp, main

UTC = dt.timezone.utc
CUTOFF = dt.datetime(2026, 11, 15, 2, 59, 59, tzinfo=UTC)
BASE = 'root/LIBRARY_IMAGES/LANDSAT/ANNUALLY/MINNBR/2025/'
NAME = 'image_testland_fire_landsat_minnbr_blue_2025'


def blob(name, created=CUTOFF, generation=1):
    return SimpleNamespace(name=name, time_created=created, generation=generation,
                           updated=created, metageneration=1)


class Client:
    def __init__(self, objects, fail=False):
        self.objects, self.fail = objects, fail

    def list_blobs(self, bucket, prefix):
        for obj in self.objects:
            if obj.name.startswith(prefix):
                yield obj
        if self.fail:
            raise RuntimeError('listing failed')


def run(objects, **kwargs):
    return inventory(Client(objects, kwargs.pop('fail', False)), 'example-bucket',
                     ['root/'], 'testland', ['campaign_a', 'campaign_a', 'campaign_b'],
                     'landsat', 'minnbr', CUTOFF, **kwargs)


class InventoryTests(unittest.TestCase):
    def test_complete_manifest_item_is_not_a_candidate(self):
        entry = dict(country='testland', sensor='landsat', period='yearly',
                     mosaic='minnbr', period_key='2025', band='blue',
                     type='GCS', status='complete')
        result = run([blob(BASE + 'COG/' + NAME + '_cog.tif')],
                     manifest={'version': 1, 'items': [entry]})
        self.assertTrue(result['ready'])
        self.assertEqual(result['candidates'], [])
        self.assertEqual(len(result['already_complete']), 1)
        self.assertEqual(result['observed_groups'][0]['manifest_status'], 'complete')

    def test_distinct_cogs_for_same_identity_conflict(self):
        first = blob(BASE + 'COG/' + NAME + '_cog.tif')
        second = blob(BASE + 'COG/' + NAME.replace('_blue_', '_BLUE_') + '_cog.tif')
        result = run([first, second])
        self.assertFalse(result['ready'])
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['observed_groups'][0]['cogs']), 2)
        self.assertTrue(result['conflicts'])

    def test_equivalent_chunk_tiles_conflict(self):
        first = blob(BASE + 'CHUNKS/' + NAME + '_0000000000-0000000000.tif')
        second = blob(BASE + 'CHUNKS/' + NAME.replace('_blue_', '_BLUE_') +
                      '_0000000000-0000000000.tif')
        result = run([first, second])
        self.assertFalse(result['ready'])
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['observed_groups'][0]['chunks']), 2)
        self.assertTrue(result['conflicts'])

    def test_later_metadata_and_later_cog_require_review(self):
        chunk = blob(BASE + 'CHUNKS/' + NAME + '_0000000000-0000000000.tif')
        chunk.updated = CUTOFF + dt.timedelta(microseconds=1)
        chunk.metageneration = 2
        cog = blob(BASE + 'COG/' + NAME + '_cog.tif', CUTOFF + dt.timedelta(seconds=1))
        result = run([chunk, cog])
        self.assertFalse(result['ready'])
        self.assertEqual(result['candidates'], [])
        self.assertEqual(len(result['observed_groups'][0]['chunks']), 1)
        self.assertEqual(len(result['observed_groups'][0]['cogs']), 1)
        self.assertEqual(result['observed_groups'][0]['chunks'][0]['metageneration'], '2')

    def test_legacy_unresolved_blocks_safe_candidates(self):
        legacy = blob(BASE + 'CHUNKS/minnbr_landsat_fire_testland_2025_blue_0000000000-0000000000.tif')
        normal = blob(BASE + 'COG/' + NAME + '_cog.tif')
        result = run([normal, legacy])
        self.assertFalse(result['ready'])
        self.assertFalse(result['candidates'])
        self.assertEqual(result['unresolved'][0]['warning'], 'legacy_or_unrecognized_name')
        self.assertEqual(len(result['observed_groups']), 1)

    def test_missing_metadata_and_incomplete_scan_block_candidates(self):
        item = blob(BASE + 'COG/' + NAME + '_cog.tif')
        item.updated = None
        result = run([item])
        self.assertFalse(result['ready'])
        self.assertFalse(result['candidates'])
        result = run([blob(item.name)], fail=True)
        self.assertFalse(result['ready'])
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['observed_groups']), 1)

    def test_cli_exit_status_and_report_offline(self):
        item = blob(BASE + 'COG/' + NAME + '_cog.tif')
        fake_storage = SimpleNamespace(Client=lambda: Client([item], fail=True))
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'audit.json')
            args = ['--bucket', 'example-bucket', '--country', 'testland',
                    '--campaign', 'campaign_a', '--sensor', 'landsat', '--method', 'minnbr',
                    '--prefix', 'root/', '--cutoff', '2026-11-15T02:59:59Z', '--output', output]
            with mock.patch.dict(sys.modules, {'google': SimpleNamespace(cloud=SimpleNamespace(storage=fake_storage)),
                                              'google.cloud': SimpleNamespace(storage=fake_storage)}):
                self.assertEqual(main(args), 1)
            with open(output, encoding='utf-8') as stream:
                report = json.load(stream)
            self.assertFalse(report['ready'])
            self.assertEqual(report['candidates'], [])
            self.assertTrue(report['errors'])

    def test_cli_success_and_review_exit_codes(self):
        for objects, expected in [
            ([blob(BASE + 'COG/' + NAME + '_cog.tif')], 0),
            ([blob('root/unrecognized.tif')], 1),
            ([blob(BASE + 'COG/' + NAME + '_cog.tif',
                   CUTOFF + dt.timedelta(microseconds=1))], 1),
        ]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                output = os.path.join(directory, 'audit.json')
                args = ['--bucket', 'example-bucket', '--country', 'testland',
                        '--campaign', 'campaign_a', '--sensor', 'landsat', '--method', 'minnbr',
                        '--prefix', 'root/', '--cutoff', '2026-11-14T23:59:59-03:00', '--output', output]
                fake = SimpleNamespace(Client=lambda: Client(objects))
                with mock.patch.dict(sys.modules, {'google': SimpleNamespace(cloud=SimpleNamespace(storage=fake)),
                                                  'google.cloud': SimpleNamespace(storage=fake)}):
                    self.assertEqual(main(args), expected)
                with open(output, encoding='utf-8') as stream:
                    report = json.load(stream)
                self.assertEqual(report['ready'], expected == 0)
                self.assertEqual(report['cutoff'], '2026-11-15T02:59:59Z')

    def test_metadata_boundaries_and_timezones(self):
        item = blob(BASE + 'COG/' + NAME + '_cog.tif')
        item.updated = dt.datetime(2026, 11, 14, 23, 59, 59,
                                   tzinfo=dt.timezone(dt.timedelta(hours=-3)))
        result = run([item])
        self.assertTrue(result['ready'])
        self.assertEqual(result['candidates'][0]['cogs'][0]['updated'], '2026-11-15T02:59:59Z')
        item.metageneration = 2
        self.assertFalse(run([item])['ready'])
        item.metageneration = 1
        item.updated = dt.datetime(2026, 11, 15, 2, 59, 59)
        self.assertEqual(run([item])['unresolved'][0]['warning'], 'invalid_or_missing_metadata')

    def test_cli_rejects_naive_cutoff_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'audit.json')
            args = ['--bucket', 'example-bucket', '--country', 'testland',
                    '--campaign', 'campaign_a', '--sensor', 'landsat', '--method', 'minnbr',
                    '--prefix', 'root/', '--cutoff', '2026-11-15T02:59:59', '--output', output]
            with mock.patch('sys.stderr'), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(os.path.exists(output))

    def test_exact_cutoff_groups_and_manifest(self):
        objects = [blob(BASE + 'CHUNKS/' + NAME + '_0000000000-0000000000.tif'),
                   blob(BASE + 'CHUNKS/' + NAME + '_0000000000-0000000001.tif'),
                   blob(BASE + 'COG/' + NAME + '_cog.tif')]
        manifest = {'version': 1, 'items': [dict(country='testland', sensor='landsat',
                    period='yearly', mosaic='minnbr', period_key='2025', band='blue',
                    type='GCS', status='complete')]}
        result = run(objects, manifest=manifest)
        self.assertEqual(len(result['candidates']), 0)
        item = result['already_complete'][0]
        self.assertEqual(item['manifest_status'], 'complete')
        self.assertEqual(result['campaign_context'], ['campaign_a', 'campaign_b'])
        self.assertNotIn('campaigns', item)
        self.assertEqual(len(item['chunks']), 2)
        self.assertEqual(len(item['cogs']), 1)
        self.assertTrue(result['ready'])

    def test_unknown_time_and_bad_names_are_unresolved(self):
        result = run([blob(BASE + 'CHUNKS/' + NAME + '_x.tif', None),
                      blob('root/unknown.tif')])
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['unresolved']), 2)

    def test_partial_failure_and_duplicates(self):
        obj = blob(BASE + 'COG/' + NAME + '_cog.tif')
        result = run([obj, obj], fail=True)
        self.assertFalse(result['scan_complete'])
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['observed_groups'][0]['cogs']), 1)
        self.assertTrue(result['errors'])

    def test_timestamps_require_timezone(self):
        with self.assertRaises(ValueError):
            parse_timestamp('2026-11-15T02:59:59')
        self.assertEqual(parse_timestamp('2026-11-14T23:59:59-03:00'), CUTOFF)

    def test_invalid_context_and_manifest_are_rejected(self):
        for kwargs in ({'manifest': {'version': 2, 'items': []}},
                       {'manifest': {'version': 1, 'items': [None]}}):
            with self.assertRaises(ValueError):
                run([], **kwargs)
        with self.assertRaises(ValueError):
            inventory(Client([]), 'bucket', [''], 'testland', ['a'],
                      'landsat', 'minnbr', CUTOFF)

    def test_missing_generation_and_naive_creation_are_unresolved(self):
        name = BASE + 'COG/' + NAME + '_cog.tif'
        result = run([blob(name, CUTOFF, None),
                      blob(name, dt.datetime(2026, 1, 1), 2)])
        self.assertEqual(len(result['unresolved']), 2)
        self.assertEqual(result['candidates'], [])

    def test_overlapping_prefixes_deduplicate_and_conflicting_manifest(self):
        obj = blob(BASE + 'COG/' + NAME + '_cog.tif')
        entry = dict(country='testland', sensor='landsat', period='yearly',
                     mosaic='minnbr', period_key='2025', band='blue',
                     type='GCS', status='complete')
        result = inventory(Client([obj]), 'bucket', ['root/', BASE], 'testland',
                           ['a', 'a'], 'landsat', 'minnbr', CUTOFF,
                           manifest={'version': 1, 'items': [entry, entry]})
        self.assertFalse(result['candidates'])
        self.assertEqual(len(result['observed_groups'][0]['cogs']), 1)
        self.assertIsNone(result['observed_groups'][0]['manifest_status'])
        self.assertTrue(result['conflicts'])


if __name__ == '__main__':
    unittest.main()
