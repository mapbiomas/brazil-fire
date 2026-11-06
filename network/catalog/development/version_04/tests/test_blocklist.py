"""Offline tests for the GEE blocklist resolver."""

import os
import sys
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M_blocklist as mb


class ParserTests(unittest.TestCase):
    def test_root_export(self):
        source = "exports.landsat = function(){ return ['A', 'B']; };"
        result = mb.extract_blocklists(
            lambda asset: source,
            root_require='users/x/y:file',
            root_sensor='landsat',
        )
        self.assertEqual(result['landsat'], ['A', 'B'])

    def test_child_require(self):
        sources = {
            'users/x/y:root': "var x = require('users/x/y:child').hls_l30();",
            'users/x/y:child': "exports.hls_l30 = function(){ return ['H1']; };",
        }
        result = mb.extract_blocklists(
            lambda asset: sources[asset],
            root_require='users/x/y:root',
        )
        self.assertEqual(result, {'hls_l30': ['H1']})

    def test_comments_do_not_create_entries(self):
        source = """
        // 'ignored'
        exports.sentinel = function(){
          return ['20190829T142759_20190829T142756_T20MNT'];
        };
        """
        result = mb.extract_blocklists(
            lambda asset: source,
            root_require='users/x/y:file',
            root_sensor='sentinel',
        )
        self.assertEqual(result['sentinel'], ['20190829T142759_20190829T142756_T20MNT'])

    def test_unclosed_list_raises(self):
        source = "exports.landsat = function(){ return ['A'; };"
        with self.assertRaises(mb.BlocklistSourceError):
            mb.extract_blocklists(
                lambda asset: source,
                root_require='users/x/y:file',
                root_sensor='landsat',
            )


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.sources = mb.BLOCKLIST_SOURCES
        self.aliases = mb.BLOCKLIST_ALIASES
        self.export_names = mb.BLOCKLIST_EXPORT_NAMES
        mb._BLOCKLISTS = None
        mb._BLOCKLIST_META = None

    def tearDown(self):
        mb.BLOCKLIST_SOURCES = self.sources
        mb.BLOCKLIST_ALIASES = self.aliases
        mb.BLOCKLIST_EXPORT_NAMES = self.export_names
        mb._BLOCKLISTS = None
        mb._BLOCKLIST_META = None

    def test_missing_source_does_not_raise(self):
        mb.BLOCKLIST_SOURCES = {'landsat': ('users/x/y:missing',)}

        def reader(asset):
            raise mb.BlocklistSourceError('missing')

        result = mb.refresh_blocklists(['landsat'], read_source=reader)
        self.assertEqual(result['landsat'], [])

        report = mb.blocklist_report(['landsat'])
        self.assertEqual(report['landsat']['count'], 0)
        self.assertEqual(report['landsat']['sources'][0]['status'], 'error')

    def test_partial_sources_keep_the_valid_one(self):
        mb.BLOCKLIST_SOURCES = {
            'landsat': ('users/x/y:missing', 'users/x/y:ok'),
        }

        def reader(asset):
            if asset.endswith('missing'):
                raise mb.BlocklistSourceError('missing')
            return "exports.landsat = function(){ return ['A']; };"

        result = mb.refresh_blocklists(['landsat'], read_source=reader)
        self.assertEqual(result['landsat'], ['A'])

        report = mb.blocklist_report(['landsat'])
        statuses = [source['status'] for source in report['landsat']['sources']]
        self.assertEqual(statuses, ['error', 'ok'])

    def test_alias_union(self):
        mb.BLOCKLIST_SOURCES = {
            'hls': ('users/x/y:hls',),
            'hls_l30': ('users/x/y:l30',),
        }
        mb.BLOCKLIST_ALIASES = {'hls_l30': ('hls_l30', 'hls')}
        sources = {
            'users/x/y:l30': "exports.hls_l30 = function(){ return ['L1']; };",
            'users/x/y:hls': "exports.hls = function(){ return ['L1', 'L2']; };",
        }

        mb.refresh_blocklists(['hls_l30'], read_source=lambda asset: sources[asset])
        self.assertEqual(mb.get_blocklist('hls_l30'), ['L1', 'L2'])

    def test_sentinel_export_name(self):
        mb.BLOCKLIST_SOURCES = {'sentinel2': ('users/x/y:module',)}
        mb.BLOCKLIST_EXPORT_NAMES = {'sentinel2': 'sentinel'}
        source = "exports.sentinel = function(){ return ['S1']; };"

        mb.refresh_blocklists(['sentinel2'], read_source=lambda asset: source)
        self.assertEqual(mb.get_blocklist('sentinel2'), ['S1'])

    def test_sensor_without_sources_skips_fetch(self):
        mb.BLOCKLIST_SOURCES = {'hls': (), 'hls_l30': ()}
        mb.BLOCKLIST_ALIASES = {'hls_l30': ('hls_l30', 'hls')}
        self.assertEqual(mb.get_blocklist('hls_l30'), [])


class GitEnvironmentTests(unittest.TestCase):
    def test_non_interactive_and_header_reset(self):
        env = mb._git_environment(token='')
        self.assertEqual(env['GIT_TERMINAL_PROMPT'], '0')

        count = int(env['GIT_CONFIG_COUNT'])
        keys = [env[f'GIT_CONFIG_KEY_{index}'] for index in range(count)]
        values = [env[f'GIT_CONFIG_VALUE_{index}'] for index in range(count)]
        header = 'http.https://earthengine.googlesource.com/.extraheader'
        self.assertIn(header, keys)
        self.assertEqual(values[keys.index(header)], '')
        self.assertNotIn('Authorization: Bearer ', ' '.join(values))

    def test_fresh_token_is_added_after_the_reset(self):
        env = mb._git_environment(token='abc123')

        count = int(env['GIT_CONFIG_COUNT'])
        keys = [env[f'GIT_CONFIG_KEY_{index}'] for index in range(count)]
        values = [env[f'GIT_CONFIG_VALUE_{index}'] for index in range(count)]
        header_values = [
            value for key, value in zip(keys, values)
            if key == 'http.https://earthengine.googlesource.com/.extraheader'
        ]
        self.assertEqual(header_values, ['', 'Authorization: Bearer abc123'])


if __name__ == '__main__':
    unittest.main()
