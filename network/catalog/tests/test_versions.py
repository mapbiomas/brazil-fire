"""Offline validation of every catalog version layout and entry-point path."""

import glob
import os
import unittest

CATALOG = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
NETWORK = os.path.dirname(CATALOG)


def _read(path):
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def _notebooks(base):
    return sorted(
        glob.glob(os.path.join(base, 'notebooks', 'monitor_01_*.ipynb'))
        + glob.glob(os.path.join(base, 'notebooks', 'collection_*.ipynb'))
    )


def _campaigns(base):
    names = []
    for path in glob.glob(os.path.join(base, 'campaigns', '*.json')):
        name = os.path.basename(path)
        if name != 'index.json':
            names.append(name)
    return set(names)


class RootLegacyMirrorTests(unittest.TestCase):
    def test_version_01_is_an_unchanged_mirror(self):
        base = os.path.join(NETWORK, 'version_01', 'classification_algorithms')
        a01 = os.path.join(base, 'A_0_1_basic_authentication_and_parameterization.py')
        notebook = os.path.join(base, 'mapbiomas_fire_classification_for_collab.ipynb')
        self.assertTrue(os.path.exists(a01))
        self.assertTrue(os.path.exists(notebook))
        # The notebook is the entry point; the legacy A_0_1 kept its historical
        # network/classification_algorithms path in the original mirror.
        self.assertIn('network/version_01/classification_algorithms', _read(notebook))

    def test_version_02_is_an_unchanged_mirror(self):
        base = os.path.join(NETWORK, 'version_02', 'classification_algorithms')
        a01 = os.path.join(base, 'A_0_1_basic_authentication_and_parameterization.py')
        notebook = os.path.join(base, 'mapbiomas_fire_classification_for_collab.ipynb')
        self.assertTrue(os.path.exists(a01))
        self.assertTrue(os.path.exists(notebook))
        self.assertIn('network/version_02/classification_algorithms', _read(a01))
        self.assertIn('network/version_02/classification_algorithms', _read(notebook))


class CollectionVersionPathsTests(unittest.TestCase):
    def test_version_00_points_to_catalog(self):
        base = os.path.join(CATALOG, 'development', 'version_00', 'classification_algorithms')
        expected = 'network/catalog/development/version_00/classification_algorithms'
        a01 = os.path.join(base, 'A_0_1_basic_authentication_and_parameterization.py')
        notebook = os.path.join(base, 'mapbiomas_fire_classification_for_collab.ipynb')
        self.assertIn(expected, _read(a01))
        self.assertIn(expected, _read(notebook))

    def test_version_01_points_to_catalog(self):
        base = os.path.join(CATALOG, 'development', 'version_01', 'classification_algorithms')
        expected = 'network/catalog/development/version_01/classification_algorithms'
        a01 = os.path.join(base, 'A_0_1_basic_authentication_and_parameterization.py')
        notebook = os.path.join(base, 'mapbiomas_fire_classification_for_collab.ipynb')
        self.assertIn(expected, _read(a01))
        self.assertIn(expected, _read(notebook))


class MonitorVersionPathsTests(unittest.TestCase):
    EXPECTED = {
        'development/version_03': 'network/catalog/development/version_03/src/core',
        'development/version_04': 'network/catalog/development/version_04/src/core',
        'production': 'network/catalog/production/src/core',
    }

    def test_each_version_has_14_notebooks_with_the_right_repo_path(self):
        for rel, expected in self.EXPECTED.items():
            base = os.path.join(CATALOG, rel)
            notebooks = _notebooks(base)
            self.assertEqual(len(notebooks), 14, f'{rel}: expected 14 notebooks')
            for notebook in notebooks:
                self.assertIn(expected, _read(notebook), notebook)

    def test_colab_link_lists_exist(self):
        for rel in self.EXPECTED:
            base = os.path.join(CATALOG, rel)
            self.assertTrue(os.path.exists(os.path.join(base, 'COLAB_LINKS.csv')), rel)
            self.assertTrue(os.path.exists(os.path.join(base, 'COLAB_LINKS.txt')), rel)

    def test_version_03_and_04_campaigns_match(self):
        v03 = _campaigns(os.path.join(CATALOG, 'development', 'version_03'))
        v04 = _campaigns(os.path.join(CATALOG, 'development', 'version_04'))
        self.assertEqual(v03, v04)

    def test_production_matches_version_03(self):
        v03 = _campaigns(os.path.join(CATALOG, 'development', 'version_03'))
        production = _campaigns(os.path.join(CATALOG, 'production'))
        self.assertEqual(v03, production)


if __name__ == '__main__':
    unittest.main()
