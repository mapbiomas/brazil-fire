import importlib.util
import json
import unittest
from unittest.mock import patch
from pathlib import Path

from gif_factory.legends import style_for

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = json.loads((ROOT / 'src/gif_factory/reference_inventory.json').read_text(encoding='utf-8'))['records']


def inventory_record(product):
    return next(r.copy() for r in INVENTORY if r['initiative'] == 'brasil'
                and r['theme'] == 'fire' and r['collection'] == 'collection_05_1'
                and r['product'] == product)


class VeraFireLegendsTests(unittest.TestCase):
    def test_real_inventory_months_and_collection_alias(self):
        record = inventory_record('monthly_burned')
        style = style_for(record, 'es')
        self.assertEqual((style['visualization']['min'], style['visualization']['max']), (1, 12))
        self.assertEqual(style['legend'][0], {'label': 'Enero', 'color': 'cc00ff'})
        self.assertFalse(style['visualization']['prefix_labels'])
        self.assertEqual(style_for({**record, 'collection': '5.1'}, 'es'), style)

    def test_severity_classes_and_localization(self):
        style = style_for(inventory_record('severity_class'), 'en')
        self.assertEqual((style['visualization']['min'], style['visualization']['max']), (1, 6))
        self.assertEqual(style['visualization']['palette'][-1], '99151d')
        self.assertEqual(style['legend'][-1]['label'], 'Extreme')

    def test_nonrepresentable_classes_return_none(self):
        for product in ('annual_burned', 'accumulated_burned', 'annual_burned_coverage'):
            with self.subTest(product=product):
                self.assertIsNone(style_for(inventory_record(product)))

    def test_absent_or_mismatched_key_never_falls_back(self):
        record = inventory_record('monthly_burned')
        for field, value in (('collection', '5'), ('collection', 'collection_05'),
                             ('asset', 'other'), ('initiative', 'peru'),
                             ('product', 'monthly burned'), ('asset', None),
                             ('collection', None)):
            with self.subTest(field=field, value=value):
                self.assertIsNone(style_for({**record, field: value}))
        self.assertIsNone(style_for(inventory_record('time_after_fire')))
        self.assertIsNone(style_for({'theme': 'fire', 'product': 'annual_burned'}))

    def test_snapshot_reproducible_and_classes_consecutive(self):
        spec = importlib.util.spec_from_file_location('build_vera', ROOT / 'build_vera_fire_legends.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        snapshot = json.loads((ROOT / 'src/gif_factory/vera_fire_legends.json').read_text(encoding='utf-8'))
        self.assertEqual(module.build(), snapshot)
        for entry in snapshot['entries']:
            self.assertTrue(all(entry['key']))
            if entry['representable']:
                values = [c['value'] for c in entry['classes']]
                self.assertGreaterEqual(len(values), 2)
                self.assertEqual(values, list(range(values[0], values[-1] + 1)))

    def test_peru_inventory_numeric_collection_aliases_and_distinct_versions(self):
        record = next(r.copy() for r in INVENTORY if r['initiative'] == 'peru'
                      and r['theme'] == 'fire' and r['collection'] == 'collection_2'
                      and r['product'] == 'monthly_burned')
        # O registro de Vera não tem asset para Peru: não é habilitado em produção.
        self.assertIsNone(style_for(record))
        fake = {'representable': True, 'classes': [
            {'value': 1, 'color': '000000', 'labels': {'pt': '1'}},
            {'value': 2, 'color': 'ffffff', 'labels': {'pt': '2'}},
        ]}
        key = ('peru', '2', record['product'], record['asset'])
        with patch.dict('gif_factory.legends._VERA', {key: fake}):
            for collection in ('2', '02', 'collection_2', 'collection_02'):
                with self.subTest(collection=collection):
                    self.assertIsNotNone(style_for({**record, 'collection': collection}))
            for collection in ('3', '2.1', 'collection_02_1', 'monitor',
                               'beta', 'collection_02_beta', 'collection_2monitor'):
                with self.subTest(collection=collection):
                    self.assertIsNone(style_for({**record, 'collection': collection}))

    def test_multicomponent_versions_are_distinct(self):
        record = inventory_record('monthly_burned')
        for collection in ('5.1', '05.01', 'collection_5_1', 'collection_05_1'):
            with self.subTest(collection=collection):
                self.assertIsNotNone(style_for({**record, 'collection': collection}))
        for collection in ('5', '5.10', '5.1.0', 'collection_05_1_beta', 'monitor'):
            with self.subTest(collection=collection):
                self.assertIsNone(style_for({**record, 'collection': collection}))


if __name__ == '__main__':
    unittest.main()
