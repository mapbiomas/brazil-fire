import unittest

from gif_factory.territory_search import search, search_municipalities, territory


class TerritorySearchTests(unittest.TestCase):
    def test_search_uses_unique_id(self):
        class FC:
            def filter(self, condition): return self
            def limit(self, limit): return self
            def getInfo(self):
                return {"features": [{"id": "42", "properties": {"NM_MUN": "Petrolina"}}]}
        class EE:
            FeatureCollection = staticmethod(lambda source: FC())
            class Filter:
                stringContains = staticmethod(lambda field, value: (field, value))
        result = search(EE, "asset", "NM_MUN", "system:index", "Petro")
        self.assertEqual(result[0]["value"], "42")

    def test_combination_is_order_independent(self):
        records = [{"name": "Juazeiro", "value": "a", "asset": "asset", "field": "id"},
                   {"name": "Petrolina", "value": "b", "asset": "asset", "field": "id"}]
        first = territory(records)
        self.assertEqual(first, territory(list(reversed(records))))
        self.assertEqual(first["filter"]["values"], ["a", "b"])

    def test_municipality_shows_uf_and_keeps_code(self):
        class FC:
            def filter(self, condition): return self
            def select(self, fields, retainGeometry): return self
            def limit(self, limit): return self
            def getInfo(self):
                return {"features": [{"properties": {"GAUL2_NAME": "Petrolina", "GAUL1_NAME": "Pernambuco", "GAUL2_CODE": 123}}]}
        class EE:
            FeatureCollection = staticmethod(lambda source: FC())
            class Filter:
                eq = staticmethod(lambda field, value: (field, value))
                stringContains = staticmethod(lambda field, value: (field, value))
        result = search_municipalities(EE, "Petro")
        self.assertEqual(result[0]["name"], "Petrolina (PE)")
        self.assertEqual(result[0]["value"], 123)

    def test_combination_of_different_sources_is_rejected(self):
        records = [{"name": "A", "value": "a", "asset": "x", "field": "id"},
                   {"name": "B", "value": "b", "asset": "y", "field": "id"}]
        with self.assertRaises(ValueError):
            territory(records)
