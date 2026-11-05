import unittest

from gif_factory.reference_catalog import (experimental_layer, layer_for_record,
                                            suggest_style, build_reference_layers)


class ReferenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.record = {"initiative": "colombia", "theme": "fire", "collection": "c1",
                       "product": "annual", "asset": "projects/mapbiomas-public/assets/test"}

    def test_explicit_style_is_preserved(self):
        layer = experimental_layer(self.record, "ffffff,#800000", "Sem fogo,Fogo", 0, 1)
        self.assertEqual(layer["visualization"]["palette"], ["ffffff", "800000"])
        self.assertEqual(layer["legend"][1]["label"], "Fogo")
        self.assertEqual(layer["initiative"], "Colômbia")

    def test_invalid_or_unavailable_configuration_is_rejected(self):
        for palette, labels, minimum, maximum in [("fff,000", "a,b", 0, 1),
                                                   ("ffffff,000000", "a", 0, 1),
                                                   ("ffffff,000000", "a,b", 1, 1)]:
            with self.assertRaises(ValueError):
                experimental_layer(self.record, palette, labels, minimum, maximum)
        with self.assertRaises(ValueError):
            experimental_layer({**self.record, "asset": None}, "ffffff,000000", "a,b", 0, 1)

    def test_suggest_style_for_known_fire_products(self):
        self.assertEqual(suggest_style("annual_burned")["visualization"]["palette"], ["fdfdfd", "800000"])
        self.assertEqual(len(suggest_style("monthly_burned")["visualization"]["palette"]), 12)
        self.assertEqual(len(suggest_style("severity_class")["visualization"]["palette"]), 6)
        self.assertIsNone(suggest_style("lulc_coverage"))

    def test_layer_for_record_marks_unknown_style_disabled(self):
        enabled = layer_for_record({**self.record, "product": "annual_burned"})
        self.assertTrue(enabled.get("enabled", True))
        self.assertIn("visualization", enabled)
        unknown = layer_for_record({**self.record, "product": "lulc_coverage"})
        self.assertFalse(unknown["enabled"])
        self.assertIn("Sem paleta", unknown["reason"])
        missing = layer_for_record({**self.record, "asset": None})
        self.assertFalse(missing["enabled"])

    def test_build_reference_layers_preserves_count(self):
        records = [dict(self.record, product="annual_burned"),
                   dict(self.record, product="lulc_coverage")]
        layers = build_reference_layers(records)
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0]["name"], "annual_burned")
