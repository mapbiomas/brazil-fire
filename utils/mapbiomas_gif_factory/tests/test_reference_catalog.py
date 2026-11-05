import unittest

from gif_factory.reference_catalog import (experimental_layer, layer_for_record,
                                            build_reference_layers)
from gif_factory.legends import style_for


class ReferenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.record = {"initiative": "colombia", "theme": "fire", "collection": "c1",
                       "product": "annual_burned", "asset": "projects/mapbiomas-public/assets/test"}

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

    def test_fire_binary_uses_official_red(self):
        style = style_for({**self.record, "product": "annual_burned"}, "pt")
        self.assertEqual(style["visualization"]["palette"], ["ffffff", "d4271e"])

    def test_fire_products_have_palettes(self):
        for product in ["annual_burned", "accumulated_burned", "monthly_burned", "fire_frequency",
                        "severity_class", "annual_burned_scar_size_range", "fire_return_interval",
                        "year_last_fire", "time_after_fire", "annual_burned_coverage"]:
            style = style_for({**self.record, "product": product}, "pt")
            self.assertIsNotNone(style, product)
            self.assertTrue(len(style["visualization"]["palette"]) >= 2, product)

    def test_style_follows_country_language(self):
        self.assertEqual(style_for({**self.record, "initiative": "peru"}, "es")["legend"][1]["label"], "Ocurrencia de fuego")
        self.assertEqual(style_for({**self.record, "initiative": "indonesia"}, "en")["legend"][1]["label"], "Burned")

    def test_other_themes_have_palettes(self):
        for theme, product in [("lulc", "integration"), ("lulc", "coverage"),
                               ("water", "annual_water"), ("water", "water_frequency"),
                               ("soil", "organic_carbon"), ("soil", "granulometry_clay"),
                               ("soil", "textural_class"), ("soil", "stoniness"),
                               ("atmosphere", "air_temperature_annual"),
                               ("atmosphere", "precipitation_annual"),
                               ("urban", "annual_urbanization"),
                               ("climate_risk", "landslide_susceptibility")]:
            style = style_for({"theme": theme, "product": product, "initiative": "brasil"}, "pt")
            self.assertIsNotNone(style, f"{theme}/{product}")
            self.assertTrue(len(style["visualization"]["palette"]) >= 2, f"{theme}/{product}")

    def test_layer_for_record_marks_unknown_style_disabled(self):
        enabled = layer_for_record(self.record)
        self.assertTrue(enabled.get("enabled", True))
        self.assertIn("visualization", enabled)
        unknown = layer_for_record({**self.record, "product": "unmapped_product"})
        self.assertFalse(unknown["enabled"])
        self.assertIn("Paleta/legenda", unknown["reason"])
        missing = layer_for_record({**self.record, "asset": None})
        self.assertFalse(missing["enabled"])

    def test_layer_for_record_uses_initiative_language(self):
        peru = layer_for_record({**self.record, "initiative": "peru"})
        self.assertEqual(peru["legend"][0]["label"], "Ausencia de fuego")
        indonesia = layer_for_record({**self.record, "initiative": "indonesia"})
        self.assertEqual(indonesia["legend"][1]["label"], "Burned")

    def test_build_reference_layers_preserves_count(self):
        records = [dict(self.record), dict(self.record, product="monthly_burned")]
        layers = build_reference_layers(records)
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0]["name"], "annual_burned")
