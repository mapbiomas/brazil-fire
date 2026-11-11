import json
import unittest
from pathlib import Path

from gif_factory.reference_catalog import (experimental_layer, layer_for_record,
                                            build_reference_layers, normalize_collection)
from gif_factory.legends import style_for

INVENTORY = json.loads((Path(__file__).resolve().parents[1] / "src/gif_factory/reference_inventory.json").read_text(encoding="utf-8"))["records"]
MONTHLY = next(r for r in INVENTORY if r["initiative"] == "brasil" and r["theme"] == "fire" and r["collection"] == "collection_05_1" and r["product"] == "monthly_burned")


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

    def test_unknown_fire_identity_has_no_fallback(self):
        for product in ["annual_burned", "accumulated_burned", "monthly_burned", "fire_frequency",
                        "severity_class", "annual_burned_scar_size_range", "fire_return_interval",
                        "year_last_fire", "time_after_fire", "annual_burned_coverage"]:
            self.assertIsNone(style_for({**self.record, "product": product}, "pt"), product)

    def test_style_follows_country_language(self):
        self.assertEqual(style_for(MONTHLY, "es")["legend"][0]["label"], "Enero")
        self.assertEqual(style_for(MONTHLY, "en")["legend"][0]["label"], "January")

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
        enabled = layer_for_record(MONTHLY)
        self.assertTrue(enabled.get("enabled", True))
        self.assertIn("visualization", enabled)
        unknown = layer_for_record({**self.record, "product": "unmapped_product"})
        self.assertFalse(unknown["enabled"])
        self.assertIn("Paleta/legenda", unknown["reason"])
        missing = layer_for_record({**self.record, "asset": None})
        self.assertFalse(missing["enabled"])

    def test_real_non_fire_inventory_styles_are_not_inferred(self):
        products = ("soil_organic_carbon_0_30cm_kg_m2", "soil_organic_carbon_0_30cm_t_ha",
                    "soil_carbon", "soil_clay_fraction")
        for product in products:
            records = [r for r in INVENTORY if r["initiative"] == "brasil" and r["product"] == product]
            self.assertTrue(records, product)
            for record in records:
                with self.subTest(product=product, asset=record["asset"]):
                    layer = layer_for_record(record)
                    self.assertFalse(layer["enabled"])
                    self.assertIn("identidade", layer["reason"])
                    self.assertNotIn("visualization", layer)
                    self.assertNotIn("legend", layer)

    def test_explicit_non_fire_experimental_style_remains_available(self):
        record = next(r for r in INVENTORY if r["product"] == "soil_organic_carbon_0_30cm_kg_m2")
        layer = experimental_layer(record, "ffffff,000000", "baixo,alto", 0, 10)
        self.assertTrue(layer.get("enabled", True))
        self.assertEqual(layer["visualization"]["max"], 10)

    def test_layer_for_record_uses_initiative_language(self):
        layer = layer_for_record(MONTHLY)
        self.assertEqual(layer["legend"][0]["label"], "Janeiro")
        self.assertFalse(layer_for_record({**self.record, "initiative": "peru"})["enabled"])

    def test_build_reference_layers_preserves_count(self):
        records = [dict(self.record), dict(self.record, product="monthly_burned")]
        layers = build_reference_layers(records)
        self.assertEqual(len(layers), 2)
        self.assertEqual(layers[0]["name"], "annual_burned")

    def test_legacy_asset_exclusion_does_not_hide_records(self):
        records = [self.record, dict(self.record, product="monthly_burned")]
        self.assertEqual(len(build_reference_layers(records, exclude_assets={self.record["asset"]})), 2)

    def test_presets_only_exclude_equivalent_enabled_layers(self):
        record = MONTHLY
        reference = layer_for_record(record)
        equivalent = {**reference, "id": "preset", "name": "Outro nome",
                      "product": "monthly_burned",
                      "theme": "Fogo", "collection": "Coleção 5.1"}
        self.assertEqual(build_reference_layers([record], presets=[equivalent]), [])
        for change in ({"enabled": False}, {"reclass": [(0, 1, 2)]},
                       {"bands_slice": [0, 10]}, {"collection": "beta"},
                       {"initiative": "Peru"}, {"theme": "soil"},
                       {"product": "annual_burned"},
                       {"visualization": {"min": 0, "max": 2}}):
            with self.subTest(change=change):
                self.assertEqual(len(build_reference_layers([record], presets=[{**equivalent, **change}])), 1)

    def test_disabled_reference_is_not_hidden(self):
        record = {**self.record, "product": "unknown"}
        preset = {**layer_for_record(record), "enabled": True}
        self.assertEqual(len(build_reference_layers([record], presets=[preset])), 1)

    def test_collection_normalization_keeps_qualifiers_distinct(self):
        for value in ("c1", "collection_01", "Colección 1", "01"):
            with self.subTest(value=value):
                self.assertEqual(normalize_collection(value), "1")
        for value in ("beta", "monitor", "collection_01_beta", "Coleção 1 monitor"):
            with self.subTest(value=value):
                self.assertEqual(normalize_collection(value), value)
