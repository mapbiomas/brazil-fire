import unittest
from unittest.mock import patch

from gif_factory.presets import LAYERS
from gif_factory.reference_catalog import build_reference_layers


class DisabledDuplicateTests(unittest.TestCase):
    def test_disabled_reference_with_same_product_and_enabled_preset_is_hidden(self):
        preset = next(item for item in LAYERS if item["id"] == "fire_annual_br51")
        record = {"initiative": "brasil", "theme": "fire", "collection": "collection_05_1",
                  "product": "annual_burned", "asset": preset["asset"]}
        with patch("gif_factory.reference_catalog.style_for", return_value=None):
            self.assertEqual(build_reference_layers([record], presets=[preset]), [])
            self.assertEqual(len(build_reference_layers([record], presets=[{**preset, "enabled": False}])), 1)
            self.assertEqual(len(build_reference_layers([record], presets=[{**preset, "bands_slice": [0, 1]}])), 1)
