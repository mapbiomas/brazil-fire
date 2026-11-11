import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from gif_factory.presets import LAYERS
from gif_factory.reference_catalog import build_reference_layers, layer_for_record


class CatalogDeduplicationTests(unittest.TestCase):
    def test_inventory_shared_assets_are_explicitly_curated(self):
        inventory = Path(__file__).parents[1] / "src/gif_factory/reference_inventory.json"
        records = json.loads(inventory.read_text(encoding="utf-8"))["records"]
        shared = [(p, r) for p in LAYERS for r in records
                  if r.get("asset") and p.get("asset") == r["asset"]]
        self.assertEqual(len(shared), 18)
        removed = {p["id"] for p, r in shared if not build_reference_layers([r], presets=[p])}
        self.assertEqual(removed, {"annual_burned", "annual_burned_col4", "peru_fire2_annual", "fire_annual_br51"})
        self.assertTrue(all(p.get("dedup_style_reason") for p, _ in shared if p["id"] in removed))

    def test_explicit_style_preference_never_overrides_operational_differences(self):
        record = {"initiative": "brasil", "theme": "fire", "collection": "collection_05",
                  "product": "annual_burned", "asset": LAYERS[0]["asset"]}
        style = {"visualization": {"min": 0, "max": 1, "palette": ["ffffff", "ff0000"]}, "legend": []}
        with patch("gif_factory.reference_catalog.style_for", return_value=style):
            preset = {**layer_for_record(record), "product": record["product"],
                      "visualization": {"min": 0, "max": 1, "palette": ["ffffff", "000000"]},
                      "dedup_style_reason": "Preferência visual documentada."}
            self.assertEqual(build_reference_layers([record], presets=[preset]), [])
            for change in ({"bands_slice": [0, 2]}, {"processor": "other"},
                           {"band": "other"}, {"selection": ["b1"]}, {"enabled": False},
                           {"product": "monthly_burned"}, {"asset": "other"},
                           {"dedup_style_reason": ""}):
                with self.subTest(change=change):
                    modified = copy.deepcopy(preset)
                    modified.update(change)
                    self.assertEqual(len(build_reference_layers([record], presets=[modified])), 1)
