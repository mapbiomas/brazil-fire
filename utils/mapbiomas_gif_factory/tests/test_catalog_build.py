"""Gerador do catálogo (sync/catalog_build.py): estável, fiel às legendas oficiais e sem tocar no curado."""
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catalog_build", ROOT / "sync" / "catalog_build.py")
catalog_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog_build)

TREE = {"source": "x", "fetched": "2026-11-19", "legends": [
    {"key": "cobertura", "items": [
        {"pixelValue": 1, "level": 1, "color": "#1F8D49", "name": {"pt-BR": "Floresta"}, "children": [
            {"pixelValue": 3, "level": 2, "color": "#1f8d49", "name": {"pt-BR": "Formação Florestal"}},
            {"pixelValue": 4, "level": 2, "color": "#7dc975", "name": {"pt-BR": "Formação Savânica"}}]},
        {"pixelValue": 14, "level": 1, "color": "#ffefc3", "name": {"pt-BR": "Agropecuária"}, "children": [
            {"pixelValue": 18, "level": 2, "color": "#e974ed", "name": {"pt-BR": "Agricultura"}, "children": [
                {"pixelValue": 39, "level": 3, "color": "#f5b3c8", "name": {"pt-BR": "Soja"}}]},
            {"pixelValue": 15, "level": 2, "color": "#edde8e", "name": {"pt-BR": "Pastagem"}}]},
        {"pixelValue": 27, "level": 1, "color": "#ffffff", "name": {"pt-BR": "Não observado"}}]},
    {"key": "natural", "items": [
        {"pixelValue": 100, "level": 1, "color": "#139912", "name": {"pt-BR": "Natural"},
         "children": [{"pixelValue": 3, "level": 2}, {"pixelValue": 4, "level": 2}]},
        {"pixelValue": 200, "level": 1, "color": "#FFD966", "name": {"pt-BR": "Antrópico"},
         "children": [{"pixelValue": 39, "level": 2}, {"pixelValue": 15, "level": 2}]}]}]}
INVENTORY = [{"id": f"projects/mapbiomas-public/assets/xx/lulc/collection{c}/mapbiomas_xx_collection{c}_{k}_v{v}",
              "type": "IMAGE"} for c, k, v in [(2, "integration", 1), (3, "integration", 2), (3, "coverage", 1)]]


class CatalogBuildTests(unittest.TestCase):
    def build(self):
        rule = {"gee": "xx", "code": "xx", "name": "Xis", "lang": "pt-BR"}
        return catalog_build.build_initiative("xx", rule, TREE, INVENTORY)

    def test_picks_the_latest_collection_preferring_coverage(self):
        collection, _, problem = self.build()
        self.assertIsNone(problem)
        self.assertEqual(collection["collection"], "3")
        self.assertTrue(collection["products"]["lulc_nivel1"]["asset"].endswith("collection3_coverage_v1"))

    def test_each_level_maps_leaves_to_their_official_ancestor(self):
        collection, styles, _ = self.build()
        products = collection["products"]
        self.assertEqual(set(products), {"lulc_nivel0", "lulc_nivel1", "lulc_nivel2", "lulc_nivel3"})
        level1 = dict(zip(products["lulc_nivel1"]["remap"]["from"], products["lulc_nivel1"]["remap"]["to"]))
        self.assertEqual(level1, {3: 1, 4: 1, 39: 14, 15: 14, 27: 27})
        level2 = dict(zip(products["lulc_nivel2"]["remap"]["from"], products["lulc_nivel2"]["remap"]["to"]))
        self.assertEqual(level2[39], 18)
        self.assertNotIn("remap", products["lulc_nivel3"])  # o último nível é o dado bruto
        level0 = dict(zip(products["lulc_nivel0"]["remap"]["from"], products["lulc_nivel0"]["remap"]["to"]))
        self.assertEqual(level0, {3: 100, 4: 100, 39: 200, 15: 200})

    def test_styles_carry_only_the_level_classes_with_official_colors(self):
        _, styles, _ = self.build()
        style = styles["gen_xx_lulc_nivel1"]
        named = {i: (style["palette"][i], label) for i, label in enumerate(style["discrete_labels"]) if label}
        self.assertEqual(named, {1: ("1f8d49", "Floresta"), 14: ("ffefc3", "Agropecuária"),
                                 27: ("ffffff", "Não observado")})

    def test_committed_generated_files_are_up_to_date(self):
        result = subprocess.run([sys.executable, str(ROOT / "sync" / "catalog_build.py"), "--check"],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_curated_land_cover_is_never_generated(self):
        rules = yaml.safe_load((ROOT / "sync" / "generator.yaml").read_text(encoding="utf-8"))
        for project, rule in rules["initiatives"].items():
            generated = ROOT / "src/gif_factory/config/initiatives" / rule["code"] / "collections/generated_lulc.yaml"
            if rule.get("skip_lulc"):
                self.assertFalse(generated.exists(), project)

    def test_generated_products_are_beta_until_reviewed(self):
        from gif_factory.catalog import catalog_layers
        generated = [layer for layer in catalog_layers(("validado", "beta", "pendente"))
                     if layer["dataset"].startswith("gen_")]
        self.assertGreaterEqual(len(generated), 40)
        self.assertTrue(all(layer["status"] == "beta" for layer in generated))
        self.assertTrue(all(layer["legend"] and all(item["label"] for item in layer["legend"]) for layer in generated))


if __name__ == "__main__":
    unittest.main()
