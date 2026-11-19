"""Solo (Coleção 3, por profundidade) e Atmosfera (Coleção 1) no catálogo, sem Earth Engine."""
import unittest
from unittest.mock import patch

from gif_factory.catalog import catalog_layers, config_loader
from gif_factory.engine import CatalogConfig, special_catalog_image
from gif_factory.temporal import resolve_bands, sequence_labels


def _products(dataset):
    return {layer["product"]: layer for layer in catalog_layers(("validado", "beta", "pendente"))
            if layer["dataset"] == dataset}


class SoilAtmosphereCatalogTests(unittest.TestCase):
    def test_soil_col3_is_validated_and_uses_official_styles(self):
        soil = _products("brasil_soil_col3")
        self.assertEqual(set(soil), {"carbon", "clay_fraction", "sand_fraction", "silt_fraction",
                                     "textural_group", "textural_subgroup", "textural_class"})
        self.assertTrue(all(layer["status"] == "validado" for layer in soil.values()))
        self.assertEqual(len(soil["textural_class"]["legend"]), 15)
        self.assertEqual(soil["textural_class"]["legend"][2], {"label": "Muito argilosa", "color": "a83800"})

    def test_depth_products_are_marked_and_labelled_in_cm(self):
        product = config_loader().datasets["brasil_soil_col3"]["products"]["textural_class"]
        self.assertEqual(product["axis"], "depth")
        bands = ["depth_060_100", "depth_000_010", "depth_000_020", "depth_020_040"]
        ordered = resolve_bands(bands, axis="depth")
        self.assertEqual(ordered, ["depth_000_010", "depth_000_020", "depth_020_040", "depth_060_100"])
        self.assertEqual(sequence_labels(ordered, axis="depth"), ["0–10 cm", "0–20 cm", "20–40 cm", "60–100 cm"])

    def test_atmosphere_is_beta_with_a_reason(self):
        atmosphere = _products("brasil_atmosphere_col1")
        self.assertEqual(len(atmosphere), 11)
        self.assertTrue(all(layer["status"] == "beta" and layer["reason"] for layer in atmosphere.values()))
        self.assertEqual(atmosphere["water_availability_annual"]["name"], "Disponibilidade hídrica")

    def test_temperature_variants_pick_one_image_of_the_collection(self):
        datasets = config_loader().datasets["brasil_atmosphere_col1"]["products"]
        filters = {key: datasets[key]["collection_filter"]["value"]
                   for key in ("air_temperature_max_annual", "air_temperature_mean_annual", "air_temperature_min_annual")}
        self.assertEqual(filters, {"air_temperature_max_annual": "maximum", "air_temperature_mean_annual": "mean",
                                   "air_temperature_min_annual": "minimum"})

    def test_rain_free_days_leaves_out_incomplete_2025(self):
        product = config_loader().datasets["brasil_atmosphere_col1"]["products"]["rain_free_days_annual"]
        self.assertEqual(product["bands_slice"], [0, 27])
        self.assertEqual(product["temporal_range"], [1998, 2024])

    def test_regular_products_keep_the_pipeline_loader(self):
        layer = next(l for l in catalog_layers() if l["id"] == "brasil_fire_col5__annual_burned")
        config = CatalogConfig(dict(layer), None, "test")
        self.assertIsNone(special_catalog_image(config.product))

    def test_every_new_style_resolves(self):
        from mapbiomas_data.core import VisualizationManager
        manager = VisualizationManager(config_loader())
        for dataset in ("brasil_soil_col3", "brasil_atmosphere_col1"):
            for product in config_loader().datasets[dataset]["products"].values():
                viz = manager.get_viz_params(product["visualization"])
                self.assertGreaterEqual(len(viz["palette"]), 2, product["visualization"])


if __name__ == "__main__":
    unittest.main()
