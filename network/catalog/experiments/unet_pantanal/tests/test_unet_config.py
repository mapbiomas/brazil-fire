"""Config and GCS paths of the U-Net experiment (no cloud access)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config, mosaic_band_paths, paths  # noqa: E402


class TestPaths(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()

    def test_mosaic_paths_follow_catalog_names(self):
        p = mosaic_band_paths(self.cfg, 2024)
        self.assertEqual(
            p["nir"],
            "mapbiomas-fire/sudamerica/brazil/CATALOG_01/LIBRARY_IMAGES/LANDSAT/ANNUALLY/MINNBR/"
            "2024/COG/image_brazil_fire_landsat_minnbr_nir_2024_cog.tif")
        self.assertEqual(set(p), {"blue", "red", "nir", "swir1", "swir2", "dayOfYear"})

    def test_outputs_inside_experiment_folder(self):
        P = paths(self.cfg)
        self.assertEqual(P["samples"], "sudamerica/brazil/CATALOG_01/COLLECTION_06/LIBRARY_SAMPLES")
        self.assertEqual(P["root"], "sudamerica/brazil/CATALOG_01/COLLECTION_06/UNET_01")
        for key in ("chips", "stats", "splits", "models", "predict", "reports"):
            self.assertTrue(P[key].startswith(P["root"] + "/"), key)

    def test_unknown_key_fails(self):
        with self.assertRaises(KeyError):
            make_config(REGOIN="pantanal")

    def test_refuses_writes_outside(self):
        try:
            from unet_io import _check_inside
        except ImportError as exc:  # numpy missing
            self.skipTest(str(exc))
        _check_inside(self.cfg, paths(self.cfg)["reports"] + "/x.json")
        with self.assertRaises(PermissionError):
            _check_inside(self.cfg, paths(self.cfg)["samples"] + "/x.csv")


if __name__ == "__main__":
    unittest.main()
