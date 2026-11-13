"""Config, periods and GCS paths of the U-Net experiment (no cloud access)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import (make_config, mosaic_band_paths, normalize_period, paths,  # noqa: E402
                         period_from_file_name, period_info, split_chip_name)


class TestPeriods(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_period("202610"), "2026_10")
        self.assertEqual(normalize_period(202610), "2026_10")
        self.assertEqual(normalize_period("2024.0"), "2024")
        self.assertEqual(normalize_period(float("nan")), "")
        self.assertEqual(normalize_period(None), "")

    def test_period_info(self):
        self.assertEqual(period_info("2026_10"), ("2026_10", "monthly"))
        self.assertEqual(period_info("2026_1"), ("2026_01", "monthly"))
        self.assertEqual(period_info(2024), ("2024", "yearly"))
        for bad in ("2026_13", "26_10", "2026_S1", "abc"):
            with self.assertRaises(ValueError):
                period_info(bad)

    def test_file_name_date(self):
        self.assertEqual(period_from_file_name("samples_brazil_monitor_2026_10.csv"), "2026_10")
        self.assertEqual(period_from_file_name("samples_brazil_2024.csv"), "2024")
        self.assertEqual(period_from_file_name("samples_brazil.csv"), "")

    def test_chip_name(self):
        self.assertEqual(split_chip_name("gs://b/x/2026_10_train_003.tfrecord.gz"), ("2026_10", "train", "003"))
        self.assertEqual(split_chip_name("/tmp/2024_val_000.tfrecord.gz"), ("2024", "val", "000"))


class TestPaths(unittest.TestCase):
    def test_monitor_defaults(self):
        cfg = make_config()
        self.assertEqual(cfg["CAMPAIGN"], "MONITOR_01")
        self.assertEqual(cfg["PERIODS"], ["2026_10"])
        p = mosaic_band_paths(cfg, "2026_10")
        # same object as
        # https://storage.googleapis.com/mapbiomas-fire/sudamerica/brazil/CATALOG_01/LIBRARY_IMAGES/
        #   SENTINEL2/MONTHLY/MINNBR/2026_10/COG/image_brazil_fire_sentinel2_minnbr_green_2026_10_cog.tif
        self.assertEqual(
            p["green"],
            "mapbiomas-fire/sudamerica/brazil/CATALOG_01/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/"
            "2026_10/COG/image_brazil_fire_sentinel2_minnbr_green_2026_10_cog.tif")
        self.assertEqual(set(p), {"green", "swir2"})  # no dayOfYear yet
        full = mosaic_band_paths(make_config(BANDS=["blue", "red", "nir", "swir1", "swir2"],
                                             VALID_BAND="dayOfYear"), "2026_10")
        self.assertTrue(full["dayOfYear"].endswith("image_brazil_fire_sentinel2_minnbr_dayOfYear_2026_10_cog.tif"))
        P = paths(cfg)
        self.assertEqual(P["samples"], "sudamerica/brazil/CATALOG_01/MONITOR_01/LIBRARY_SAMPLES")
        self.assertEqual(P["root"], "sudamerica/brazil/CATALOG_01/MONITOR_01/UNET_01")
        for key in ("chips", "stats", "splits", "models", "predict", "reports"):
            self.assertTrue(P[key].startswith(P["root"] + "/"), key)

    def test_annual_landsat(self):
        cfg = make_config(CAMPAIGN="COLLECTION_06", SENSOR="landsat", PERIODS=[2024], CHIP=128,
                          BANDS=["blue", "red", "nir", "swir1", "swir2"], VALID_BAND="dayOfYear")
        self.assertEqual(cfg["PERIODS"], ["2024"])
        self.assertEqual(
            mosaic_band_paths(cfg, 2024)["nir"],
            "mapbiomas-fire/sudamerica/brazil/CATALOG_01/LIBRARY_IMAGES/LANDSAT/ANNUALLY/MINNBR/"
            "2024/COG/image_brazil_fire_landsat_minnbr_nir_2024_cog.tif")

    def test_predict_periods(self):
        self.assertEqual(make_config()["PREDICT_PERIODS"], ["2026_10"])
        cfg = make_config(PERIODS=["2026_09", "2026_10"], PREDICT_PERIODS=["202611", "2026_10"])
        self.assertEqual(cfg["PREDICT_PERIODS"], ["2026_11", "2026_10"])

    def test_index_needs_bands(self):
        with self.assertRaises(ValueError):
            make_config(INDICES=["nbr"])  # default bands are green + swir2
        make_config(BANDS=["nir", "swir2"], INDICES=["nbr"])

    def test_unknown_key_fails(self):
        with self.assertRaises(KeyError):
            make_config(YEARS=[2024])

    def test_refuses_writes_outside(self):
        try:
            from unet_io import _check_inside
        except ImportError as exc:  # numpy missing
            self.skipTest(str(exc))
        cfg = make_config()
        _check_inside(cfg, paths(cfg)["reports"] + "/x.json")
        with self.assertRaises(PermissionError):
            _check_inside(cfg, paths(cfg)["samples"] + "/x.csv")


class TestOutputWithoutDoy(unittest.TestCase):
    def test_to_output(self):
        try:
            import numpy as np
            from unet_predict import NODATA, to_output
        except ImportError as exc:
            self.skipTest(str(exc))
        prob = np.array([[0.0, 0.4], [0.9, 0.2]])
        valid = np.array([[True, True], [True, False]])
        p, d = to_output(prob, valid, None)
        self.assertEqual(p.tolist(), [[0, 400], [900, NODATA]])
        self.assertTrue((d == NODATA).all())
        p, d = to_output(prob, valid, np.full((2, 2), 280))
        self.assertEqual(d.tolist(), [[NODATA, 280], [280, NODATA]])


class TestSamplePeriods(unittest.TestCase):
    def setUp(self):
        try:
            import pandas as pd  # noqa: F401
            from unet_io import sample_periods  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_rules_as_m4(self):
        import pandas as pd
        from unet_io import sample_periods
        df = pd.DataFrame({"period": ["202610", "", "2026_10"]})
        per, fixed = sample_periods(df, "s_brazil_2026_10.csv")
        self.assertEqual(per.tolist(), ["2026_10"] * 3)
        self.assertEqual(fixed, 0)
        # no column -> file date
        per, _ = sample_periods(pd.DataFrame({"x": [1]}), "s_2026_09.csv")
        self.assertEqual(per.tolist(), ["2026_09"])
        # later year than the file -> file date wins (M4 rule)
        per, fixed = sample_periods(pd.DataFrame({"period": ["2027_04", "2026_10"]}), "s_2026_10.csv")
        self.assertEqual(per.tolist(), ["2026_10", "2026_10"])
        self.assertEqual(fixed, 2)

    def test_chip_files(self):
        try:
            from unet_model import chip_files
        except ImportError as exc:
            self.skipTest(str(exc))
        files = ["/t/2026_10_train_000.tfrecord.gz", "/t/2026_09_train_000.tfrecord.gz",
                 "/t/2026_10_val_000.tfrecord.gz", "/t/notes.txt"]
        self.assertEqual(chip_files(files, ["2026_10"], "train"), ["/t/2026_10_train_000.tfrecord.gz"])


if __name__ == "__main__":
    unittest.main()
