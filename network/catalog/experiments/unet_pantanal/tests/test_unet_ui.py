"""Lógica da interface de escolha de períodos e bandas (sem GCS; widgets só se ipywidgets existir)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config  # noqa: E402
from unet_ui import available_periods, build_config, period_label  # noqa: E402

BASE = "mapbiomas-fire/sudamerica/brazil/CATALOG_01"
TREE = {
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR": ["2026_09", "2026_10", "lixo"],
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/2026_10/COG": [
        f"image_brazil_fire_sentinel2_minnbr_{b}_2026_10_cog.tif"
        for b in ("blue", "green", "red", "nir", "swir1", "swir2", "dayOfYear")],
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/2026_09/COG": [
        "image_brazil_fire_sentinel2_minnbr_red_2026_09_cog.tif"],
    f"{BASE}/MONITOR_01/LIBRARY_SAMPLES": ["samples_01_pantanal_r1_sentinel_2026_10.csv", "outro.txt"],
}


def fake_ls(path):
    return TREE.get(path.rstrip("/"), [])


class TestAvailablePeriods(unittest.TestCase):
    def test_lists_mosaic_and_samples(self):
        cfg = make_config()
        got = available_periods(cfg, ls=fake_ls)
        self.assertEqual(list(got), ["2026_09", "2026_10"])
        self.assertEqual(got["2026_10"]["bands"], {"blue", "green", "red", "nir", "swir1", "swir2", "dayOfYear"})
        self.assertEqual(got["2026_10"]["samples"], ["samples_01_pantanal_r1_sentinel_2026_10.csv"])
        self.assertEqual(got["2026_09"]["samples"], [])

    def test_label_flags_missing_bands(self):
        got = available_periods(make_config(), ls=fake_ls)
        self.assertIn("mosaico ✅", period_label("2026_10", got["2026_10"], ["blue", "swir2"]))
        self.assertIn("faltam bandas: blue, swir2", period_label("2026_09", got["2026_09"], ["red", "blue", "swir2"]))
        self.assertIn("sem amostras", period_label("2026_09", got["2026_09"], ["red"]))


class TestBuildConfig(unittest.TestCase):
    def test_ok_and_orders_bands(self):
        cfg = build_config(make_config(), ["2026_10"], [], ["swir2", "red", "nir", "blue"], ["nbr"], 30, 4)
        self.assertEqual(cfg["PERIODS"], ["2026_10"])
        self.assertEqual(cfg["PREDICT_PERIODS"], ["2026_10"])  # vazio = mesmo do treino
        self.assertEqual(cfg["BANDS"], ["blue", "red", "nir", "swir2"])
        self.assertEqual((cfg["EPOCHS"], cfg["BATCH"]), (30, 4))

    def test_separate_predict_periods(self):
        cfg = build_config(make_config(), ["2026_10"], ["2026_09", "2026_10"], ["red"], [], 5, 2)
        self.assertEqual(cfg["PREDICT_PERIODS"], ["2026_09", "2026_10"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            build_config(make_config(), [], [], ["red"], [], 5, 2)
        with self.assertRaises(ValueError):
            build_config(make_config(), ["2026_10"], [], [], [], 5, 2)
        with self.assertRaises(ValueError):  # índice sem as bandas dele
            build_config(make_config(), ["2026_10"], [], ["red"], ["nbr"], 5, 2)


@unittest.skipUnless(__import__("importlib").util.find_spec("ipywidgets"), "ipywidgets ausente")
class TestWidgets(unittest.TestCase):
    def test_selected_roundtrip(self):
        from unet_ui import ConfigUI
        cfg = make_config(PERIODS=["2026_10"], BANDS=["red", "nir", "swir2"], INDICES=["nbr"])
        ui = ConfigUI(cfg, listing=available_periods(cfg, ls=fake_ls))
        sel = ui.selected()
        self.assertEqual(sel["train_periods"], ["2026_10"])
        self.assertEqual(sel["bands"], ["red", "nir", "swir2"])
        self.assertEqual(sel["indices"], ["nbr"])
        ui.chk_pred[0].value = True  # 2026_09
        new = build_config(ui.cfg, **ui.selected())
        self.assertEqual(new["PREDICT_PERIODS"], ["2026_09", "2026_10"])


if __name__ == "__main__":
    unittest.main()
