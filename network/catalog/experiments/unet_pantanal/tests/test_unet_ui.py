"""Lógica da interface de escolha de períodos e bandas (sem GCS; widgets só se ipywidgets existir)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config  # noqa: E402
from unet_ui import (ClassificarUI, OpcoesUI, TreinoUI, available_periods, build_classificar,  # noqa: E402
                     build_opcoes, build_treino, mosaic_table, period_label,
                     predict_keys, train_keys)

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


class TestBuilders(unittest.TestCase):
    def test_treino_orders_bands(self):
        cfg = build_treino(make_config(), ["2026_10"], ["swir2", "red", "nir", "blue"], ["nbr"])
        self.assertEqual(cfg["PERIODS"], ["2026_10"])
        self.assertEqual(cfg["BANDS"], ["blue", "red", "nir", "swir2"])
        self.assertEqual(cfg["INDICES"], ["nbr"])

    def test_treino_errors(self):
        with self.assertRaises(ValueError):
            build_treino(make_config(), [], ["red"], [])
        with self.assertRaises(ValueError):
            build_treino(make_config(), ["2026_10"], [], [])
        with self.assertRaises(ValueError):  # índice sem as bandas dele
            build_treino(make_config(), ["2026_10"], ["red"], ["nbr"])

    def test_opcoes_and_classificar(self):
        cfg = build_opcoes(make_config(), 30, 4)
        self.assertEqual((cfg["EPOCHS"], cfg["BATCH"]), (30, 4))
        with self.assertRaises(ValueError):
            build_opcoes(make_config(), 0, 4)
        cfg = build_classificar(make_config(), ["2026_10", "2026_09"])
        self.assertEqual(cfg["PREDICT_PERIODS"], ["2026_09", "2026_10"])
        self.assertEqual(cfg["PERIODS"], ["2026_10"])  # treino não muda
        with self.assertRaises(ValueError):
            build_classificar(make_config(), [])


class TestKeys(unittest.TestCase):
    def test_train_needs_samples_predict_needs_mosaic(self):
        got = available_periods(make_config(), ls=fake_ls)
        got["2027_01"] = {"bands": set(), "samples": ["x_2027_01.csv"]}  # amostra sem mosaico
        self.assertEqual(train_keys(got), ["2026_10"])
        self.assertEqual(predict_keys(got), ["2026_09", "2026_10"])


class TestMosaicos(unittest.TestCase):
    def test_table(self):
        rows = mosaic_table(available_periods(make_config(), ls=fake_ls))
        self.assertEqual(len(rows), 3)
        self.assertIn("2026_10", rows[2])
        self.assertEqual(rows[2].count("✅"), 7)
        self.assertEqual(rows[1].count("✅"), 1)


@unittest.skipUnless(__import__("importlib").util.find_spec("ipywidgets"), "ipywidgets ausente")
class TestWidgets(unittest.TestCase):
    def test_treino_roundtrip(self):
        cfg = make_config(PERIODS=["2026_10"], BANDS=["red", "nir", "swir2"], INDICES=["nbr"])
        ui = TreinoUI(cfg, listing=available_periods(cfg, ls=fake_ls))
        self.assertEqual(ui.selected(), dict(train_periods=["2026_10"], bands=["red", "nir", "swir2"], indices=["nbr"]))
        self.assertEqual(ui.keys, ["2026_10"])  # 2026_09 não tem amostras: não aparece no treino

    def test_classificar_and_opcoes(self):
        cfg = make_config()
        ui = ClassificarUI(cfg, listing=available_periods(cfg, ls=fake_ls))
        ui.chk_pred[0].value = True
        self.assertEqual(build_classificar(ui.cfg, **ui.selected())["PREDICT_PERIODS"], ["2026_09", "2026_10"])
        op = OpcoesUI(cfg)
        op.w_epochs.value = 12
        self.assertEqual(build_opcoes(op.cfg, **op.selected())["EPOCHS"], 12)


if __name__ == "__main__":
    unittest.main()
