"""Interfaces do notebook: lógica sem GCS (ls falso); widgets só se ipywidgets existir."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config, paths  # noqa: E402
from unet_ui import (auto_shortname, available_periods, build_classificar, build_opcoes,  # noqa: E402
                     build_treino, config_from_saved, list_versions, mosaic_table, next_training_id,
                     period_label, predict_keys, ready_keys, sample_choices, sample_id, train_keys)

BASE = "mapbiomas-fire/sudamerica/brazil/CATALOG_01"
CSV = "samples_01_pantanal_r1_2026_10.csv"
TREE = {
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR": ["2026_09", "2026_10", "lixo"],
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/2026_10/COG": [
        f"image_brazil_fire_sentinel2_minnbr_{b}_2026_10_cog.tif"
        for b in ("blue", "green", "red", "nir", "swir1", "swir2", "dayOfYear")],
    f"{BASE}/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/2026_09/COG": [
        "image_brazil_fire_sentinel2_minnbr_red_2026_09_cog.tif",
        "image_brazil_fire_sentinel2_minnbr_dayOfYear_2026_09_cog.tif"],
    f"{BASE}/MONITOR_01/LIBRARY_SAMPLES": [CSV, "samples_02_pantanal_r1_2027_01.csv", "outro.txt"],
    f"{BASE}/MONITOR_01/UNET_01": ["chips", "training_009_antigo_sentinel2", "pantanal"],
    f"{BASE}/MONITOR_01/UNET_01/pantanal/pantanal_r1": ["chips", "training_001_pantanal_r1_6c_sentinel2",
                                                     "training_003_x_sentinel2", "models"],
}


def fake_ls(path):
    return TREE.get(path.rstrip("/"), [])


class TestListing(unittest.TestCase):
    def test_lists_mosaic_and_samples(self):
        got = available_periods(make_config(), ls=fake_ls)
        self.assertEqual(list(got), ["2026_09", "2026_10", "2027_01"])
        self.assertEqual(got["2026_10"]["bands"], {"blue", "green", "red", "nir", "swir1", "swir2", "dayOfYear"})
        self.assertEqual(got["2026_10"]["samples"], [CSV])
        self.assertEqual(got["2026_09"]["samples"], [])
        self.assertEqual(got["2027_01"]["bands"], set())  # amostra sem mosaico

    def test_keys_and_choices(self):
        got = available_periods(make_config(), ls=fake_ls)
        self.assertEqual(train_keys(got), ["2026_10"])
        self.assertEqual(predict_keys(got), ["2026_09", "2026_10"])
        self.assertEqual(sample_choices(got), [(CSV, "2026_10")])  # 2027_01 não tem mosaico

    def test_ready_keys(self):
        got = available_periods(make_config(), ls=fake_ls)
        ok, missing = ready_keys(got, ["red", "nir", "dayOfYear"])
        self.assertEqual(ok, ["2026_10"])  # 2026_09 só tem red e dayOfYear
        self.assertEqual(missing, {"2026_09": ["nir"]})
        self.assertEqual(ready_keys(got, ["red", "dayOfYear"])[0], ["2026_10", "2026_09"])  # novos primeiro

    def test_label_flags_missing_bands(self):
        got = available_periods(make_config(), ls=fake_ls)
        self.assertIn("mosaico ✅", period_label("2026_10", got["2026_10"], ["blue", "swir2"]))
        self.assertIn("faltam bandas: blue, swir2", period_label("2026_09", got["2026_09"], ["red", "blue", "swir2"]))
        self.assertIn("sem amostras", period_label("2026_09", got["2026_09"], ["red"]))

    def test_mosaic_table(self):
        rows = mosaic_table(available_periods(make_config(), ls=fake_ls))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[2].count("✅"), 7)
        self.assertEqual(rows[1].count("✅"), 2)


class TestVersions(unittest.TestCase):
    def test_list_and_next_id(self):
        v = list_versions(make_config(), ls=fake_ls)
        self.assertEqual(v, ["training_001_pantanal_r1_6c_sentinel2", "training_003_x_sentinel2"])
        self.assertEqual(next_training_id(v), "002")
        self.assertEqual(next_training_id([]), "001")

    def test_paths_per_version(self):
        cfg = make_config(TRAINING_ID="7", SHORTNAME="Pantanal R1 6c")
        P = paths(cfg)
        self.assertEqual(cfg["MODEL_NAME"], "training_007_pantanal_r1_6c_sentinel2")
        self.assertTrue(P["root"].endswith("/UNET_01/pantanal/pantanal_r1/training_007_pantanal_r1_6c_sentinel2"))
        self.assertTrue(P["chips"].startswith(P["root"] + "/"))
        self.assertEqual(P["experiment"], f"{BASE.split('/', 1)[1]}/MONITOR_01/UNET_01")
        self.assertEqual(paths(make_config())["root"], P["experiment"])  # sem versão: layout antigo

    def test_biome_folders(self):
        from unet_config import biome_slug
        self.assertEqual(biome_slug(make_config(REGION="Mata_Atlântica_r3")), "mata_atlantica")
        self.assertEqual(biome_slug(make_config(REGION="Cerrado_r12")), "cerrado")
        self.assertEqual(biome_slug(make_config(REGION="Cerrado_r12", BIOME="Cerrado Norte")), "cerrado_norte")
        P = paths(make_config(REGION="Mata_Atlântica_r3", TRAINING_ID="1"))
        self.assertTrue(P["versions"].endswith("/UNET_01/mata_atlantica/mata_atlantica_r3"))
        self.assertTrue(P["root"].startswith(P["versions"] + "/training_001_"))

    def test_auto_shortname_and_sample_id(self):
        self.assertEqual(auto_shortname(make_config(REGION="Pantanal_r1")), "pantanal_r1")
        self.assertEqual(sample_id([CSV]), "01")
        self.assertEqual(sample_id(["samples_07_x_2026_10.csv", CSV]), "01")
        self.assertIsNone(sample_id(["amostras_2026_10.csv"]))

    def test_saved_roundtrip(self):
        import json
        cfg = build_treino(make_config(), [CSV], ["red", "nir", "swir2"], ["nbr"], "4", "teste")
        saved = json.loads(json.dumps(cfg, default=str))
        back = config_from_saved(make_config(), saved)
        for k in ("SAMPLE_FILES", "PERIODS", "BANDS", "INDICES", "MODEL_NAME", "TRAINING_ID", "SPLIT"):
            self.assertEqual(back[k], cfg[k], k)


class TestBuilders(unittest.TestCase):
    def test_treino(self):
        cfg = build_treino(make_config(), [CSV], ["swir2", "red", "nir", "blue"], ["nbr"], "2", "", "obs")
        self.assertEqual(cfg["PERIODS"], ["2026_10"])  # sai do nome do CSV
        self.assertEqual(cfg["SAMPLE_FILES"], [CSV])
        self.assertEqual(cfg["BANDS"], ["blue", "red", "nir", "swir2"])
        self.assertEqual(cfg["TRAINING_ID"], "002")
        self.assertEqual(cfg["COMMENT"], "obs")
        self.assertTrue(cfg["MODEL_NAME"].startswith("training_002_"))

    def test_treino_errors(self):
        with self.assertRaises(ValueError):
            build_treino(make_config(), [], ["red"], [])
        with self.assertRaises(ValueError):
            build_treino(make_config(), [CSV], [], [])
        with self.assertRaises(ValueError):  # índice sem as bandas dele
            build_treino(make_config(), [CSV], ["red"], ["nbr"])
        with self.assertRaises(ValueError):  # ID não numérico
            build_treino(make_config(), [CSV], ["red"], [], "abc")
        with self.assertRaises(ValueError):  # nome sem período
            build_treino(make_config(), ["amostras.csv"], ["red"], [])

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


@unittest.skipUnless(__import__("importlib").util.find_spec("ipywidgets"), "ipywidgets ausente")
class TestWidgets(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config(PERIODS=["2026_10"], BANDS=["red", "nir", "swir2"], INDICES=["nbr"],
                               VALID_BAND="dayOfYear")
        self.listing = available_periods(self.cfg, ls=fake_ls)
        self.versions = list_versions(self.cfg, ls=fake_ls)

    def test_treino_new(self):
        from unet_ui import TreinoUI, apply_treino
        ui = TreinoUI(self.cfg, self.listing, self.versions)
        self.assertEqual(ui.selected_samples, {CSV})  # pré-seleção pelos períodos da configuração
        self.assertEqual(ui.w_id.value, "01")  # vem de samples_01_...
        self.assertEqual(ui.w_short.value, "pantanal_r1")
        self.assertIn("já é usado", ui.lbl_version.value)  # training_001 já existe no GCS falso
        ui._clear()
        self.assertEqual(ui.w_id.value, "002")  # sem amostras com ID: menor ID livre
        ui.w_id.value = "5"
        ui.w_short.value = "meu_teste"
        ui._select_all()
        self.assertEqual((ui.w_id.value, ui.w_short.value), ("5", "meu_teste"))  # digitado: não é sobrescrito
        ui.chk_bands[4].value = True  # swir1
        ui.chk_idx[1].value = True  # nbr2
        cfg = apply_treino(ui, save=False)
        self.assertEqual(cfg["MODEL_NAME"], "training_005_meu_teste_sentinel2")
        self.assertEqual(cfg["INDICES"], ["nbr", "nbr2"])
        self.assertEqual(cfg["SAMPLE_FILES"], [CSV])

    def test_treino_id_clash_warns(self):
        from unet_ui import TreinoUI
        ui = TreinoUI(self.cfg, self.listing, self.versions)
        ui.w_id.value = "1"
        self.assertIn("já é usado", ui.lbl_version.value)

    def test_existing_version_tab(self):
        from unet_ui import TreinoUI
        ui = TreinoUI(self.cfg, self.listing, self.versions)
        ui.tabs.selected_index = 1
        self.assertEqual(ui.mode, "existente")
        self.assertEqual(ui.dd_version.value, self.versions[0])

    def test_classificar_and_opcoes(self):
        from unet_ui import ClassificarUI, OpcoesUI
        other = build_treino(self.cfg, [CSV], ["red"], [], "3", "x")
        ui = ClassificarUI(self.cfg, self.listing, self.versions, loader=lambda name: other)
        self.assertEqual(ui.dd_version.value, "(configuração atual)")
        self.assertEqual(ui.keys, ["2026_10"])  # 2026_09 só tem red e dayOfYear
        self.assertIn("2026_09 (faltam nir, swir2)", ui.lbl_info.value)
        self.assertTrue(ui.chk_pred[0].value)  # mês de treino já marcado
        ui.dd_version.value = "training_003_x_sentinel2"
        self.assertEqual(ui.cfg["BANDS"], ["red"])
        self.assertEqual(ui.keys, ["2026_10", "2026_09"])  # o modelo 003 só precisa de red
        self.assertTrue(ui.chk_pred[0].value)  # a escolha anterior é mantida
        ui._set_all(False)
        self.assertEqual(ui.selected(), {"predict_periods": []})
        ui._set_all(True)
        new = build_classificar(ui.cfg, **ui.selected())
        self.assertEqual(new["PREDICT_PERIODS"], ["2026_09", "2026_10"])
        self.assertEqual(new["MODEL_NAME"], "training_003_x_sentinel2")
        from unet_ui import PublicarUI
        cfg = make_config(TRAINING_ID="1", SHORTNAME="pantanal_r1")
        pub = PublicarUI(cfg, maps=["2026_09", "2026_10"], assets={"pantanal_r1_2026_10"})
        self.assertIn("GEE ✅", pub.chk[1].description)
        self.assertEqual(pub.selected(), ["2026_09"])  # só o que ainda não está no GEE
        self.assertEqual(PublicarUI(cfg, maps=[], assets=set()).selected(), [])  # versão sem mapas
        op = OpcoesUI(self.cfg)
        op.w_epochs.value = 12
        self.assertEqual(build_opcoes(op.cfg, **op.selected())["EPOCHS"], 12)


if __name__ == "__main__":
    unittest.main()
