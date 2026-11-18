"""Notebook em 3 blocos: lógica pura (parâmetros, cartões) e construção das interfaces sem Colab."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from unet_config import make_config  # noqa: E402
import unet_app  # noqa: E402

CFG = make_config(REGION="Pantanal_r1", SAMPLE_REGION="Pantanal_r1")
RES = {"val_best_threshold": 0.63,
       "test_at_best": {"tp": 80, "fn": 20, "fp": 30, "tn": 870, "precision": 80 / 110, "recall": 0.8, "f1": 0.77,
                        "iou": 0.62, "omission": 0.2, "commission": 30 / 110},
       "test_at_0_5": {"tp": 90, "fn": 10, "fp": 90, "tn": 810, "precision": 0.5, "recall": 0.9, "f1": 0.64,
                       "iou": 0.47, "omission": 0.1, "commission": 0.5},
       "val_curve": {"threshold": [0.25, 0.5, 0.75], "f1": [0.5, 0.7, 0.6], "precision": [0.4, 0.6, 0.8],
                     "recall": [0.9, 0.8, 0.5]}}


class TestPure(unittest.TestCase):
    def test_regions(self):
        self.assertEqual(unet_app.split_region("Pantanal_r1"), ("Pantanal", "Pantanal_r1"))
        self.assertEqual(unet_app.split_region("Mata Atlântica_r12")[0], "Mata Atlântica")
        self.assertEqual(unet_app.split_region("Pampa")[0], "Pampa")
        self.assertEqual(unet_app.regions_by_biome(["Pantanal_r1", "Cerrado_r2", "Cerrado_r1", "Cerrado_r1"]),
                         {"Cerrado": ["Cerrado_r1", "Cerrado_r2"], "Pantanal": ["Pantanal_r1"]})

    def test_fmt_tempo(self):
        f = unet_app.fmt_tempo
        self.assertEqual((f(5), f(125), f(3599), f(4020), f(7260)), ("5 s", "2 min 05 s", "59 min 59 s", "1 h 07 min", "2 h 01 min"))

    def values(self, **over):
        v = dict(EPOCHS=30, BATCH=4, LR=5e-5, PATIENCE=5, FILTERS=[16, 32, 64, 128], MIXED_PRECISION=True,
                 CHIP=256, POINTS_PER_POLYGON=4, BLOCK_KM=25, SPLIT=(0.7, 0.15, 0.15), SEED=42, OVERLAP=16,
                 INFER_TILE=512, INFER_BATCH=8)
        v.update(over)
        return v

    def test_build_params(self):
        new, chips = unet_app.build_params(CFG, **self.values())
        self.assertEqual((new["EPOCHS"], new["BATCH"], new["LR"]), (30, 4, 5e-5))
        self.assertEqual((new["FILTERS"], new["BOTTOM_FILTERS"]), ([16, 32, 64, 128], 256))
        self.assertEqual(chips, [])  # nada que mude os chips
        new, chips = unet_app.build_params(CFG, **self.values(BLOCK_KM=10, SPLIT=(0.8, 0.1, 0.1)))
        self.assertEqual(chips, ["BLOCK_KM", "SPLIT"])

    def test_build_params_validates(self):
        with self.assertRaises(ValueError):
            unet_app.build_params(CFG, **self.values(EPOCHS=0))
        with self.assertRaises(ValueError):  # 500 + 32 não é múltiplo de 16
            unet_app.build_params(CFG, **self.values(INFER_TILE=500))

    def test_cards_and_diagnosis(self):
        html = unet_app.cards_html(RES)
        for txt in ("80%", "0.63", "de cada 100 pixels que queimaram", "alarme falso", "100 deles são fogo"):
            self.assertIn(txt, html)
        self.assertEqual(unet_app.diagnostico(RES)[0][0], "ℹ️")
        self.assertEqual(unet_app.pick_summary({"2026_10": 1, "all": 2}), 2)
        self.assertEqual(unet_app.pick_summary({"2026_10": 1}), 1)


try:
    import ipywidgets  # noqa: F401
    HAVE_W = True
except ImportError:
    HAVE_W = False


@unittest.skipUnless(HAVE_W, "ipywidgets ausente")
class TestWidgets(unittest.TestCase):
    def test_config_confirm_new_and_existing(self):
        estado = unet_app.Estado(CFG)
        versions = {"Pantanal_r1": ["training_001_pantanal_r1_sentinel2"]}
        ui = unet_app.ConfigUI(estado, regions=["Pantanal_r1", "Cerrado_r1", "Cerrado_r2"],
                               versions_fn=lambda c: versions.get(c["REGION"], []))
        self.assertEqual(list(ui.dd_bioma.options), ["Cerrado", "Pantanal"])
        self.assertEqual(ui.dd_bioma.value, "Pantanal")
        self.assertEqual([v for _, v in ui.dd_treino.options], ["", "training_001_pantanal_r1_sentinel2"])
        ui.dd_bioma.value = "Cerrado"
        self.assertEqual(ui.region(), "Cerrado_r1")
        self.assertEqual([v for _, v in ui.dd_treino.options], [""])  # outra região: nenhum treino
        ui.confirmar()
        self.assertTrue(estado.confirmado and not estado.existente)
        self.assertEqual(estado.cfg["REGION"], "Cerrado_r1")
        self.assertEqual(estado.cfg["SAMPLE_REGION"], "Cerrado_r1")
        self.assertIn("cerrado", ui.lbl.value)
        # treino existente: reabre o config.json salvo
        saved = dict(CFG, TRAINING_ID="001", SHORTNAME="pantanal_r1", EPOCHS=12)
        from unittest import mock
        ui.dd_bioma.value = "Pantanal"
        with mock.patch.object(unet_app.ui, "load_version",
                               lambda base, name: make_config(**dict(saved, MODEL_NAME=name))):
            ui.dd_treino.value = "training_001_pantanal_r1_sentinel2"
            ui.confirmar()
        self.assertTrue(estado.existente)
        self.assertEqual(estado.cfg["EPOCHS"], 12)

    def test_config_without_regions_uses_text(self):
        estado = unet_app.Estado(CFG)
        ui = unet_app.ConfigUI(estado, regions=[], versions_fn=lambda c: [])
        self.assertIsNone(ui.dd_bioma)
        ui.txt_regiao.value = "Pampa_r1"
        ui.confirmar()
        self.assertEqual(estado.cfg["REGION"], "Pampa_r1")

    def test_params_ui_roundtrip(self):
        pui = unet_app.ParamsUI(CFG)
        v = pui.values()
        new, chips = unet_app.build_params(CFG, **v)
        self.assertEqual(chips, [])
        for k in ("EPOCHS", "BATCH", "LR", "FILTERS", "SPLIT", "CHIP", "INFER_TILE"):
            self.assertEqual(new[k], CFG[k] if not isinstance(CFG[k], list) else list(CFG[k]))

    def test_blocks_build_with_fake_gcs(self):
        from unittest import mock
        from test_unet_ui import fake_ls
        import unet_env
        estado = unet_app.Estado(CFG)
        estado.confirmado = True
        with mock.patch.object(unet_env, "ls", fake_ls):
            self.assertIsNone(unet_app.abrir_treino(estado))  # devolver o widget duplicaria a interface
            treino = estado.tabs_treino
            self.assertEqual(len(treino.children), 4)
            self.assertEqual(treino.get_title(3), "4 Métricas")
            self.assertIsNone(unet_app.abrir_classificacao(estado))
            clas = estado.tabs_class
            self.assertEqual(len(clas.children), 4)
            self.assertIsNotNone(estado.cfg_class)
        estado.existente = True
        with mock.patch.object(unet_env, "ls", fake_ls):
            unet_app.abrir_treino(estado)
            self.assertEqual(len(estado.tabs_treino.children), 4)  # versão aberta: só o resumo na aba 1

    def test_blocks_wait_for_confirmation(self):
        estado = unet_app.Estado(CFG)
        self.assertIsNone(unet_app.abrir_treino(estado))
        self.assertIsNone(unet_app.abrir_classificacao(estado))


@unittest.skipUnless(HAVE_W, "ipywidgets ausente")
class TestPlots(unittest.TestCase):
    def test_figures_run(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("matplotlib ausente")
        unet_app.fig_limiar(RES)
        unet_app.fig_matriz(RES)
        unet_app.fig_comparacao({"A_mlp_bands": RES, "C_unet": RES})


if __name__ == "__main__":
    unittest.main()
