import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType
from gif_factory.engine import validate, generate, NotebookConfig
from gif_factory.storage import copy_artifacts, list_artifacts
from gif_factory.presets import LAYERS, TERRITORIES
from gif_factory.layout import combination_path, ensure_unique_paths, safe_name


class BasicTests(unittest.TestCase):
    def test_presets_and_invalid_id(self):
        validate(LAYERS, TERRITORIES)
        with self.assertRaises(ValueError):
            validate([{**LAYERS[0], "id": "../bad"}], TERRITORIES)

    def test_copy_is_selective_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "output"
            root.mkdir()
            first = root / "one.gif"
            first.write_bytes(b"first")
            (root / "two.png").write_bytes(b"second")
            dest = Path(tmp) / "drive"
            self.assertEqual(len(list_artifacts(root)), 2)
            self.assertEqual(copy_artifacts([first], root, dest)[0][1], "copiado")
            first.write_bytes(b"changed")
            self.assertEqual(copy_artifacts([first], root, dest)[0][1], "já existe; não sobrescrito")
            self.assertEqual((dest / "one.gif").read_bytes(), b"first")
            self.assertFalse((dest / "two.png").exists())

    def test_reject_outside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside.gif"
            outside.write_bytes(b"x")
            self.assertEqual(copy_artifacts([outside], root, Path(tmp) / "dest")[0][1], "falha")

    def test_config_preserves_filter_and_cartographic_labels(self):
        cfg = NotebookConfig(LAYERS[0], TERRITORIES[0], "my-project")
        self.assertEqual(cfg.ee_project_id, "my-project")
        self.assertEqual(cfg.territories["biomes"][TERRITORIES[0]["id"]]["filter_in"]["values"], ["Amazônia"])
        self.assertEqual(cfg.visualizations["notebook"]["discrete_labels"], ["Não observado", "Fogo"])

    def test_pipeline_adapter_only_exports_selected_files(self):
        module = ModuleType("mapbiomas_data.core.pipeline")
        class FakePipeline:
            def __init__(self, config):
                self.downloader = type("Downloader", (), {})()
            def run(self, dataset, product, territory, output_dir, **kwargs):
                self.downloader.progress_callback("frame_done", 1, 1, "band")
                for folder, filename in [("gifs", "a.gif"), ("frames_maps", "a.png"), ("frames_pure", "b.png")]:
                    path = Path(output_dir) / folder
                    path.mkdir()
                    (path / filename).write_bytes(b"file")
                return {"status": "success"}
        module.Pipeline = FakePipeline
        events, artifacts = [], []
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {"mapbiomas_data.core.pipeline": module}):
            result = generate(LAYERS[0], TERRITORIES[0], tmp, {"gif", "finais"}, artifacts.append,
                              project="test", on_progress=lambda *event: events.append(event))
            self.assertEqual(len(result), 2)
            self.assertEqual(result, artifacts)
            self.assertEqual(events[0][0], "frame_done")
            self.assertFalse((Path(tmp) / "Frames/Mapa_basico").exists())
            self.assertTrue((Path(tmp) / "GIFs/Amazonia_a.gif").exists())
            self.assertTrue((Path(tmp) / "Frames/Mapa_completo/Amazonia_a.png").exists())

    def test_layout_groups_biomes_and_detects_duplicates(self):
        amazon = combination_path("Brasil", LAYERS[0], TERRITORIES[0])
        caatinga = combination_path("Brasil", LAYERS[0], TERRITORIES[1])
        self.assertEqual(amazon.parent, caatinga.parent)
        self.assertEqual(str(amazon).replace("\\", "/"), "Brasil/Fogo/Colecao_5/Area_queimada_anual/Amazonia")
        combos = [("Brasil", LAYERS[0], TERRITORIES[0]), ("Brasil", LAYERS[0], TERRITORIES[1])]
        ensure_unique_paths(combos)
        with self.assertRaises(ValueError):
            ensure_unique_paths(combos + [combos[0]])

    def test_custom_names_are_safe_and_collision_is_not_silent(self):
        self.assertNotIn("/", safe_name("../../Mata Atlântica"))
        custom = {**LAYERS[0], "name": "Área de teste", "theme": "Água", "collection": "Coleção 1"}
        self.assertEqual(str(combination_path("Personalizado", custom, TERRITORIES[0])).replace("\\", "/"),
                         "Personalizado/Agua/Colecao_1/Area_de_teste/Amazonia")
        with self.assertRaises(ValueError):
            ensure_unique_paths([("Personalizado", custom, TERRITORIES[0]),
                                 ("Personalizado", {**custom, "name": "Area de teste"}, TERRITORIES[0])])

    def test_drive_preserves_shared_product_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            local, drive = Path(tmp) / "local", Path(tmp) / "drive"
            files = []
            for territory in TERRITORIES[:2]:
                path = local / "Entrega_2026" / combination_path("Brasil", LAYERS[0], territory) / "GIFs" / "mapa.gif"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"gif")
                files.append(path)
            self.assertTrue(all(item[1] == "copiado" for item in copy_artifacts(files, local, drive)))
            product = drive / "Entrega_2026/Brasil/Fogo/Colecao_5/Area_queimada_anual"
            self.assertTrue((product / "Amazonia/GIFs/mapa.gif").exists())
            self.assertTrue((product / "Caatinga/GIFs/mapa.gif").exists())


if __name__ == "__main__":
    unittest.main()
