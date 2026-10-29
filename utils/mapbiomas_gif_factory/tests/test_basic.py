import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from types import ModuleType
from gif_factory.engine import validate, generate, NotebookConfig
from gif_factory.storage import copy_artifacts, list_artifacts
from gif_factory.presets import LAYERS, TERRITORIES
from gif_factory.layout import combination_path, ensure_unique_paths, safe_name
from gif_factory.temporal import resolve_bands, sequence_labels
from gif_factory.progress import ProgressState
from gif_factory.library import assess, fingerprint, ingest, sync_selected, variant_for


class BasicTests(unittest.TestCase):
    def test_progress_clock_retry_estimate_and_stages(self):
        now = [100.0]
        state = ProgressState(2, clock=lambda: now[0])
        state.begin(1, "Brasil / Cerrado")
        state.update("discovered", ["1985", "1986", "1987"])
        state.update("frame_start", 1, 3, "1985")
        now[0] += 5
        state.update("frame_retry", 1, 3, "1985", 1, 3, 2, "erro")
        self.assertIn("tentativa 1 de 3", state.snapshot()["detail"])
        self.assertEqual(state.snapshot()["elapsed"], 5)
        state.update("frame_done", 1, 3, "1985", 5)
        self.assertIsNone(state.snapshot()["eta"])
        state.update("frame_start", 2, 3, "1986")
        state.update("frame_done", 2, 3, "1986", 3)
        self.assertEqual(state.snapshot()["eta"], 4)
        self.assertEqual(state.snapshot()["frame_done"], 2)
        state.update("stage", "Gerando GIF")
        self.assertIsNone(state.snapshot()["eta"])
        state.finish_combo()
        self.assertEqual(state.snapshot()["completed"], 1)
        now[0] += 3
        state.begin(2, "Brasil / Amazônia")
        self.assertEqual(state.snapshot()["frame_done"], 0)
        self.assertEqual(state.snapshot()["elapsed"], 8)
        self.assertEqual(state.snapshot()["combo_elapsed"], 0)

    def test_progress_frame_failure_keeps_real_index(self):
        state = ProgressState(1)
        state.begin(1, "Teste")
        state.update("frame_start", 1, 3, "a")
        state.update("frame_failed", 1, 3, "a", "sem acesso")
        self.assertIn("falhou", state.snapshot()["detail"])
        state.update("frame_start", 2, 3, "b")
        self.assertIn("Frame 2 de 3", state.snapshot()["detail"])
        state.update("frame_done", 2, 3, "b", 1)
        self.assertEqual(state.snapshot()["frame_done"], 1)

    def test_presets_and_invalid_id(self):
        validate(LAYERS, TERRITORIES)
        self.assertTrue({"pais_brasil", "peru", "peru_bioma_andes", "peru_bioma_desierto", "peru_bioma_bosque_seco"}.issubset({x["id"] for x in TERRITORIES}))
        with self.assertRaises(ValueError):
            validate([{**LAYERS[0], "id": "../bad"}], TERRITORIES)

    def test_pending_layer_requires_reason_and_valid_id(self):
        pending = {"id": "lulc_10", "name": "Uso e cobertura", "enabled": False, "reason": "Legenda pendente"}
        validate([pending], TERRITORIES)
        with self.assertRaises(ValueError):
            validate([{**pending, "reason": ""}], TERRITORIES)
        with self.assertRaises(ValueError):
            validate([pending, pending], TERRITORIES)

    def test_temporal_discovery_and_range(self):
        self.assertEqual(resolve_bands(["classification_2024", "classification_1985", "classification_2023"], start_year=2023),
                         ["classification_2023", "classification_2024"])
        self.assertEqual(resolve_bands(["freq_1985_2024", "freq_1985_2023"]), ["freq_1985_2023", "freq_1985_2024"])
        self.assertEqual(resolve_bands(["f_2020_12", "f_2020_01"]), ["f_2020_01", "f_2020_12"])
        with self.assertRaises(ValueError):
            resolve_bands(["foo", "bar"])
        with self.assertRaises(ValueError):
            resolve_bands(["a_2023", "b_2023"])
        with self.assertRaises(ValueError):
            resolve_bands(["classification_2020"], start_year=2021)
        self.assertEqual(resolve_bands(["foo", "bar"], ["bar", "foo"]), ["bar", "foo"])

    def test_cumulative_windows_are_labelled_by_end_year(self):
        labels = sequence_labels(["fire_frequency_1985_1985", "fire_frequency_1985_1999", "fire_frequency_1985_2025"])
        self.assertEqual(labels, ["1985", "1999", "2025"])
        self.assertEqual(sequence_labels(["burned_area_2020", "burned_area_2021"]), ["2020", "2021"])
        self.assertEqual(sequence_labels(["ano_2020", "ano_2021"], year_offset=-1), ["2019", "2020"])
        self.assertEqual(sequence_labels(["depth_000_010", "depth_010_020"], axis="depth"),
                         ["0–10 cm", "10–20 cm"])
        # Janelas com inícios diferentes mantêm o intervalo explícito.
        self.assertEqual(sequence_labels(["a_1985_2000", "a_1990_2020"]), ["1985→2000", "1990→2020"])

    def test_depth_sequence_is_not_interpreted_as_years(self):
        depths = ["depth_030_060", "depth_000_030", "depth_000_010", "depth_000_020"]
        self.assertEqual(resolve_bands(depths, axis="depth"), ["depth_000_010", "depth_000_020", "depth_000_030", "depth_030_060"])
        with self.assertRaises(ValueError):
            resolve_bands(depths, start_year=2020, axis="depth")
        with self.assertRaises(ValueError):
            resolve_bands(["depth_030_010"], axis="depth")
        config = NotebookConfig({**LAYERS[0], "axis": "depth"}, TERRITORIES[0], "p")
        self.assertEqual(config.datasets["notebook"]["products"][LAYERS[0]["id"]]["axis"], "depth")

    def test_layer_year_limits_are_independent_and_validated(self):
        layer = {**LAYERS[0], "start_year": 2020, "end_year": 2023}
        validate([layer], TERRITORIES)
        cfg = NotebookConfig(layer, TERRITORIES[0], "project")
        product = cfg.datasets["notebook"]["products"][layer["id"]]
        self.assertEqual((product["start_year"], product["end_year"]), (2020, 2023))
        self.assertIsNone(NotebookConfig(LAYERS[0], TERRITORIES[0], "project").datasets["notebook"]["products"][LAYERS[0]["id"]]["start_year"])
        for invalid in ({"start_year": 2025, "end_year": 2020}, {"start_year": "2020"}, {"end_year": -1}):
            with self.assertRaises(ValueError):
                validate([{**LAYERS[0], **invalid}], TERRITORIES)

    def test_notebook_custom_arrays_are_explicit_and_valid(self):
        notebook = json.loads((Path(__file__).parents[1] / "notebooks/piloto.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        for cell in notebook["cells"][2:4]:
            exec("".join(cell["source"]), namespace)
        layers, territories = namespace["LAYERS"], namespace["TERRITORIES"]
        validate(layers, territories)
        self.assertEqual({x["id"] for x in territories}, {"matopiba", "matopiba_cerrado"})
        self.assertEqual([x["id"] for x in layers], ["fogo_anual"])
        self.assertNotIn("bands", layers[0])
        self.assertEqual(layers[0]["status"], "experimental")

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

    def test_territory_filter_supports_one_or_many_values(self):
        territory = {**TERRITORIES[0], "filter": {"field": "Bioma", "values": ["Cerrado", "Caatinga"]}}
        validate([LAYERS[0]], [territory])
        cfg = NotebookConfig(LAYERS[0], territory, "project")
        self.assertEqual(cfg.territories["biomes"][territory["id"]]["filter_in"]["values"], ["Cerrado", "Caatinga"])
        with self.assertRaises(ValueError):
            validate([LAYERS[0]], [{**territory, "filter": {"field": "Bioma", "values": []}}])
        with self.assertRaises(ValueError):
            validate([LAYERS[0]], [{**territory, "filter": {"field": "Bioma", "value": "A", "values": ["B"]}}])

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
        self.assertEqual(str(amazon).replace("\\", "/"), "Brasil_Fogo_Colecao_5/Area_queimada_anual/Amazonia")
        combos = [("Brasil", LAYERS[0], TERRITORIES[0]), ("Brasil", LAYERS[0], TERRITORIES[1])]
        ensure_unique_paths(combos)
        with self.assertRaises(ValueError):
            ensure_unique_paths(combos + [combos[0]])

    def test_custom_names_are_safe_and_collision_is_not_silent(self):
        self.assertNotIn("/", safe_name("../../Mata Atlântica"))
        custom = {**LAYERS[0], "name": "Área de teste", "theme": "Água", "collection": "Coleção 1"}
        self.assertEqual(str(combination_path("Personalizado", custom, TERRITORIES[0])).replace("\\", "/"),
                         "Personalizado_Agua_Colecao_1/Area_de_teste/Amazonia")
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
            product = drive / "Entrega_2026/Brasil_Fogo_Colecao_5/Area_queimada_anual"
            self.assertTrue((product / "Amazonia/GIFs/mapa.gif").exists())
            self.assertTrue((product / "Caatinga/GIFs/mapa.gif").exists())

    def test_library_variants_and_deletion_are_safe(self):
        all_bands = ["burned_area_2020", "burned_area_2021", "burned_area_2022"]
        self.assertEqual(variant_for(all_bands, all_bands), "completo")
        self.assertEqual(variant_for(all_bands, all_bands[:2]), "2020-2021")
        self.assertTrue(variant_for(all_bands, all_bands[::2]).startswith("selecao_"))
        with tempfile.TemporaryDirectory() as tmp:
            stage, dest = Path(tmp) / "stage", Path(tmp) / "dest"
            (stage / "GIFs").mkdir(parents=True)
            file = stage / "GIFs/gif.gif"
            file.write_bytes(b"original")
            sig = fingerprint(LAYERS[0], TERRITORIES[0], all_bands)
            self.assertEqual(assess(dest, "completo", sig, {"gif"}), "nao_gerado")
            ingest(stage, dest, "completo", sig, all_bands, LAYERS[0], TERRITORIES[0])
            self.assertEqual(assess(dest, "completo", sig, {"gif"}), "disponivel")
            self.assertEqual(assess(dest, "completo", sig, {"gif", "finais"}), "parcial")
            (dest / "GIFs/gif.gif").unlink()
            self.assertEqual(assess(dest, "completo", sig, {"gif"}), "parcial")
            ingest(stage, dest, "completo", sig, all_bands, LAYERS[0], TERRITORIES[0])
            (dest / "GIFs/gif.gif").write_bytes(b"alterado")
            self.assertEqual(assess(dest, "completo", sig, {"gif"}), "verificar")
            with self.assertRaises(FileExistsError):
                ingest(stage, dest, "completo", sig, all_bands, LAYERS[0], TERRITORIES[0])
            self.assertEqual((dest / "GIFs/gif.gif").read_bytes(), b"alterado")

    def test_selected_drive_copy_keeps_partial_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, dest, stage = Path(tmp) / "local", Path(tmp) / "drive", Path(tmp) / "stage"
            (stage / "Frames/Mapa_completo").mkdir(parents=True)
            (stage / "Frames/Mapa_completo/2020.png").write_bytes(b"a")
            (stage / "Frames/Mapa_completo/2021.png").write_bytes(b"b")
            combo = root / "Brasil_Fogo_Colecao_5/produto/territorio"
            bands = ["a_2020", "a_2021"]
            sig = fingerprint(LAYERS[0], TERRITORIES[0], bands)
            files = ingest(stage, combo, "completo", sig, bands, LAYERS[0], TERRITORIES[0])
            sync_selected(files[:1], root, dest)
            target = dest / combo.relative_to(root)
            self.assertEqual(assess(target, "completo", sig, {"finais"}), "parcial")
            sync_selected(files[1:], root, dest)
            self.assertEqual(assess(target, "completo", sig, {"finais"}), "disponivel")

    def test_full_and_partial_versions_coexist_and_new_bands_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, destination = Path(tmp) / "stage", Path(tmp) / "library"
            (stage / "GIFs").mkdir(parents=True)
            (stage / "GIFs/mapa_2020-2021.gif").write_bytes(b"partial")
            partial = ["ano_2020", "ano_2021"]
            complete = partial + ["ano_2022"]
            partial_sig = fingerprint(LAYERS[0], TERRITORIES[0], partial)
            full_sig = fingerprint(LAYERS[0], TERRITORIES[0], complete)
            ingest(stage, destination, "2020-2021", partial_sig, partial, LAYERS[0], TERRITORIES[0])
            self.assertEqual(assess(destination, "completo", full_sig, {"gif"}), "nao_gerado")
            another = Path(tmp) / "stage2/GIFs"
            another.mkdir(parents=True)
            (another / "mapa.gif").write_bytes(b"full")
            ingest(another.parent, destination, "completo", full_sig, complete, LAYERS[0], TERRITORIES[0])
            self.assertEqual(assess(destination, "completo", full_sig, {"gif"}), "disponivel")
            next_sig = fingerprint(LAYERS[0], TERRITORIES[0], complete + ["ano_2023"])
            self.assertEqual(assess(destination, "completo", next_sig, {"gif"}), "verificar")


if __name__ == "__main__":
    unittest.main()
