import importlib.util
import threading
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("ipywidgets") and importlib.util.find_spec("IPython"), "ipywidgets indisponível")
class UIProgressTests(unittest.TestCase):
    def test_initiative_switch_clears_selections_and_prevents_cross_country(self):
        from gif_factory import ui as module
        seen = []
        def fake_generate(layer, territory, root, kinds, after, **kwargs):
            seen.append((layer["id"], territory["id"]))
            kwargs["on_progress"]("discovered", ["ano_2023"])
            (root / "GIFs").mkdir(parents=True, exist_ok=True)
            file = root / "GIFs/result.gif"
            file.write_bytes(b"gif")
            return [file]
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate), patch.object(module, "discover", return_value=(["ano_2023"], ["ano_2023"])):
            widget = module.show_ui([], [], output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "temp"), project="test")
            brasil = widget.children[2].children[0].children[1].children[1]
            brazil_territory = widget.children[3].children[1].children[1]
            brasil.value = brazil_territory.value = True
            widget.initiative.value = "Peru"
            self.assertFalse(brasil.value)
            self.assertFalse(brazil_territory.value)
            peru = widget.children[2].children[1].children[1].children[1]
            peru_territory = widget.children[3].children[2].children[1]
            peru.value = peru_territory.value = True
            brasil.value = brazil_territory.value = True  # hidden stale state, still excluded
            widget.start_batch()
            self.assertEqual(seen, [("peru_fire2_annual", "peru")])

    def test_existing_complete_output_skips_regeneration(self):
        from gif_factory import ui as module
        calls = []
        def fake_generate(layer, territory, root, kinds, after, **kwargs):
            calls.append(kinds)
            kwargs["on_progress"]("discovered", ["ano_2023"])
            (root / "GIFs").mkdir(parents=True, exist_ok=True)
            file = root / "GIFs/result.gif"
            file.write_bytes(b"gif")
            return [file]
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate), patch.object(module, "discover", return_value=(["ano_2023"], ["ano_2023"])):
            widget = module.show_ui([], [], output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "temp"), project="test")
            widget.children[2].children[0].children[1].children[1].value = True
            widget.children[3].children[1].children[1].value = True
            widget.start_batch()
            widget.start_batch()
            self.assertEqual(len(calls), 1)
            self.assertIn("1 já disponível", widget.children[7].value)

    def test_selection_panels_are_visible_without_tabs_or_accordions(self):
        from gif_factory import ui as module
        import ipywidgets as widgets
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"):
            widget = module.show_ui([module.LAYERS[0]], [], output_root=tmp, project="test")
            def walk(node):
                yield node
                for child in getattr(node, "children", ()):
                    yield from walk(child)
            nodes = list(walk(widget))
            self.assertFalse(any(isinstance(node, (widgets.Tab, widgets.Accordion)) for node in nodes))
            self.assertTrue(any(isinstance(node, widgets.Checkbox) and "Área queimada" in node.description for node in nodes))
            self.assertTrue(any(isinstance(node, widgets.Checkbox) and "Amazônia" in node.description for node in nodes))
            self.assertTrue(any(isinstance(node, widgets.Checkbox) and node.description == "GIFs" for node in nodes))

    def test_fallback_respects_each_personalized_period(self):
        from gif_factory import ui as module
        base = module.LAYERS[0]
        layers = [{**base, "id": "primeira", "name": "Primeira", "start_year": 2020, "end_year": 2021},
                  {**base, "id": "segunda", "name": "Segunda", "start_year": 2023, "end_year": 2024}]
        seen = []
        def fake_generate(layer, territory, root, kinds, after, **kwargs):
            seen.append((layer.get("start_year"), layer.get("end_year")))
            kwargs["on_progress"]("discovered", ["ano_2023"])
            (root / "GIFs").mkdir(parents=True, exist_ok=True)
            file = root / "GIFs/result.gif"
            file.write_bytes(b"gif")
            return [file]
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate), patch.object(module, "discover", return_value=(["ano_2023"], ["ano_2023"])):
            widget = module.show_ui(layers, [], output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "temp"), project="test")
            catalog = widget.children[2]
            for checkbox in catalog.children[2].children[1].children[1:]:
                checkbox.value = True
            widget.children[3].children[1].children[1].value = True
            widget.start_batch()
            self.assertEqual(seen, [(2020, 2021), (2023, 2024)])

    def test_startup_failure_is_visible(self):
        from gif_factory import ui as module
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"):
            widget = module.show_ui([], [], output_root=tmp, project="test")
            status = widget.children[7]
            widget.children[2].children[0].children[1].children[1].value = True
            widget.children[3].children[1].children[1].value = True
            with patch.object(module, "Thread", side_effect=RuntimeError("não iniciou")):
                widget.start_batch()
            self.assertIn("Erro ao iniciar", status.value)
            self.assertIn("não iniciou", status.value)

    def test_running_clock_and_completion(self):
        from gif_factory import ui as module
        entered = threading.Event()
        release = threading.Event()

        def fake_generate(layer, territory, root, kinds, after, **kwargs):
            event = kwargs["on_progress"]
            event("discovered", ["ano_2023", "ano_2024"])
            event("frame_start", 1, 2, "ano_2023")
            entered.set()
            release.wait(4)
            event("frame_done", 1, 2, "ano_2023", 1)
            event("stage", "Gerando GIF")
            (root / "GIFs").mkdir(parents=True, exist_ok=True)
            result = root / "GIFs/mapa.gif"
            result.write_bytes(b"gif")
            return [result]

        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate), patch.object(module, "discover", return_value=(["ano_2023", "ano_2024"], ["ano_2023", "ano_2024"])):
            widget = module.show_ui([], [], output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "temp"), project="test")
            sprout = widget.children[6]
            status, overall, download = widget.children[7:10]
            widget.children[2].children[0].children[1].children[1].value = True
            widget.children[3].children[1].children[1].value = True
            widget.start_batch(wait=False)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(sprout.layout.display, "")
                self.assertIn("Frame 1 de 2", status.value)
                self.assertEqual(download.layout.display, "")
                time.sleep(1.2)
                self.assertIn("00:01", status.value)
                widget.start_batch(wait=False)
                self.assertIn("Já existe um lote", status.value)
            finally:
                release.set()
            deadline = time.monotonic() + 4
            while sprout.layout.display != "none" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIn("LOTE CONCLUÍDO", status.value)
            self.assertEqual(overall.value, 1)
            self.assertEqual(sprout.layout.display, "none")
            self.assertEqual(download.layout.display, "none")
