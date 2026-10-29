import importlib.util
import threading
import time
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("ipywidgets") and importlib.util.find_spec("IPython"), "ipywidgets indisponível")
class UIProgressTests(unittest.TestCase):
    def test_fallback_respects_each_personalized_period(self):
        from gif_factory import ui as module
        base = module.LAYERS[0]
        layers = [{**base, "id": "primeira", "name": "Primeira", "start_year": 2020, "end_year": 2021},
                  {**base, "id": "segunda", "name": "Segunda", "start_year": 2023, "end_year": 2024}]
        seen = []
        def fake_generate(layer, territory, root, kinds, after, **kwargs):
            seen.append((layer.get("start_year"), layer.get("end_year")))
            kwargs["on_progress"]("discovered", ["ano_2023"])
            root.mkdir(parents=True, exist_ok=True)
            file = root / "result.gif"
            file.write_bytes(b"gif")
            return [file]
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate):
            widget = module.show_ui(layers, [], output_root=tmp, project="test")
            tabs = widget.children[1]
            tabs.children[0].children[0].children[0].value = False
            for checkbox in tabs.children[1].children[0].children:
                checkbox.value = True
            button = widget.children[3].children[0]
            widget.start_batch()
            deadline = time.monotonic() + 4
            while button.disabled and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(button.disabled)
            self.assertEqual(seen, [(2020, 2021), (2023, 2024)])

    def test_startup_failure_is_visible_and_button_restored(self):
        from gif_factory import ui as module
        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"):
            widget = module.show_ui([], [], output_root=tmp, project="test")
            button = widget.children[3].children[0]
            status = widget.children[4]
            with patch.object(module, "Thread", side_effect=RuntimeError("não iniciou")):
                widget.start_batch()
            self.assertFalse(button.disabled)
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
            root.mkdir(parents=True, exist_ok=True)
            result = root / "mapa.gif"
            result.write_bytes(b"gif")
            return [result]

        with tempfile.TemporaryDirectory() as tmp, patch.object(module, "display"), patch.object(module, "generate", fake_generate):
            widget = module.show_ui([], [], output_root=tmp, project="test")
            button, sprout = widget.children[3].children
            status, overall, download = widget.children[4:7]
            button.click()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(button.disabled)
                self.assertEqual(sprout.layout.display, "")
                self.assertIn("Frame 1 de 2", status.value)
                self.assertEqual(download.layout.display, "")
                time.sleep(1.2)
                self.assertIn("00:01", status.value)
            finally:
                release.set()
            deadline = time.monotonic() + 4
            while button.disabled and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(button.disabled)
            self.assertIn("LOTE CONCLUÍDO", status.value)
            self.assertEqual(overall.value, 1)
            self.assertEqual(sprout.layout.display, "none")
            self.assertEqual(download.layout.display, "none")
