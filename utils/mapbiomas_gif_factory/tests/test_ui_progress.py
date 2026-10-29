import importlib.util
import threading
import time
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch


def _select(widget, layer_ids=(), territory_ids=()):
    """Liga checkboxes por id, independente da posição na árvore de widgets."""
    for _, layer, check in widget.product_checks:
        if layer["id"] in layer_ids:
            check.value = True
    for _, territory, check in widget.territory_checks:
        if territory["id"] in territory_ids:
            check.value = True


def _fake_generate(record=None, bands=("ano_2023",)):
    def generate(layer, territory, root, kinds, after, **kwargs):
        if record is not None:
            record.append((layer["id"], territory["id"], layer.get("start_year"), layer.get("end_year")))
        kwargs["on_progress"]("discovered", list(bands))
        (root / "GIFs").mkdir(parents=True, exist_ok=True)
        file = root / "GIFs/result.gif"
        file.write_bytes(b"gif")
        return [file]
    return generate


@unittest.skipUnless(importlib.util.find_spec("ipywidgets") and importlib.util.find_spec("IPython"), "ipywidgets indisponível")
class UIProgressTests(unittest.TestCase):
    def run_ui(self, tmp, layers=(), territories=(), generate=None, discover_bands=("ano_2023",)):
        from gif_factory import ui as module
        patches = [patch.object(module, "display")]
        if generate is not None:
            patches.append(patch.object(module, "generate", generate))
        patches.append(patch.object(module, "discover", return_value=(list(discover_bands), list(discover_bands))))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return module.show_ui(list(layers), list(territories),
                              output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "temp"), project="test")

    def test_initiative_switch_clears_selections_and_keeps_custom_layers_separate(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, generate=_fake_generate(seen))
            _select(widget, {"annual_burned"}, {"bioma_amazonia"})
            widget.initiative.value = "Peru"
            self.assertTrue(all(not check.value for _, _, check in widget.product_checks))
            self.assertTrue(all(not check.value for _, _, check in widget.territory_checks))
            _select(widget, {"peru_fire2_annual"}, {"peru"})
            widget.start_batch()
            self.assertEqual([(a, b) for a, b, _, _ in seen], [("peru_fire2_annual", "peru")])

    def test_personalized_initiative_only_offers_user_layers_and_any_territory(self):
        seen = []
        custom = {"id": "minha_camada", "name": "Minha camada", "theme": "Meu tema",
                  "asset": "projects/x/y", "visualization": {"min": 0, "max": 1, "palette": ["fff", "000"]},
                  "legend": [{"label": "a", "color": "fff"}, {"label": "b", "color": "000"}]}
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, layers=[custom], territories=[{"id": "meu_territorio", "name": "Meu território", "asset": "projects/x/fc"}],
                                 generate=_fake_generate(seen))
            widget.initiative.value = "Personalizado"
            # Presets e territórios de qualquer país ficam acessíveis nesse espaço.
            for origin, _, check in widget.product_checks:
                if origin != "Personalizado":
                    check.value = True
            _select(widget, {"minha_camada"}, {"meu_territorio", "bioma_amazonia", "peru"})
            widget.start_batch()
            kinds = {(a, b) for a, b, _, _ in seen}
            self.assertEqual(kinds, {("minha_camada", "meu_territorio"),
                                     ("minha_camada", "bioma_amazonia"),
                                     ("minha_camada", "peru")})

    def test_preset_layer_accepts_unforeseen_territory(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, territories=[{"id": "territorio_novo", "name": "Território novo", "asset": "projects/x/fc"}],
                                 generate=_fake_generate(seen))
            _select(widget, {"annual_burned"}, {"territorio_novo"})
            widget.start_batch()
            self.assertEqual([(a, b) for a, b, _, _ in seen], [("annual_burned", "territorio_novo")])

    def test_theme_navigation_filters_products(self):
        def custom(pid, name, theme):
            return {"id": pid, "name": name, "theme": theme, "asset": f"projects/x/{pid}",
                    "visualization": {"min": 0, "max": 1, "palette": ["fff", "000"]},
                    "legend": [{"label": "a", "color": "fff"}, {"label": "b", "color": "000"}]}
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, layers=[custom("camada_a", "Camada A", "Meu tema"),
                                              custom("solo_teste", "Solo teste", "Solo")])
            widget.initiative.value = "Personalizado"
            self.assertEqual(list(widget.theme_nav.options), ["Meu tema", "Solo"])
            widget.theme_nav.value = "Solo"
            visible = [panel for panel in widget.children[2].children[2].children
                       if panel.layout.display != "none" and getattr(panel, "theme", None)]
            self.assertEqual([panel.theme for panel in visible], ["Solo"])
            widget.theme_nav.value = "Meu tema"
            visible = [panel for panel in widget.children[2].children[2].children
                       if panel.layout.display != "none" and getattr(panel, "theme", None)]
            self.assertEqual([panel.theme for panel in visible], ["Meu tema"])

    def test_existing_complete_output_skips_regeneration(self):
        calls = []
        def generate(layer, territory, root, kinds, after, **kwargs):
            calls.append(kinds)
            return _fake_generate()(layer, territory, root, kinds, after, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, generate=generate)
            _select(widget, {"annual_burned"}, {"bioma_amazonia"})
            widget.start_batch()
            widget.start_batch()
            self.assertEqual(len(calls), 1)
            self.assertIn("1 já disponível", widget.status_html.value)

    def test_selection_panels_are_visible_without_tabs_or_accordions(self):
        import ipywidgets as widgets
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, layers=[{"id": "x", "name": "X", "theme": "T", "asset": "a",
                                              "visualization": {"min": 0, "max": 1, "palette": ["fff", "000"]},
                                              "legend": [{"label": "a", "color": "fff"}]}])
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
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, layers=layers,
                                 territories=[{"id": "meu_territorio", "name": "Meu território", "asset": "projects/x/fc"}],
                                 generate=_fake_generate(seen))
            widget.initiative.value = "Personalizado"
            _select(widget, {"primeira", "segunda"}, {"meu_territorio"})
            widget.start_batch()
            self.assertEqual([(c, d) for a, b, c, d in seen], [(2020, 2021), (2023, 2024)])

    def test_startup_failure_is_visible(self):
        from gif_factory import ui as module
        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp)
            _select(widget, {"annual_burned"}, {"bioma_amazonia"})
            with patch.object(module, "Thread", side_effect=RuntimeError("não iniciou")):
                widget.start_batch()
            self.assertIn("Erro ao iniciar", widget.status_html.value)
            self.assertIn("não iniciou", widget.status_html.value)
            self.assertEqual(widget.sprout.layout.display, "none")

    def test_running_clock_and_completion(self):
        entered = threading.Event()
        release = threading.Event()

        def generate(layer, territory, root, kinds, after, **kwargs):
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

        with tempfile.TemporaryDirectory() as tmp:
            widget = self.run_ui(tmp, generate=generate, discover_bands=("ano_2023", "ano_2024"))
            _select(widget, {"annual_burned"}, {"bioma_amazonia"})
            widget.start_batch(wait=False)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(widget.sprout.layout.display, "")
                self.assertIn("Frame 1 de 2", widget.status_html.value)
                self.assertEqual(widget.download_progress.layout.display, "")
                time.sleep(1.2)
                self.assertIn("00:01", widget.status_html.value)
                widget.start_batch(wait=False)
                self.assertIn("Já existe um lote", widget.status_html.value)
            finally:
                release.set()
            deadline = time.monotonic() + 4
            while widget.sprout.layout.display != "none" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIn("LOTE CONCLUÍDO", widget.status_html.value)
            self.assertEqual(widget.overall.value, 1)
            self.assertEqual(widget.download_progress.layout.display, "none")
