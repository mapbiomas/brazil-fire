"""Interface em 4 passos, execução do lote e territórios (sem Earth Engine)."""
import importlib.util
import io
import json
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

FIRE = "brasil_fire_col5__annual_burned"
LULC = "brasil_lulc_col11__lulc_nivel1"
TERRITORY = {"id": "busca_teste", "name": "Barreiras (BA)", "asset": "projects/x/fc",
             "filter": {"field": "FEATURE_ID", "values": [1]}}
DRAWN = {"id": "area_teste", "name": "Área desenhada",
         "geometry": {"type": "Polygon", "coordinates": [[[-45, -12], [-44, -12], [-44, -11], [-45, -11], [-45, -12]]]}}


def fake_generate(record=None, bands=("ano_2023",)):
    def generate(layer, territory, root, kinds, after, **kwargs):
        if record is not None:
            record.append((layer["id"], territory["id"], layer.get("start_year"), layer.get("end_year"), set(kinds)))
        kwargs["on_progress"]("discovered", list(bands))
        (root / "GIFs").mkdir(parents=True, exist_ok=True)
        file = root / "GIFs/result.gif"
        file.write_bytes(b"GIF89a")
        return [file]
    return generate


@unittest.skipUnless(importlib.util.find_spec("ipywidgets"), "ipywidgets indisponível")
class AppTests(unittest.TestCase):
    def make(self, tmp, generate=None, bands=("ano_2023",), **kwargs):
        from gif_factory import app, runner
        patches = [patch.object(runner, "discover", return_value=(list(bands), list(bands)))]
        if generate is not None:
            patches.append(patch.object(runner, "generate", generate))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        kwargs.setdefault("library", str(Path(tmp) / "meus_itens.json"))
        return app.FactoryApp("test", output_root=str(Path(tmp) / "out"), temp_root=str(Path(tmp) / "tmp"),
                              show=False, **kwargs)

    def test_catalog_search_and_selection_feed_the_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            self.assertEqual(a.initiative.value, "Brasil")
            a.map_search.value = "queimada anual"
            names = [b.description for b in a.product_list.children]
            self.assertTrue(names and all("queimada" in n.lower() for n in names))
            a.product_buttons[FIRE].value = True
            self.assertEqual(a.selected_layers, [FIRE])
            self.assertIn("Área queimada anual", a.summary.value)
            self.assertIn("Pixel queimado", a.product_detail.value)
            a.product_buttons[FIRE].value = False
            self.assertEqual(a.selected_layers, [])
            self.assertIn("Clique em um mapa", a.product_detail.value)

    def test_accent_insensitive_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            a.map_search.value = "frequencia fogo"
            self.assertTrue(any("Frequência" in b.description for b in a.product_list.children))

    def test_beta_maps_are_hidden_until_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            beta = [layer for layer in a.catalog if layer["status"] == "beta" and layer["initiative"] == "Brasil"]
            self.assertTrue(beta)
            visible = {b.description for b in a.product_list.children}
            self.assertFalse(any(layer["name"] in d for layer in beta for d in visible if "beta" in d))
            a.show_beta.value = True
            self.assertTrue(any("beta" in b.description for b in a.product_list.children))

    def test_combinations_and_period_are_clamped_per_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, territories=[TERRITORY, DRAWN])
            a.product_buttons[FIRE].value = True
            a.product_buttons[LULC].value = True
            a.year_ranges = {FIRE: (1985, 2025), LULC: (2000, 2010)}
            a._apply_years()
            self.assertEqual((a.period.min, a.period.max), (1985, 2025))
            a.all_years.value = False
            a.period.value = (2008, 2020)
            combos = a.combinations()
            self.assertEqual(len(combos), 4)
            periods = {(layer["id"], layer["start_year"], layer["end_year"]) for _, layer, _ in combos}
            self.assertEqual(periods, {(FIRE, 2008, 2020), (LULC, 2008, 2010)})
            self.assertIn("4 combinação", a.summary.value)

    def test_generate_runs_each_combination_and_shows_results(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=fake_generate(seen), territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.kind_checks["colagem"].value = True
            a.save_drive.value = False
            a.start()
            self.assertEqual([(s[0], s[1]) for s in seen], [(FIRE, TERRITORY["id"])])
            self.assertEqual(seen[0][4], {"gif", "colagem"})
            self.assertIn("LOTE CONCLUÍDO", a.runner.status.value)
            card = a.gallery.card(0)
            self.assertEqual(card.state, "pronto")
            self.assertTrue(card.files and all(f.suffix == ".gif" for f in card.files))
            self.assertIn("1 de 1 prontos", a.gallery.counts.value)

    def test_existing_result_is_not_regenerated(self):
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=fake_generate(seen), territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.save_drive.value = False
            a.start()
            a.start()
            self.assertEqual(len(seen), 1)
            self.assertIn("1 já disponível", a.runner.status.value)

    def test_missing_selection_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            a.start()
            self.assertIn("Escolha ao menos um mapa", a.runner.status.value)

    def test_startup_failure_is_visible(self):
        from gif_factory import runner
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            with patch.object(runner, "Thread", side_effect=RuntimeError("não iniciou")):
                a.start()
            self.assertIn("Erro ao iniciar", a.runner.status.value)
            self.assertEqual(a.runner.sprout.layout.display, "none")

    def test_running_clock_and_single_batch_at_a_time(self):
        entered, release = threading.Event(), threading.Event()

        def generate(layer, territory, root, kinds, after, **kwargs):
            event = kwargs["on_progress"]
            event("discovered", ["ano_2023", "ano_2024"])
            event("frame_start", 1, 2, "ano_2023")
            entered.set()
            release.wait(4)
            event("frame_done", 1, 2, "ano_2023", 1)
            (root / "GIFs").mkdir(parents=True, exist_ok=True)
            result = root / "GIFs/mapa.gif"
            result.write_bytes(b"GIF89a")
            return [result]

        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=generate, bands=("ano_2023", "ano_2024"), territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.save_drive.value = False
            a.start(wait=False)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(a.runner.sprout.layout.display, "")
                self.assertIn("Frame 1 de 2", a.runner.status.value)
                time.sleep(1.2)
                self.assertIn("00:01", a.runner.status.value)
                a.start(wait=False)
                self.assertIn("Já existe um lote", a.runner.status.value)
            finally:
                release.set()
            deadline = time.monotonic() + 4
            while a.runner.sprout.layout.display != "none" and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIn("LOTE CONCLUÍDO", a.runner.status.value)

    def test_custom_maps_appear_under_my_maps(self):
        custom = {"id": "meu_mapa", "name": "Meu mapa beta", "theme": "Fogo", "asset": "projects/x/img",
                  "visualization": {"min": 0, "max": 1, "palette": ["ffffff", "000000"]},
                  "legend": [{"label": "a", "color": "ffffff"}, {"label": "b", "color": "000000"}]}
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, layers=[custom])
            self.assertIn("Meus mapas", a.initiative.options)
            a.initiative.value = "Meus mapas"
            self.assertEqual([b.description for b in a.product_list.children], ["Meu mapa beta"])

    def test_old_show_ui_entry_point_still_works(self):
        from gif_factory import ui
        with patch("gif_factory.app.display"), tempfile.TemporaryDirectory() as tmp:
            with patch("gif_factory.user_library.default_path", return_value=Path(tmp) / "lib.json"):
                a = ui.show_ui([], [TERRITORY], output_root=tmp, project="test", temp_root=tmp)
            self.assertEqual([t["id"] for t in a.selected_territories], [TERRITORY["id"]])

    def test_combinations_run_in_parallel_and_fill_the_gallery(self):
        both = threading.Barrier(2, timeout=4)
        seen = []

        def generate(layer, territory, root, kinds, after, **kwargs):
            both.wait()  # só passa se as duas combinações estiverem rodando ao mesmo tempo
            seen.append(territory["id"])
            return fake_generate()(layer, territory, root, kinds, after, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=generate, territories=[TERRITORY, DRAWN])
            a.product_buttons[FIRE].value = True
            a.save_drive.value = False
            a.workers.value = 2
            a.start()
            self.assertEqual(sorted(seen), sorted([TERRITORY["id"], DRAWN["id"]]))
            self.assertEqual([a.gallery.card(i).state for i in range(2)], ["pronto", "pronto"])
            self.assertIn("2 de 2 prontos", a.gallery.counts.value)
            self.assertIn("Volume:", a.runner.status.value)

    def test_failure_shows_on_its_card(self):
        def generate(layer, territory, root, kinds, after, **kwargs):
            raise RuntimeError("sem cota")

        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=generate, territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.save_drive.value = False
            a.start()
            card = a.gallery.card(0)
            self.assertEqual(card.state, "falha")
            self.assertIn("sem cota", card.status.value)
            self.assertTrue(card.open.disabled)

    def test_library_lists_what_was_generated(self):
        from gif_factory.gallery import scan_library
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, generate=fake_generate(), territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.save_drive.value = False
            a.start()
            entries = scan_library(Path(tmp) / "out")
            self.assertEqual([(e["title"], e["subtitle"]) for e in entries], [("Área queimada anual", "Barreiras (BA)")])
            a.library_source.value = "local"
            a._library_entries = entries
            a._fill_library()
            card = next(iter(a.library_gallery.cards.values()))
            self.assertEqual(card.state, "pronto")
            card.open.value = True
            self.assertEqual(a.library_gallery.detail_box.layout.display, "")
            a.library_search.value = "outro lugar"
            self.assertFalse(a.library_gallery.cards)

    def test_estimate_is_shown_before_generating(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp, territories=[TERRITORY])
            a.product_buttons[FIRE].value = True
            a.year_ranges = {FIRE: (1985, 2024)}
            a.stepper.value = "4 · Gerar"
            self.assertIn("para 40 quadro(s)", a.estimate_box.value)
            self.assertIn("MB", a.estimate_box.value)
            self.assertIn("~", a.summary.value)

    def test_labels_share_one_width_and_next_names_the_step(self):
        from gif_factory.app import LABEL_WIDTH
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            self.assertEqual(a.initiative.style.description_width, LABEL_WIDTH)
            self.assertEqual(a.asset_field.style.description_width, LABEL_WIDTH)
            self.assertEqual(a.next.description, "Avançar: Território")
            a.stepper.value = "5 · Biblioteca"
            self.assertEqual(a.next.layout.display, "none")


class EstimateTests(unittest.TestCase):
    def test_time_and_size_texts(self):
        from gif_factory import estimate as E
        self.assertEqual(E.human_time(30), "menos de 1 minuto")
        self.assertEqual(E.human_time(3 * 3600 + 5 * 60), "3 h 05 min")
        self.assertEqual(E.human_time(50 * 3600), "2 dia(s) e 2 h")
        self.assertEqual(E.human_size(2048), "2.0 GB")
        self.assertEqual(E.mood(60)[2], 0)
        self.assertEqual(E.mood(2 * 3600)[2], 1)
        self.assertEqual(E.mood(3 * 86400)[2], 2)

    def test_batch_uses_workers_and_session_pace(self):
        from gif_factory import estimate as E
        one = E.batch([40, 40], {"gif"}, workers=1)
        two = E.batch([40, 40], {"gif"}, workers=2)
        self.assertAlmostEqual(two["seconds"] * 2, one["seconds"])
        self.assertEqual(one["frames"], 80)
        self.assertAlmostEqual(E.batch([10], {"gif"}, to_drive=True)["saved_mb"],
                               2 * E.batch([10], {"gif"})["saved_mb"])
        pace = E.Calibration()
        pace.add(20, 2)
        self.assertIsNone(pace.per_frame())  # poucos quadros ainda
        pace.add(40, 2)
        self.assertEqual(pace.per_frame(), 15)
        self.assertTrue(E.batch([10], {"gif"}, per_frame=pace.per_frame())["calibrated"])

    def test_workers_follow_memory(self):
        from gif_factory.environment import detect, suggested_workers
        self.assertEqual(suggested_workers(2, 12.7), 2)   # Colab gratuito
        self.assertEqual(suggested_workers(8, 51), 4)     # Colab com mais memória
        self.assertEqual(suggested_workers(2, 4), 1)
        self.assertEqual(suggested_workers(16, None), 1)
        self.assertGreaterEqual(detect()["workers"], 1)


@unittest.skipUnless(importlib.util.find_spec("ipywidgets"), "ipywidgets indisponível")
class AnalystModeTests(unittest.TestCase):
    make = AppTests.make

    def fill_custom_map(self, a):
        a.new_map.value = True
        a.beta_asset.value = "projects/meu/assets/fogo_beta"
        a.beta_name.value = "Fogo beta do Cerrado"
        a.beta_mode.value = "Minhas cores"
        a.beta_min.value, a.beta_max.value = 0, 1
        a.beta_colors.value = "ffffff, 990000"
        a.beta_labels.value = "Não queimado, Queimado"

    def test_new_map_is_saved_selected_and_comes_back_next_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            self.fill_custom_map(a)
            self.assertTrue(a.beta_style.options)  # estilos do catálogo carregados ao abrir o painel
            a._save_beta()
            self.assertIn("salvo em Meus mapas", a.beta_msg.value)
            layer_id = a.selected_layers[-1]
            self.assertEqual(a.initiative.value, "Meus mapas")
            self.assertEqual(a.by_id[layer_id]["visualization"]["palette"], ["ffffff", "990000"])
            saved = json.loads((Path(tmp) / "meus_itens.json").read_text(encoding="utf-8"))
            self.assertEqual([m["name"] for m in saved["maps"]], ["Fogo beta do Cerrado"])
            b = self.make(tmp)
            self.assertIn("Meus mapas", b.initiative.options)
            self.assertIn(layer_id, b.by_id)
            b.initiative.value = "Meus mapas"
            b._show_proposal(layer_id)
            self.assertIn("fogo_beta_do_cerrado", b.proposal.value)
            b._remove_custom_layer(layer_id)
            self.assertNotIn("Meus mapas", b.initiative.options)
            self.assertEqual(json.loads((Path(tmp) / "meus_itens.json").read_text(encoding="utf-8"))["maps"], [])

    def test_catalog_style_can_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            a.new_map.value = True
            a.beta_asset.value = "projects/meu/assets/lulc_beta"
            a.beta_name.value = "LULC beta"
            a.beta_style.value = "lulc_nivel1"
            self.assertIn("Formação Florestal", a.beta_style_legend.value)
            a._save_beta()
            layer = a.by_id[a.selected_layers[-1]]
            self.assertEqual(layer["style_key"], "lulc_nivel1")
            self.assertTrue(layer["legend"])

    def test_invalid_colors_are_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            self.fill_custom_map(a)
            a.beta_colors.value = "branco, 990000"
            a._save_beta()
            self.assertIn("Cores inválidas", a.beta_msg.value)
            a.beta_colors.value = "ffffff, 990000"
            a.beta_labels.value = "só um"
            a._save_beta()
            self.assertIn("um rótulo por cor", a.beta_msg.value)

    def test_drawn_territories_are_kept_for_next_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make(tmp)
            a._keep(DRAWN)
            self.assertEqual([t["id"] for t in a.selected_territories], [DRAWN["id"]])
            b = self.make(tmp)
            b._territories_started = True
            b.base.value = "meus"
            self.assertEqual([r["name"] for r in b._found], [DRAWN["name"]])
            b._marked = list(b._found)
            b._add_found()
            self.assertEqual([t["id"] for t in b.selected_territories], [DRAWN["id"]])
            b._marked = list(b._found)
            b._forget_saved()
            self.assertEqual(b._found, [])


class AnalystHelperTests(unittest.TestCase):
    def test_snippet_is_valid_yaml(self):
        import yaml
        from gif_factory.analyst import build_layer, catalog_snippet
        layer = build_layer("projects/x/img", "Minha frequência", "fire", "beta", minimum=0, maximum=10,
                            colors="ffffff ffcc00 990000")
        text = catalog_snippet(layer)
        products = yaml.safe_load(text.split("# --- e em config/visualization.yaml")[0])
        self.assertEqual(products["minha_frequencia"]["visualization"], "minha_frequencia_beta")
        self.assertEqual(layer["legend"][0], {"label": "0", "color": "ffffff"})

    def test_min_must_be_below_max(self):
        from gif_factory.analyst import build_layer
        with self.assertRaises(ValueError):
            build_layer("projects/x/img", "X", "fire", "beta", minimum=5, maximum=5, colors="ffffff,000000")


class TerritoryTests(unittest.TestCase):
    def test_display_name_and_plain_text(self):
        from gif_factory.territories import display_name, plain
        self.assertEqual(display_name("[Porto Velho][RO][1100205]"), "Porto Velho (RO)")
        self.assertEqual(display_name("[Amazônia]"), "Amazônia")
        self.assertEqual(display_name("[Chachapoyas][Chachapoyas][Amazonas][010101]"), "Chachapoyas (Amazonas)")
        self.assertEqual(plain("  São Félix "), "sao felix")

    def test_search_prefers_names_that_start_with_the_text(self):
        from gif_factory import territories as T
        source = T.sources_for("brasil")[4]
        rows = [(n, i, T.plain(n)) for n, i in [("Santa Maria das Barreiras (PA)", 3), ("Barreiras (BA)", 1),
                                                ("Barreiras do Piauí (PI)", 2)]]
        with patch.dict(T._INDEX, {(source["asset"], "null"): sorted(rows, key=lambda r: r[2])}):
            found = T.search(None, source, "barreiras")
        self.assertEqual([r["name"] for r in found][:2], ["Barreiras (BA)", "Barreiras do Piauí (PI)"])
        self.assertEqual(found[-1]["name"], "Santa Maria das Barreiras (PA)")

    def test_records_become_one_or_many_territories(self):
        from gif_factory.territories import from_records
        records = [{"name": "A", "value": 1, "asset": "x", "field": "FEATURE_ID", "kind": "Município"},
                   {"name": "B", "value": 2, "asset": "x", "field": "FEATURE_ID", "kind": "Município"}]
        self.assertEqual(len(from_records(records)), 2)
        combined = from_records(records, combine=True)
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["filter"], {"field": "FEATURE_ID", "values": [1, 2]})
        self.assertEqual(combined[0]["name"], "A + B")

    def test_drawings_and_geojson_upload(self):
        from gif_factory.territories import from_geometry, merge_drawings, read_upload
        square = DRAWN["geometry"]
        merged = merge_drawings([{"geometry": square}, {"geometry": square}])
        self.assertEqual(merged["type"], "MultiPolygon")
        with self.assertRaises(ValueError):
            merge_drawings([])
        data = json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": square}]})
        geometry = read_upload("area.geojson", data.encode("utf-8"))
        territory = from_geometry(geometry, "Minha área")
        self.assertEqual(territory["name"], "Minha área")
        self.assertTrue(territory["id"].startswith("area_"))

    @unittest.skipUnless(importlib.util.find_spec("geopandas"), "geopandas indisponível")
    def test_zipped_shapefile_upload(self):
        import geopandas as gpd
        from shapely.geometry import shape
        from gif_factory.territories import read_upload
        with tempfile.TemporaryDirectory() as tmp:
            gpd.GeoDataFrame({"n": [1]}, geometry=[shape(DRAWN["geometry"])], crs=4326).to_file(Path(tmp) / "a.shp")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                for f in Path(tmp).iterdir():
                    archive.write(f, f.name)
            geometry = read_upload("area.zip", buffer.getvalue())
        self.assertEqual(geometry["type"], "Polygon")

    def test_asset_with_optional_filter(self):
        from gif_factory.territories import from_asset
        item = from_asset("projects/p/assets/areas", "Duas áreas", "codigo", "10, 20")
        self.assertEqual(item["filter"], {"field": "codigo", "values": [10, 20]})
        self.assertEqual(from_asset("projects/p/assets/areas")["name"], "areas")
        with self.assertRaises(ValueError):
            from_asset("  ")

    def test_engine_accepts_drawn_territories(self):
        from gif_factory.engine import _territory_entries, validate
        validate([], [DRAWN])
        entry = _territory_entries(DRAWN)["biomes"][DRAWN["id"]]
        self.assertEqual(entry["geometry"], DRAWN["geometry"])

    def test_every_initiative_has_territory_sources(self):
        from gif_factory.territories import sources_for
        for code in ("brasil", "peru", "paraguay", "drc"):
            self.assertTrue(sources_for(code), code)


if __name__ == "__main__":
    unittest.main()
