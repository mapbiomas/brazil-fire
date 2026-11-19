"""Idiomas: textos vêm dos CSVs; célula vazia usa o original."""
import unittest
from unittest.mock import patch


class I18nTests(unittest.TestCase):
    def tearDown(self):
        from gif_factory import i18n
        i18n.set_language(ui="pt", maps="pt")

    def test_language_names_and_codes(self):
        from gif_factory.i18n import code_for
        self.assertEqual(code_for("English"), "en")
        self.assertEqual(code_for("中文 (Mandarim)"), "zh")
        self.assertEqual(code_for("Bahasa Indonesia"), "id")
        self.assertEqual(code_for("??"), "pt")

    def test_translation_falls_back_to_original(self):
        from gif_factory import i18n
        table = {"Gerar": {"en": "Generate"}, "{n} combinação(ões)": {"en": "{n} combination(s)"}}
        with patch.object(i18n, "interface_table", return_value=table):
            i18n.set_language(ui="en")
            self.assertEqual(i18n._("Gerar"), "Generate")
            self.assertEqual(i18n._("{n} combinação(ões)", n=3), "3 combination(s)")
            self.assertEqual(i18n._("Voltar"), "Voltar")

    def test_map_texts_and_legends(self):
        from gif_factory import i18n
        with patch.object(i18n, "maps_table", return_value={"Fogo": {"en": "Fire"}, "Área queimada": {"en": "Burned area"}}):
            viz = i18n.translate_visualization({"label": "Área queimada", "discrete_labels": ["Fogo", ""]}, "en")
            self.assertEqual(viz, {"label": "Burned area", "discrete_labels": ["Fire", ""]})

    def test_csvs_list_every_interface_text(self):
        from gif_factory.i18n import _read, INTERFACE_CSV, MAPS_CSV, interface_strings
        rows = _read(INTERFACE_CSV, "pt")
        missing = [text for text, _where in interface_strings() if text not in rows]
        self.assertEqual(missing, [], "rode: python -m gif_factory.i18n")
        self.assertTrue(_read(MAPS_CSV, "original"))
