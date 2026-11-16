"""The correction state uses the same label across supported UI languages."""
import importlib.util
import unittest
from pathlib import Path


class UpdateLabelTest(unittest.TestCase):
    def test_update_label_is_consistent(self):
        path = Path(__file__).resolve().parents[1] / "src" / "core" / "M_lang.py"
        spec = importlib.util.spec_from_file_location("m_lang_label_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.L.M1_UPDATE_LABEL, "UPDATE")
        for name in ("STRINGS_ES", "STRINGS_PT", "STRINGS_FR", "STRINGS_ID"):
            with self.subTest(locale=name):
                self.assertEqual(getattr(module, name)["M1_UPDATE_LABEL"], "UPDATE")


if __name__ == "__main__":
    unittest.main()
