import tempfile
import unittest
from pathlib import Path
from gif_factory.engine import validate
from gif_factory.storage import copy_artifacts, list_artifacts
from gif_factory.presets import LAYERS, TERRITORIES


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


if __name__ == "__main__":
    unittest.main()
