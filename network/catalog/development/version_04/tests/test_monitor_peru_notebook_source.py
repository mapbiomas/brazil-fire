"""Regression checks for the Peru monitor notebook's code source."""
import json
import unittest
from pathlib import Path


class MonitorPeruNotebookSourceTest(unittest.TestCase):
    def test_setup_clones_main_and_loads_version_04(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "monitor_01_peru.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]
                         if cell.get("cell_type") == "code")
        self.assertIn('"git", "clone", "--branch", "main"', code)
        self.assertIn("network/catalog/development/version_04/src/core", code)
        self.assertNotIn("fix/v04-region-only-classification", code)


if __name__ == "__main__":
    unittest.main()
