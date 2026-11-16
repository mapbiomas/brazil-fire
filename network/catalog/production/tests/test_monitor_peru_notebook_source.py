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
        self.assertIn("network/catalog/production/src/core", code)
        self.assertNotIn("fix/v04-region-only-classification", code)
        self.assertIn("gee_project='mapbiomas-fire-485203'", code)

    def test_all_official_notebooks_use_execution_project(self):
        notebooks = Path(__file__).resolve().parents[1] / "notebooks"
        paths = sorted(notebooks.glob("*.ipynb"))
        configured = 0
        for path in paths:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            code = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]
                             if cell.get("cell_type") == "code")
            if "gee_project=" not in code:
                continue
            configured += 1
            with self.subTest(notebook=path.name):
                self.assertIn("gee_project='mapbiomas-fire-485203'", code)
                self.assertNotIn("gee_project=f'mapbiomas-{COUNTRY_SELECTED}'", code)
        self.assertGreater(configured, 1)


if __name__ == "__main__":
    unittest.main()
