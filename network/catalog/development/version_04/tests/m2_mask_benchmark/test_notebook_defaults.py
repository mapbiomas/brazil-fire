"""Check the published notebook loads its benchmark from a pinned revision."""
import json
import ast
import re
import unittest
from pathlib import Path
from types import SimpleNamespace


class NotebookDefaultsTest(unittest.TestCase):
    def test_published_module_defaults(self):
        notebook = json.loads((Path(__file__).parent / "m2_mask_benchmark.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        exec("".join(notebook["cells"][1]["source"]), namespace)
        self.assertEqual(namespace["GITHUB_REPO"], "mapbiomas/brazil-fire")
        self.assertRegex(namespace["GITHUB_COMMIT"], re.compile(r"^[0-9a-f]{40}$"))
        self.assertEqual(namespace["LOCAL_MODULE_PATH"], "")

    def test_national_rejects_old_module_and_accepts_pinned_compatible_module(self):
        notebook = json.loads((Path(__file__).parent / "m2_mask_benchmark.ipynb").read_text(encoding="utf-8"))
        namespace = {"CONFIG": {"mode": "national"}, "LOCAL_MODULE_PATH": "",
                     "benchmark": SimpleNamespace(), "report": "old", "run_dir": "old"}
        with self.assertRaisesRegex(RuntimeError, "run_national"):
            exec("".join(notebook["cells"][3]["source"]), namespace)
        self.assertIsNone(namespace["report"])
        self.assertIsNone(namespace["run_dir"])
        def run(config):
            return {"status": "ok", "run_id": "test", "stages": []}, Path(".")
        namespace.update(benchmark=SimpleNamespace(run=run, run_national=run),
                         NATIONAL_CHECKPOINT_BUCKET="", NATIONAL_CHECKPOINT_PREFIX="test",
                         GCS_PROJECT="test", EE_PROJECT="")
        exec("".join(notebook["cells"][3]["source"]), namespace)
        self.assertEqual(namespace["report"]["status"], "ok")

    def test_gdal_setup_fails_early_with_install_instructions(self):
        notebook = json.loads((Path(__file__).parent / "m2_mask_benchmark.ipynb").read_text(encoding="utf-8"))
        tree = ast.parse("".join(notebook["cells"][2]["source"]))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ensure_gdal")
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {"shutil": SimpleNamespace(which=lambda name: None)}
        exec(compile(module, "<notebook>", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "apt-get install.*gdal-bin"):
            namespace["ensure_gdal"]()


if __name__ == "__main__":
    unittest.main()
