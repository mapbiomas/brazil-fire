"""Check the published notebook loads its benchmark from a pinned revision."""
import json
import re
import unittest
from pathlib import Path


class NotebookDefaultsTest(unittest.TestCase):
    def test_published_module_defaults(self):
        notebook = json.loads((Path(__file__).parent / "m2_mask_benchmark.ipynb").read_text(encoding="utf-8"))
        namespace = {}
        exec("".join(notebook["cells"][1]["source"]), namespace)
        self.assertEqual(namespace["GITHUB_REPO"], "mapbiomas/brazil-fire")
        self.assertRegex(namespace["GITHUB_COMMIT"], re.compile(r"^[0-9a-f]{40}$"))
        self.assertEqual(namespace["LOCAL_MODULE_PATH"], "")


if __name__ == "__main__":
    unittest.main()
