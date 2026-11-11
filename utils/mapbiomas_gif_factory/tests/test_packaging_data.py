"""Package metadata must include the offline legend registry."""
import unittest
from pathlib import Path


class PackagingDataTests(unittest.TestCase):
    def test_legend_registry_is_packaged(self):
        root = Path(__file__).resolve().parents[1]
        metadata = (root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(metadata, r'gif_factory\s*=\s*\[[^\]]*"legends\.json"')
        self.assertRegex(metadata, r'gif_factory\s*=\s*\[[^\]]*"vera_fire_legends\.json"')
