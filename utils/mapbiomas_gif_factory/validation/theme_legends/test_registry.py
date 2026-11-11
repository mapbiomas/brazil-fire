import csv
import io
import unittest
from pathlib import Path

import yaml

import registry


class RegistryTest(unittest.TestCase):
    def test_source_snapshot_is_available_in_standalone_repository(self):
        self.assertEqual(registry.SOURCE, Path(registry.__file__).with_name("visualization.source.yaml"))
        self.assertTrue(registry.SOURCE.is_file())

    def test_snapshot_is_reproducible_and_complete(self):
        with registry.OUTPUT.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows, registry.build_rows())
        self.assertEqual([r["style_id"] for r in rows], list(registry.STYLE_IDS))
        self.assertEqual(len(rows), len(set(registry.STYLE_IDS)))

    def test_every_field_and_literal_source_survives(self):
        text = registry.SOURCE.read_bytes().decode("utf-8")
        original = yaml.load(text, Loader=registry.StrictLoader)["visualizations"]
        for row in registry.build_rows():
            with self.subTest(style=row["style_id"]):
                self.assertIn(row["source_yaml"], text)
                self.assertEqual(yaml.load(row["source_yaml"], Loader=registry.StrictLoader),
                                 {row["style_id"]: original[row["style_id"]]})
                self.assertEqual(row["status"], "provisorio")
                for field in ("country_link", "collection_link", "asset_link", "processor_link", "validation"):
                    self.assertEqual(row[field], "nao_comprovado")

    def test_csv_roundtrip_and_byte_reproducibility(self):
        rows = registry.build_rows()
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=registry.FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        self.assertEqual(stream.getvalue().encode("utf-8"), registry.OUTPUT.read_bytes())
        stream.seek(0)
        self.assertEqual(list(csv.DictReader(stream)), rows)

    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            yaml.load("style: {name: a, name: b}", Loader=registry.StrictLoader)


if __name__ == "__main__":
    unittest.main()
