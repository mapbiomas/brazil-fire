import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import build_legends  # noqa: E402


class LegendRegistryTests(unittest.TestCase):
    def test_registry_and_legend_files_are_valid(self):
        _rows, _legends, errors = build_legends.load()
        self.assertEqual(errors, [])

    def test_generated_files_are_up_to_date(self):
        rows, legends, errors = build_legends.load()
        self.assertEqual(errors, [])
        for name, text in (("LEGENDS.md", build_legends.markdown(rows, legends)),
                           ("preview.html", build_legends.preview(rows, legends))):
            with open(os.path.join(ROOT, name), encoding="utf-8", newline="") as handle:
                self.assertEqual(handle.read().replace("\r\n", "\n"), text,
                                 f"{name} is out of date: run python build_legends.py")

    def test_brazil_col5_class_counts(self):
        _rows, legends, _errors = build_legends.load()
        expected = {
            "brazil_col5_monthly_burned_area": 12,
            "brazil_col5_fire_scar_size": 8,
            "brazil_col5_fire_severity": 6,
            "brazil_col5_fire_frequency": 41,
            "brazil_col5_fire_return_interval": 40,
            "brazil_col5_last_fire_year": 41,
        }
        for legend_id, count in expected.items():
            self.assertEqual(len(legends[legend_id]), count, legend_id)


if __name__ == "__main__":
    unittest.main()
