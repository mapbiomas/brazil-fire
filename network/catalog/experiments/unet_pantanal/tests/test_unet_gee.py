"""Publicação no GEE: caminhos, propriedades e comando (sem GEE nem GCS)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config  # noqa: E402
from unet_gee import asset_id, classified_periods, gee_folders, map_path, time_props, upload_command  # noqa: E402

CFG = make_config(TRAINING_ID="1", SHORTNAME="pantanal_r1", REGION="Pantanal_r1")
VERSION = "training_001_pantanal_r1_sentinel2"
PREFIX = "projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/MONITOR_01/UNET_01"


class TestGee(unittest.TestCase):
    def test_paths(self):
        folders, coll = gee_folders(CFG)
        self.assertEqual(folders[-1], f"{PREFIX}/pantanal")
        self.assertEqual(coll, f"{PREFIX}/pantanal/{VERSION}")
        self.assertEqual(asset_id(CFG, "2026_10"), f"{coll}/pantanal_r1_2026_10")
        self.assertTrue(map_path(CFG, "2026_10").endswith(
            f"/UNET_01/pantanal/pantanal_r1/{VERSION}/predict/pantanal_r1_2026_10_prob.tif"))

    def test_time(self):
        t = time_props("2026_10")
        self.assertEqual((t["year"], t["month"], t["periodicity"]), (2026, 10, "monthly"))
        self.assertEqual(t["time_end"] - t["time_start"], 31 * 86400 * 1000)
        self.assertEqual(time_props("2024")["month"], 0)

    def test_command(self):
        cmd = upload_command(CFG, "2026_10")
        self.assertTrue(cmd.startswith("earthengine --project mapbiomas-brazil upload image "))
        for part in (f"--asset_id={PREFIX}/pantanal/{VERSION}/pantanal_r1_2026_10", "--pyramiding_policy=mode",
                     f"--property model_id={VERSION}", "--property region=Pantanal_r1", "--property month=10"):
            self.assertIn(part, cmd)
        self.assertTrue(cmd.endswith("_2026_10_prob.tif"))

    def test_classified_periods(self):
        names = ["pantanal_r1_2026_10_prob.tif", "pantanal_r1_2026_09_prob.tif", "outra_2026_10_prob.tif", "x.json"]
        self.assertEqual(classified_periods(CFG, ls=lambda p: names), ["2026_09", "2026_10"])

    def test_publish_replaces_and_submits(self):
        calls = []
        fake = types.SimpleNamespace(data=types.SimpleNamespace(
            createAsset=lambda *a: calls.append(("create", a[1])),
            getAsset=lambda aid: calls.append(("get", aid)),
            deleteAsset=lambda aid: calls.append(("delete", aid)),
            listAssets=lambda q: {}))
        import unet_gee
        sys.modules["ee"], sleep = fake, unet_gee.time.sleep
        unet_gee.time.sleep = lambda s: None
        try:
            cmds = []
            done = unet_gee.publish(CFG, ["2026_10"], logger=lambda m: None,
                                    run=lambda c: cmds.append(c) or 0)
        finally:
            del sys.modules["ee"]
            unet_gee.time.sleep = sleep
        self.assertEqual(done, {"2026_10": True})
        self.assertIn(("delete", asset_id(CFG, "2026_10")), calls)
        self.assertEqual(len(cmds), 1)


if __name__ == "__main__":
    unittest.main()
