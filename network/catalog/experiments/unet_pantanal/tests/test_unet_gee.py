"""Publicação no GEE: caminhos, propriedades e comando (sem GEE nem GCS)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config  # noqa: E402
from unet_gee import asset_id, classified_periods, gee_folders, manifest, map_path, time_props, upload_command  # noqa: E402

CFG = make_config(TRAINING_ID="1", SHORTNAME="pantanal_r1", REGION="Pantanal_r1")
VERSION = "training_001_pantanal_r1_sentinel2"
PREFIX = "projects/mapbiomas-fire-485203/assets/FIRE/CATALOG_01/MONITOR_01/UNET_01"


class TestGee(unittest.TestCase):
    def test_paths(self):
        folders, coll = gee_folders(CFG)
        self.assertEqual(folders[-1], f"{PREFIX}/pantanal")
        self.assertEqual(folders[0], "projects/mapbiomas-fire-485203/assets/FIRE")  # pais ainda inexistentes
        self.assertEqual(folders[1], "projects/mapbiomas-fire-485203/assets/FIRE/CATALOG_01")
        self.assertEqual(len(folders), 5)
        self.assertEqual(coll, f"{PREFIX}/pantanal/{VERSION}")
        self.assertEqual(asset_id(CFG, "2026_10"), f"{coll}/pantanal_r1_2026_10")
        self.assertTrue(map_path(CFG, "2026_10").endswith(
            f"/UNET_01/pantanal/pantanal_r1/{VERSION}/predict/pantanal_r1_2026_10_prob.tif"))

    def test_time(self):
        t = time_props("2026_10")
        self.assertEqual((t["year"], t["month"], t["periodicity"]), (2026, 10, "monthly"))
        self.assertEqual(t["time_end"] - t["time_start"], 31 * 86400 * 1000)
        self.assertEqual(time_props("2024")["month"], 0)

    def test_manifest(self):
        m = manifest(CFG, "2026_10", 0.63)
        self.assertEqual(m["name"], f"{PREFIX}/pantanal/{VERSION}/pantanal_r1_2026_10")
        self.assertEqual([b["id"] for b in m["bands"]], ["burned", "prob", "doy"])
        self.assertEqual([b["tilesetBandIndex"] for b in m["bands"]], [0, 1, 2])
        self.assertTrue(m["tilesets"][0]["sources"][0]["uris"][0].endswith("_2026_10_prob.tif"))
        self.assertEqual((m["pyramidingPolicy"], m["missingData"]), ("MODE", {"values": [-9999]}))
        self.assertEqual((m["startTime"], m["endTime"]), ("2026-10-01T00:00:00Z", "2026-11-01T00:00:00Z"))
        for k, v in (("threshold", 0.63), ("model_id", VERSION), ("region", "Pantanal_r1"), ("month", 10)):
            self.assertEqual(m["properties"][k], v)
        self.assertEqual(upload_command(CFG, "m.json"),
                         "earthengine --project mapbiomas-fire-485203 upload_manifest --force m.json")

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
