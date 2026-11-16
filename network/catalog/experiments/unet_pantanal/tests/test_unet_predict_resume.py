"""Retomada da classificação: faixas, estado salvo no GCS (simulado) e junção das faixas."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config  # noqa: E402

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    import unet_predict
    HAVE = True
except ImportError:  # sem numpy/rasterio/TF no ambiente
    HAVE = False


@unittest.skipUnless(HAVE, "numpy/rasterio ausentes")
class TestStrips(unittest.TestCase):
    def test_strip_plan_covers_the_window(self):
        win = Window(10, 100, 2000, 5000)
        plan = unet_predict.strip_plan(win, 512, 4)  # faixas de 2048 linhas
        self.assertEqual(plan, [(0, 100, 2048), (1, 2148, 2048), (2, 4196, 904)])
        self.assertEqual(sum(h for _, _, h in plan), 5000)

    def test_assemble_parts(self):
        T = 4
        win = Window(0, 10, 6, 10)  # 10 linhas, faixas de 8 + 2
        strips = unet_predict.strip_plan(win, T, 2)
        self.assertEqual(strips, [(0, 10, 8), (1, 18, 2)])
        profile = {"driver": "GTiff", "width": 6, "height": 10, "count": 3, "dtype": "int16",
                   "transform": from_origin(0, 10, 1, 1), "crs": "EPSG:4326", "nodata": -9999}
        with tempfile.TemporaryDirectory() as d:
            def part(k):
                return os.path.join(d, f"p{k}.tif")
            full = np.arange(3 * 10 * 6, dtype="int16").reshape(3, 10, 6)
            for k, srow, sh in strips:
                r = srow - 10
                with rasterio.open(part(k), "w", **dict(profile, height=sh)) as dst:
                    dst.write(full[:, r:r + sh, :])
                    dst.update_tags(valid_pixels=10 * (k + 1), burned_pixels=k + 1, prob_pixels=2)
            out = os.path.join(d, "final.tif")
            counts = unet_predict.assemble_parts(out, profile, win, strips, part, T)
            with rasterio.open(out) as src:
                self.assertTrue((src.read() == full).all())
            self.assertEqual(counts, (30, 3, 4))


@unittest.skipUnless(HAVE, "numpy/rasterio ausentes")
class TestResumeState(unittest.TestCase):
    CFG = make_config(TRAINING_ID="1", SHORTNAME="pantanal_r1")
    MANIFEST = {"model": "m", "threshold": 0.63}
    REL = "bucket/x/parts/name"

    def run_state(self, files, saved):
        deleted, written = [], []
        with mock.patch.object(unet_predict.gcs, "list_files", return_value=files), \
                mock.patch.object(unet_predict, "read_json", return_value=saved), \
                mock.patch.object(unet_predict, "delete_files", side_effect=lambda cfg, p: deleted.extend(p)), \
                mock.patch.object(unet_predict, "write_json", side_effect=lambda cfg, p, m: written.append(m)):
            done = unet_predict.resume_state(self.CFG, self.REL, self.MANIFEST)
        return done, deleted, written

    def test_same_run_resumes(self):
        files = [f"gs://{self.REL}/{n}" for n in ("manifest.json", "part_000.tif", "part_002.tif")]
        done, deleted, written = self.run_state(files, dict(self.MANIFEST))
        self.assertEqual(done, {0, 2})
        self.assertEqual((deleted, written), ([], []))

    def test_other_model_discards(self):
        files = [f"gs://{self.REL}/{n}" for n in ("manifest.json", "part_000.tif")]
        done, deleted, written = self.run_state(files, {"model": "m", "threshold": 0.5})
        self.assertEqual(done, set())
        self.assertEqual(len(deleted), 2)
        self.assertEqual(written, [self.MANIFEST])

    def test_first_run_writes_manifest(self):
        done, deleted, written = self.run_state([], None)
        self.assertEqual((done, deleted, written), (set(), [], [self.MANIFEST]))


@unittest.skipUnless(HAVE, "numpy/rasterio ausentes")
class TestAssemblyStatus(unittest.TestCase):
    CFG = make_config(TRAINING_ID="1", SHORTNAME="pantanal_r1", REGION="Pantanal_r1")

    def test_status(self):
        P = unet_predict.paths(self.CFG)
        root = f"gs://{P['bucket']}/{P['predict']}/parts"
        man = {"col_off": 0, "row_off": 0, "width": 10, "height": 16, "tile": 4, "strip_rows": 2}  # 2 faixas
        files = [f"{root}/pantanal_r1_2026_10_prob/{n}" for n in ("manifest.json", "part_000.tif", "part_001.tif")]
        files += [f"{root}/pantanal_r1_2026_11_prob/{n}" for n in ("manifest.json", "part_000.tif")]
        files += [f"{root}/pantanal_r1_2026_12_prob/part_000.tif"]  # sem manifesto: ignorado
        files += [f"{root}/outra_2026_09_prob/manifest.json"]  # outra região: ignorada
        got = unet_predict.assembly_status(self.CFG, files, {"2026_10": man, "2026_11": man})
        self.assertEqual(got, [("2026_10", 2, 2), ("2026_11", 1, 2)])


if __name__ == "__main__":
    unittest.main()
