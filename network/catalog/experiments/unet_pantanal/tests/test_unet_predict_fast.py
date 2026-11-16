"""Classificação em blocos com threads: o resultado tem que ser igual ao cálculo direto, pixel a pixel.

Usa um mosaico sintético local e um "modelo" pontual (sigmoide da soma dos canais), então qualquer erro de
recorte, deslocamento, faixa ou máscara da região aparece como diferença.
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unet_config import make_config, mosaic_band_paths  # noqa: E402

try:
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    from shapely.geometry import Polygon
    import unet_io
    import unet_predict
    from unet_features import normalize
    HAVE = True
except ImportError:  # sem numpy/rasterio/shapely no ambiente
    HAVE = False

H, W = 100, 90
CFG = make_config(BANDS=["red", "nir"], INDICES=[], VALID_BAND="dayOfYear", THRESHOLD=0.5, INFER_TILE=32,
                  OVERLAP=16, INFER_BATCH=3, REGION="Pantanal_r1", TRAINING_ID="1")
STATS = {"clip_low": [0, 0], "clip_high": [1, 1], "mean": [0.5, 0.5], "std": [0.25, 0.25]}


class PointModel:
    """Probabilidade por pixel = sigmoide(canal 0 + canal 1): independe de recorte e vizinhança."""

    def predict_on_batch(self, xb):
        return (1 / (1 + np.exp(-(xb[..., 0] + xb[..., 1]))))[..., None].astype("float32")


@unittest.skipUnless(HAVE, "numpy/rasterio/shapely ausentes")
class TestFastPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(3)
        cls.data = {"red": rng.integers(1, 100, (H, W)).astype("uint8"),
                    "nir": rng.integers(1, 100, (H, W)).astype("uint8")}
        doy = rng.integers(200, 300, (H, W)).astype("uint16")
        doy[:15, :] = 0  # faixa sem dado
        doy[40:50, 30:60] = 0
        cls.data["dayOfYear"] = doy
        cls.transform = from_origin(0, H, 1, 1)
        cls.files = {}
        for b, path in mosaic_band_paths(CFG, "2026_10").items():
            f = os.path.join(cls.tmp.name, f"{b}.tif")
            with rasterio.open(f, "w", driver="GTiff", width=W, height=H, count=1, dtype=cls.data[b].dtype,
                               crs="EPSG:4326", transform=cls.transform) as dst:
                dst.write(cls.data[b], 1)
            cls.files[path] = f
        cls.region = Polygon([(5, 5), (80, 10), (85, 90), (10, 95)])  # x = coluna, y = H - linha

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def open_reader(self):
        with mock.patch.object(unet_io, "_open_raster", lambda rel: rasterio.open(self.files[rel])):
            return unet_io.MosaicReader(CFG, "2026_10", logger=lambda m, l="info": None)

    def test_read_all_parallel_equals_sequential(self):
        from concurrent.futures import ThreadPoolExecutor
        reader = self.open_reader()
        try:
            for win in (Window(10, 20, 40, 30), Window(-8, -8, 50, 50), Window(70, 80, 60, 60)):  # borda da imagem
                seq = reader.read_all(win)
                with ThreadPoolExecutor(3) as pool:
                    par = reader.read_all(win, pool=pool)
                for a, b in zip(seq, par):
                    self.assertTrue(np.array_equal(a, b))
                raw, valid = reader.read(win)
                self.assertTrue(np.array_equal(raw, seq[0]) and np.array_equal(valid, seq[1]))
        finally:
            reader.close()

    def test_block_geometry(self):
        tiles = [(64, 32, 32, 32), (64, 96, 32, 20)]
        self.assertEqual(unet_predict.block_geometry(64, tiles, 32, 16), (32, 16, 48, 96 - 32 + 32 + 32, 64))
        self.assertEqual(unet_predict.split_rows([(0, 0, 8, 8), (8, 0, 8, 8), (0, 8, 8, 8)]),
                         [(0, [(0, 0, 8, 8), (0, 8, 8, 8)]), (8, [(8, 0, 8, 8)])])

    def test_whole_pipeline_equals_direct_computation(self):
        out_dir = tempfile.mkdtemp(dir=self.tmp.name)
        with mock.patch.object(unet_io, "_open_raster", lambda rel: rasterio.open(self.files[rel])), \
                mock.patch.object(unet_predict, "region_geometry", lambda cfg, crs, logger=None: self.region), \
                mock.patch.object(unet_predict, "local_dir", lambda *a: out_dir):
            res = unet_predict.predict_mosaic(CFG, "2026_10", PointModel(), STATS,
                                              logger=lambda m, l="info": None, upload_result=False, strip_rows=1)

        # cálculo direto, imagem inteira
        refl = np.stack([self.data["red"], self.data["nir"]], axis=-1).astype("float32") / CFG["SCALE"]
        prob = PointModel().predict_on_batch(normalize(refl, STATS))[..., 0]
        valid = self.data["dayOfYear"] > 0
        valid &= geometry_mask([self.region], (H, W), self.transform, invert=True)
        p = np.where(valid, np.round(np.clip(prob, 0, 1) * 1000), unet_predict.NODATA).astype("int16")
        burned = np.where(valid, (prob >= 0.5).astype("int16"), unet_predict.NODATA)
        doy = np.where(valid & (p > 0), self.data["dayOfYear"].astype("int16"), unet_predict.NODATA)

        ref = types.SimpleNamespace(transform=self.transform, width=W, height=H)
        win = unet_predict.region_window(ref, self.region)
        sl = (slice(int(win.row_off), int(win.row_off + win.height)),
              slice(int(win.col_off), int(win.col_off + win.width)))
        with rasterio.open(res["local"]) as src:
            self.assertEqual((src.count, src.width, src.height), (3, int(win.width), int(win.height)))
            self.assertEqual(src.descriptions, ("burned", "prob", "doy"))
            got = src.read()
        self.assertTrue(np.array_equal(got[0], burned[sl]), "banda burned")
        self.assertTrue(np.array_equal(got[1], p[sl]), "banda prob")
        self.assertTrue(np.array_equal(got[2], doy[sl]), "banda doy")
        self.assertEqual(res["valid_pixels"], int(valid.sum()))
        self.assertEqual(res["burned_pixels"], int((burned == 1).sum()))
        self.assertGreater(res["valid_pixels"], 1000)


if __name__ == "__main__":
    unittest.main()
