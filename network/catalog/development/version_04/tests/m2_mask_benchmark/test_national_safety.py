"""Offline regressions for experimental safety primitives; no cloud access."""
import json
import base64
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_m2 as benchmark
import google_crc32c


class FakeBlob:
    def __init__(self):
        self.generation = None
        self.contents = None
        self.calls = []

    def upload_from_string(self, data, *, content_type, if_generation_match, timeout, retry):
        self.calls.append(if_generation_match)
        current = self.generation or 0
        if if_generation_match != current:
            raise RuntimeError("precondition failed")
        self.contents = data
        self.generation = current + 1


class NationalSafetyTest(unittest.TestCase):
    def test_disk_reserve_and_negative_inputs(self):
        with patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=100)):
            self.assertEqual(benchmark.check_disk_reserve(".", 40, 60), 100)
            with self.assertRaises(RuntimeError):
                benchmark.check_disk_reserve(".", 40, 61)
            with self.assertRaises(ValueError):
                benchmark.check_disk_reserve(".", -1)
            with self.assertRaisesRegex(RuntimeError, r"free=100 bytes, required=61 bytes, reserve=40 bytes"):
                benchmark.check_disk_reserve(".", 40, 61)

    def test_estimate_uses_source_size_and_is_not_a_guarantee(self):
        pixels, source, reserve = 28_766_214_368, 8_143_833_141, 10 * 1024**3
        estimate = benchmark.national_disk_estimate(pixels, 1, 2, source, 107_381_436_416, reserve)
        self.assertEqual(estimate["raw_data_bytes"], pixels * 2)
        self.assertEqual(estimate["byte_mask_reference_bytes"], pixels)
        self.assertGreater(estimate["estimated_additional_peak_bytes"], 0)
        self.assertIn("not an upper bound", estimate["note"])
        need = estimate["estimated_additional_peak_bytes"]
        self.assertEqual(need, 76_200_793_892)
        for free in (107_381_436_416, 234_209_320_960, need + reserve):
            with patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=free)):
                self.assertEqual(benchmark.check_disk_reserve(".", reserve, need), free)
        with patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=need + reserve - 1)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient free disk"):
                benchmark.check_disk_reserve(".", reserve, need)
        larger = benchmark.national_disk_estimate(pixels, 1, 2, source * 2, 234_209_320_960, reserve)
        self.assertGreater(larger["estimated_additional_peak_bytes"], estimate["estimated_additional_peak_bytes"])

    def test_prepare_uses_sparse_compressed_mask_with_estimated_admission(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            with rasterio.open(work / "source.tif", "w", driver="GTiff", width=128, height=128,
                               count=1, dtype="uint16", crs="EPSG:4326",
                               transform=from_origin(0, 128, 1, 1)) as dst:
                dst.write(np.zeros((128, 128), dtype="uint16"), 1)
            geometry = work / "polygon.json"
            geometry.write_text(json.dumps({"type": "Polygon", "coordinates":
                                           [[[0, 0], [64, 0], [64, 128], [0, 128], [0, 0]]]}))
            config = {"window_size": 16, "reserve_bytes": 100, "geometry_path": str(geometry),
                      "geometry_crs": "EPSG:4326"}
            estimate = benchmark.national_disk_estimate(128, 128, 2, (work / "source.tif").stat().st_size,
                                                        0, 100)
            need = estimate["estimated_additional_peak_bytes"]
            # The old raw*4 admission needs 262144 bytes in addition to the reserve.
            self.assertLess(need, 262144)
            with patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=100 + need - 1)):
                with self.assertRaisesRegex(RuntimeError, "Insufficient free disk"):
                    benchmark.national_stage("prepare", work, config)
            self.assertFalse((work / "country_mask.tif").exists())
            with patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=100 + need)):
                benchmark.national_stage("prepare", work, config)
            estimate = json.loads((work / "disk_estimate.json").read_text())
            self.assertEqual(estimate["estimated_additional_peak_bytes"], need)
            with rasterio.open(work / "country_mask.tif", "r+") as dst:
                self.assertEqual(dst.compression.value, "LZW")
                self.assertFalse(dst.read(1).any())
                dst.write(np.ones((16, 16), dtype="uint8"), 1,
                          window=rasterio.windows.Window(0, 0, 16, 16))
            with rasterio.open(work / "country_mask.tif") as src:
                self.assertEqual(int(src.read(1).sum()), 256)

    @unittest.skipUnless(shutil.which("gdal_rasterize"), "gdal_rasterize unavailable")
    def test_gdal_rasterize_updates_sparse_lzw_mask(self):
        import numpy as np
        import rasterio
        from rasterio.features import geometry_mask
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shape = {"type": "Polygon", "coordinates":
                     [[[0, 0], [32, 0], [32, 64], [0, 64], [0, 0]]]}
            geometry = work / "polygon.json"
            geometry.write_text(json.dumps(shape))
            with rasterio.open(work / "source.tif", "w", driver="GTiff", width=64, height=64,
                               count=1, dtype="uint8", crs="EPSG:4326",
                               transform=from_origin(0, 64, 1, 1)) as dst:
                dst.write(np.zeros((64, 64), dtype="uint8"), 1)
            benchmark.national_stage("prepare", work, {"window_size": 16, "reserve_bytes": 0,
                                     "geometry_path": str(geometry), "geometry_crs": "EPSG:4326"})
            subprocess.run(["gdal_rasterize", "-burn", "1", "-optim", "VECTOR",
                            str(work / "transformed.geojson"), str(work / "country_mask.tif")],
                           check=True, capture_output=True)
            with rasterio.open(work / "country_mask.tif") as src:
                expected = geometry_mask([shape], out_shape=(64, 64), transform=src.transform, invert=True)
                np.testing.assert_array_equal(src.read(1) > 0, expected)

    def test_checkpoint_conditions_isolation_and_secrets(self):
        blob = FakeBlob()
        first = benchmark.Checkpoint(blob)
        other = benchmark.Checkpoint(blob)
        state = benchmark.CheckpointWriter(first, dict(status="running", stage="download")).snapshot()
        first.write(state)
        with self.assertRaisesRegex(RuntimeError, "precondition"):
            other.write(state)
        first.write(dict(state, status="ok", completed_stages={"download": 1}))
        self.assertEqual(blob.calls, [0, 0, 1])
        self.assertEqual(json.loads(blob.contents)["completed_stages"]["download"], 1)
        with self.assertRaises(ValueError):
            first.write(dict(state, token="secret"))
        with self.assertRaises(ValueError):
            first.write(dict(state, stage="secret"))

    def test_download_ranges_generation_and_short_read(self):
        class Blob:
            size, generation = 11, 7
            name = "source.tif"
            bucket = SimpleNamespace(blob=lambda name: SimpleNamespace(exists=lambda **kwargs: False))
            crc32c = base64.b64encode(google_crc32c.Checksum(b"abcdefghijk").digest()).decode()
            def __init__(self):
                self.requests = []
                self.short = False
            def reload(self, **kwargs):
                pass
            def download_as_bytes(self, *, start, end, if_generation_match, timeout, raw_download):
                self.requests.append((start, end, if_generation_match))
                return b"abcdefghijk"[start:end + (0 if self.short else 1)]
        with tempfile.TemporaryDirectory() as temporary:
            blob = Blob()
            path = Path(temporary) / "source"
            self.assertEqual(benchmark.national_download(blob, path, 0, 4), 11)
            self.assertEqual(path.read_bytes(), b"abcdefghijk")
            self.assertEqual(blob.requests, [(0, 3, 7), (4, 7, 7), (8, 10, 7)])
            blob.short = True
            with self.assertRaisesRegex(IOError, "Incomplete"):
                benchmark.national_download(blob, Path(temporary) / "short", 0, 4)

    def test_process_stopped_on_disk_exhaustion_and_interrupt(self):
        import subprocess
        real_popen = subprocess.Popen
        for failure in (RuntimeError("disk reserve"), KeyboardInterrupt()):
            started = []
            def launch(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                started.append(process)
                return process
            with tempfile.TemporaryDirectory() as directory:
                with patch.object(subprocess, "Popen", side_effect=launch), \
                     patch.object(benchmark, "check_disk_reserve", side_effect=[100, failure]):
                    with self.assertRaises(type(failure)):
                        benchmark.monitored_process(
                            [sys.executable, "-c", "import time; time.sleep(60)"],
                            directory, 1, interval=0.01)
            self.assertIsNotNone(started[0].poll())

    def test_national_end_to_end_and_failed_cleanup(self):
        import numpy as np
        import rasterio
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.tif"
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(source, "w", driver="GTiff", width=48, height=48,
                                   count=1, dtype="uint8", crs="EPSG:4326",
                                   transform=from_origin(0, 48, 1, 1)) as dst:
                    dst.write((np.arange(48 * 48) % 256).astype("uint8").reshape(48, 48), 1)
                    valid = np.full((48, 48), 255, dtype="uint8")
                    valid[0, 1] = 0
                    dst.write_mask(valid)
            geometry = root / "polygon.json"
            geometry.write_text(json.dumps({"type": "Polygon", "coordinates": [
                [[0, 0], [24, 0], [24, 48], [0, 48], [0, 0]]]}), encoding="utf-8")
            class SourceBlob:
                size = source.stat().st_size
                generation = 123
                name = "image.tif"
                bucket = SimpleNamespace(blob=lambda name: SimpleNamespace(exists=lambda **kwargs: False))
                crc32c = base64.b64encode(google_crc32c.Checksum(source.read_bytes()).digest()).decode()
                def reload(self, **kwargs):
                    pass
                def download_as_bytes(self, *, start, end, if_generation_match, timeout, raw_download):
                    self_generation = self.generation
                    if if_generation_match != self_generation:
                        raise RuntimeError("generation mismatch")
                    return source.read_bytes()[start:end+1]
            class Bucket:
                def __init__(self):
                    self.blobs = {}
                def blob(self, name):
                    return self.blobs.setdefault(name, FakeBlob())
            class Client:
                def __init__(self):
                    self.output = Bucket()
                def bucket(self, name):
                    return SimpleNamespace(blob=lambda key: SourceBlob()) if name == "input" else self.output
            client = Client()
            config = dict(label="FREE", confirm_national=True, gcs_uri="gs://input/image.tif",
                          gcs_project="test-project",
                          window_size=16, threads=1, disk_reserve_gb=0, work_root=str(root / "runs"),
                          geometry_path=str(geometry), geometry_crs="EPSG:4326", repeats=1,
                          window_workers=[1], checkpoint_bucket="output", checkpoint_interval_seconds=0.01)
            def stages(stage, work, stage_config):
                self.assertEqual(stage_config["gcs_project"], "test-project")
                if stage == "mask":
                    shapes = list(benchmark.polygons(json.loads((work / "transformed.geojson").read_text())))
                    with rasterio.open(work / "country_mask.tif", "r+") as dst:
                        dst.write(rasterize(shapes, out_shape=(dst.height, dst.width),
                                            transform=dst.transform, dtype="uint8"), 1)
                else:
                    command = [sys.executable, str(Path(benchmark.__file__).resolve()), "--national-stage",
                               stage, str(work), str(work / "stage_config.json")]
                    benchmark.monitored_process(command, work, 0, interval=0.01, env=os.environ.copy())
            report, work = benchmark.run_national(config, client=client, stage_runner=stages)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["checkpoint"], "confirmed")
            checkpoint_blob = next(iter(client.output.blobs.values()))
            checkpoint_state = json.loads(checkpoint_blob.contents)
            self.assertEqual(checkpoint_state["status"], "ok")
            self.assertEqual(set(checkpoint_state["completed_stages"]),
                             {"download", "prepare", "mask", "application", "cog", "validation"})
            self.assertGreater(checkpoint_state["disk_free_bytes"], 0)
            self.assertNotIn("run_id", checkpoint_state)
            with rasterio.open(work / "result.tif") as dst:
                self.assertEqual(dst.count, 1)
                self.assertEqual(dst.read(1)[0, 0], 0)
                self.assertEqual((dst.dataset_mask() > 0).sum(), 1151)
            self.assertFalse((work / "source.tif").exists())
            self.assertFalse((work / "masked.tif").exists())
            self.assertFalse((work / "gdal_tmp").exists())
            def interrupted(stage, work, stage_config):
                (work / "disk_estimate.json").write_text('{"partial":', encoding="utf-8")
                raise KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                benchmark.run_national(config, client=client, stage_runner=interrupted)
            failed_work = next(p.resolve() for p in (root / "runs").iterdir() if p.resolve() != work)
            failed = json.loads((failed_work / "report.json").read_text())
            self.assertEqual(failed["status"], "interrupted")
            self.assertEqual(failed["disk_estimate_error"], "Disk estimate unavailable or invalid")
            failed_checkpoint = json.loads(next(value.contents for value in client.output.blobs.values()
                                                if json.loads(value.contents)["status"] == "interrupted"))
            self.assertEqual(failed_checkpoint["stage"], "prepare")
            self.assertIn("download", failed_checkpoint["completed_stages"])
            self.assertNotIn("prepare", failed_checkpoint["completed_stages"])
            self.assertFalse((failed_work / "source.tif").exists())
            self.assertFalse((failed_work / "result.tif").exists())
            self.assertFalse((failed_work / "gdal_tmp").exists())
            def corrupt(stage, work, stage_config):
                if stage == "validation":
                    (work / "result.tif").write_bytes(b"corrupt")
                stages(stage, work, stage_config)
            with self.assertRaises(subprocess.CalledProcessError):
                benchmark.run_national(config, client=client, stage_runner=corrupt)
            corrupted_work = next(p.resolve() for p in (root / "runs").iterdir()
                                  if p.resolve() not in (work, failed_work))
            failure = json.loads((corrupted_work / "report.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertFalse((corrupted_work / "result.tif").exists())
            entered, release = threading.Event(), threading.Event()
            class SlowBlob(FakeBlob):
                def upload_from_string(self, *args, **kwargs):
                    entered.set()
                    release.wait(5)
                    super().upload_from_string(*args, **kwargs)
            slow = SlowBlob()
            class SlowClient(Client):
                def bucket(self, name):
                    if name == "input":
                        return super().bucket(name)
                    return SimpleNamespace(blob=lambda key: slow)
            def fail_after_upload_starts(stage, work, stage_config):
                self.assertTrue(entered.wait(2))
                raise RuntimeError("synthetic")
            writers = []
            real_finish = benchmark.CheckpointWriter.finish
            def fast_finish(writer, status):
                writers.append(writer)
                return real_finish(writer, status, timeout=.01)
            with patch.object(benchmark.CheckpointWriter, "finish", fast_finish):
                with self.assertRaises(RuntimeError):
                    benchmark.run_national(config, client=SlowClient(), stage_runner=fail_after_upload_starts)
            pending_work = next(p.resolve() for p in (root / "runs").iterdir()
                                if p.resolve() not in (work, failed_work, corrupted_work))
            pending = json.loads((pending_work / "report.json").read_text())
            self.assertEqual(pending["checkpoint"], "pending")
            release.set()
            writers[0].thread.join(2)
            self.assertEqual(json.loads(slow.contents)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
