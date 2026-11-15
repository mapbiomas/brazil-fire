"""Regressions for process isolation, integrity and checkpoint reporting."""
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import google_crc32c
import benchmark_m2 as benchmark


class ReviewRegressions(unittest.TestCase):
    def test_external_mask_check_fails_closed(self):
        class Sidecar:
            def __init__(self, outcome):
                self.outcome = outcome
            def exists(self, **kwargs):
                self.options = kwargs
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return self.outcome
        class Bucket:
            def __init__(self, outcome):
                self.sidecar = Sidecar(outcome)
            def blob(self, name):
                self.name = name
                return self.sidecar
        class Source:
            name = "folder/image.tif"
            def __init__(self, outcome):
                self.bucket = Bucket(outcome)
        absent = Source(False)
        benchmark.ensure_no_external_mask(absent)
        self.assertEqual(absent.bucket.name, "folder/image.tif.msk")
        self.assertGreater(absent.bucket.sidecar.options["timeout"], 0)
        with self.assertRaisesRegex(RuntimeError, "external.*mask"):
            benchmark.ensure_no_external_mask(Source(True))
        with self.assertRaisesRegex(RuntimeError, "check.*mask"):
            benchmark.ensure_no_external_mask(Source(PermissionError("sensitive credential")))

    def test_download_rechecks_sidecar_after_transfer(self):
        class Source:
            name, size, generation = "image.tif", 4, 7
            crc32c = base64.b64encode(google_crc32c.Checksum(b"data").digest()).decode()
            def __init__(self):
                self.checks = 0
                self.bucket = self
            def blob(self, name):
                self.assert_name = name
                return self
            def exists(self, **kwargs):
                self.checks += 1
                return self.checks == 2
            def reload(self, **kwargs):
                pass
            def download_as_bytes(self, **kwargs):
                return b"data"[kwargs["start"]:kwargs["end"] + 1]
        with tempfile.TemporaryDirectory() as directory:
            source = Source()
            with self.assertRaisesRegex(RuntimeError, "external.*mask"):
                benchmark.national_download(source, Path(directory) / "input", 0)
            self.assertEqual(source.assert_name, "image.tif.msk")
            self.assertEqual(source.checks, 2)

    def test_gdal_tmp_cleanup_is_confined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "run"
            work.mkdir()
            tmp = work / "gdal_tmp"
            (tmp / "nested").mkdir(parents=True)
            (tmp / "nested" / "temporary").write_text("temporary")
            (work / "report.json").write_text("report")
            (work / "result.tif").write_text("result")
            (work / "other.txt").write_text("unrelated")
            outside = root / "other.txt"
            outside.write_text("outside")
            benchmark.cleanup_national(work, success=True, temp_owned=True)
            self.assertFalse(tmp.exists())
            self.assertEqual((work / "report.json").read_text(), "report")
            self.assertEqual((work / "result.tif").read_text(), "result")
            self.assertEqual((work / "other.txt").read_text(), "unrelated")
            self.assertEqual(outside.read_text(), "outside")

    def test_integrity_detects_same_length_corruption(self):
        class Blob:
            size, generation = 4, 1
            name = "image.tif"
            bucket = SimpleNamespace(blob=lambda name: SimpleNamespace(exists=lambda **kwargs: False))
            crc32c = base64.b64encode(google_crc32c.Checksum(b"good").digest()).decode()
            def reload(self, **kwargs):
                pass
            def download_as_bytes(self, **kwargs):
                return b"evil"[kwargs["start"]:kwargs["end"] + 1]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(IOError, "CRC32C"):
                benchmark.national_download(Blob(), Path(directory) / "image", 0, 2)

    def test_sanitized_bounded_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            secret = "token-secret-123"
            command = [sys.executable, "-c",
                       "import sys;sys.stderr.write('Permission denied " + secret + " ' + 'x'*200000);sys.exit(2)"]
            with self.assertRaises(subprocess.CalledProcessError):
                benchmark.monitored_process(command, directory, 0, interval=.01, log_name="download")
            data = (Path(directory) / "download.log").read_text()
            self.assertIn("permission", data.lower())
            self.assertNotIn(secret, data)
            self.assertLess(len(data), 4096)

    def test_checkpoint_pending_then_confirmed(self):
        entered, release = threading.Event(), threading.Event()
        class Blob:
            generation = None
            def upload_from_string(self, data, **kwargs):
                self.assertions = kwargs
                entered.set()
                release.wait(5)
                self.generation = (self.generation or 0) + 1
                self.contents = json.loads(data)
        blob = Blob()
        state = dict(run_id="free-" + "a" * 32, status="running", stage="download")
        writer = benchmark.CheckpointWriter(benchmark.Checkpoint(blob), state, interval=60)
        writer.start()
        self.assertTrue(entered.wait(2))
        self.assertEqual(writer.finish("failed", timeout=.01), "pending")
        release.set()
        writer.thread.join(2)
        self.assertEqual(blob.contents["status"], "failed")
        self.assertEqual(blob.assertions["retry"], None)
        self.assertGreater(blob.assertions["timeout"], 0)

    def test_cleanup_never_touches_external_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "scratch"
            work.mkdir()
            outside = root / "source.tif"
            outside.write_text("keep")
            (work / "source.tif").write_text("temporary")
            (work / "result.tif").write_text("partial")
            benchmark.cleanup_national(work, success=False)
            self.assertEqual(outside.read_text(), "keep")
            self.assertFalse((work / "source.tif").exists())
            self.assertFalse((work / "result.tif").exists())
            try:
                (work / "masked.tif").symlink_to(outside)
            except (OSError, NotImplementedError):
                return  # Windows may not permit creating symlinks.
            benchmark.cleanup_national(work, success=False)
            self.assertEqual(outside.read_text(), "keep")
            self.assertFalse((work / "masked.tif").exists())

    def test_child_temp_environment_is_run_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            before = dict(os.environ)
            env = benchmark.national_subprocess_env(directory, 2, 64)
            for name in ("TMPDIR", "TMP", "TEMP", "CPL_TMPDIR"):
                self.assertEqual(env[name], str((Path(directory) / "gdal_tmp").resolve()))
            self.assertEqual(env["GDAL_NUM_THREADS"], "2")
            self.assertEqual(dict(os.environ), before)

    def test_download_subprocess_uses_explicit_project(self):
        class Blob: pass
        calls = []
        class Client:
            def __init__(self, project):
                calls.append(project)
            def bucket(self, name):
                return self
            def blob(self, name):
                return Blob()
        from google.cloud import storage
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(storage, "Client", Client), \
             patch.object(benchmark, "national_download"):
            benchmark.national_stage("download", directory,
                                     {"gcs_uri": "gs://input/object", "reserve_bytes": 0,
                                      "gcs_project": "test-project"})
        self.assertEqual(calls, ["test-project"])

    @unittest.skipUnless(os.name == "posix", "POSIX process groups; Windows fallback is best effort")
    def test_reparented_grandchild_terminated(self):
        import psutil
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            code = ("import subprocess,sys,time; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    "open(sys.argv[1],'w').write(str(p.pid))")
            try:
                benchmark.monitored_process([sys.executable, "-c", code, str(pidfile)],
                                            directory, 0, interval=.01, log_name="group")
                self.assertIsNone(unrelated.poll())
            finally:
                unrelated.terminate()
                unrelated.wait()
            pid = int(pidfile.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and psutil.pid_exists(pid):
                try:
                    if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(.02)
            else:
                if psutil.pid_exists(pid):
                    psutil.Process(pid).kill()
                    self.fail("Reparented process survived")


if __name__ == "__main__":
    unittest.main()
