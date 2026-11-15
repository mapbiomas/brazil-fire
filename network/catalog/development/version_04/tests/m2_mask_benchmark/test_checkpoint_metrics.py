"""Offline tests of bounded, secret-free checkpoint evidence."""
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import benchmark_m2 as benchmark


class CheckpointMetricsTest(unittest.TestCase):
    def test_periodic_and_final_metrics(self):
        class Blob:
            generation = None
            def __init__(self):
                self.rows = []
                self.updated = threading.Event()
            def upload_from_string(self, data, **kwargs):
                self.rows.append(json.loads(data))
                self.generation = (self.generation or 0) + 1
                if len(self.rows) >= 2:
                    self.updated.set()
        blob = Blob()
        with tempfile.TemporaryDirectory() as work, \
             patch.object(benchmark.shutil, "disk_usage", return_value=SimpleNamespace(free=1234)), \
             patch("psutil.virtual_memory", return_value=SimpleNamespace(available=5678)):
            writer = benchmark.CheckpointWriter(
                benchmark.Checkpoint(blob), {"status": "running", "stage": "download"},
                interval=.01, directory=work, logical_cpus=4, threads=2)
            writer.start()
            writer.complete_stage("download", 1.25)
            writer.set_stage("mask")
            self.assertTrue(blob.updated.wait(2))
            self.assertEqual(writer.finish("interrupted"), "confirmed")
        final = blob.rows[-1]
        self.assertEqual(final["status"], "interrupted")
        self.assertEqual(final["stage"], "mask")
        self.assertEqual(final["disk_free_bytes"], 1234)
        self.assertEqual(final["memory_available_bytes"], 5678)
        self.assertEqual((final["logical_cpus"], final["threads"]), (4, 2))
        self.assertEqual(final["completed_stages"], {"download": 1.25})
        self.assertGreaterEqual(final["elapsed_seconds"], final["stage_elapsed_seconds"])
        self.assertGreaterEqual(final["stage_elapsed_seconds"], 0)
        self.assertNotIn("run_id", final)
        self.assertNotIn("token", json.dumps(blob.rows))

    def test_schema_rejects_secrets_and_invalid_numbers(self):
        class Blob:
            generation = None
            def upload_from_string(self, data, **kwargs):
                self.generation = 1
        writer = benchmark.CheckpointWriter(
            benchmark.Checkpoint(Blob()), {"status": "running", "stage": "download"},
            logical_cpus=2, threads=1)
        state = writer.snapshot()
        for value in (dict(state, run_id="secret"), dict(state, elapsed_seconds=float("nan")),
                      dict(state, completed_stages={"secret": 1}),
                      dict(state, completed_stages={"mask": "secret"})):
            with self.assertRaises(ValueError):
                benchmark.Checkpoint(Blob()).write(value)


if __name__ == "__main__":
    unittest.main()
