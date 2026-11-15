"""Isolated M2 mask benchmark; national mode can write optional checkpoints."""
import csv
import hashlib
import json
import os
import platform
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def check_disk_reserve(directory, reserve_bytes, required_bytes=0):
    """Best-effort admission check; concurrent writers can still exhaust disk."""
    if reserve_bytes < 0 or required_bytes < 0:
        raise ValueError("Disk reserve and required bytes must be non-negative")
    free = shutil.disk_usage(directory).free
    if free - required_bytes < reserve_bytes:
        raise RuntimeError("Insufficient free disk for operation and reserve")
    return free


def monitored_process(command, directory, reserve_bytes, interval=1.0, env=None, log_name="process"):
    """Supervise an isolated POSIX process group; Windows uses best-effort tree cleanup."""
    import psutil
    import re
    import signal
    import subprocess

    if interval <= 0 or not re.fullmatch(r"[a-z_]+", log_name):
        raise ValueError("Invalid interval or log name")
    check_disk_reserve(directory, reserve_bytes)
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env,
                               start_new_session=(os.name == "posix"))
    captured = bytearray()
    def drain():
        with process.stderr as stderr:
            while True:
                chunk = stderr.read(8192)
                if not chunk:
                    break
                if len(captured) < 65536:
                    captured.extend(chunk[:65536-len(captured)])
    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    children = {}
    try:
        while process.poll() is None:
            if os.name != "posix":
                try:
                    children.update({p.pid: p for p in psutil.Process(process.pid).children(recursive=True)})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            check_disk_reserve(directory, reserve_bytes)
            time.sleep(interval)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
    finally:
        if os.name == "posix":
            # Group remains addressable after its leader exits, including reparented children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.1)
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            try:
                children.update({p.pid: p for p in psutil.Process(process.pid).children(recursive=True)})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            for child in children.values():
                try:
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            if process.poll() is None:
                process.terminate()
            _, alive = psutil.wait_procs(list(children.values()), timeout=3)
            for child in alive:
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        reader.join(timeout=2)
        # Never persist arbitrary stderr, URLs, command arguments or exception text.
        # Only emit fixed diagnostic labels selected from bounded stderr.
        text = bytes(captured).lower()
        labels = [label for needle, label in (
            (b"permission denied", "permission denied"), (b"access denied", "access denied"),
            (b"no space left", "disk full"), (b"disk full", "disk full"),
            (b"no such file", "file not found"), (b"crc32c", "CRC32C integrity error"),
            (b"insufficient free disk", "disk reserve reached"),
            (b"output grid or dtype differs", "output grid or dtype differs"),
            (b"rasterized mask differs from reference", "polygon mask validation failed"),
            (b"output validity differs", "output validity validation failed"),
            (b"output data differs", "output pixel validation failed"),
            (b"source must have one non-alpha band", "unsupported source bands"),
            (b"source has no crs", "source CRS missing"),
            (b"incomplete gcs range", "incomplete download range"),
            (b"source has an external mask sidecar", "external mask sidecar present; standalone TIFF required"),
            (b"unable to check external mask sidecar", "external mask sidecar check failed; access not confirmed"),
            (b"traceback", "Python exception"), (b"error", "error reported"))
                  if needle in text]
        try:
            (Path(directory) / (log_name + ".log")).write_text(
                "exit_code: " + str(process.returncode) + "\ndiagnostics: " +
                (", ".join(labels) if labels else "no recognized stderr category") + "\n",
                encoding="utf-8")
        except OSError:
            pass


class Checkpoint:
    """Conditional JSON checkpoint on a caller-provided, unique GCS blob."""

    def __init__(self, blob):
        self.blob = blob
        self.generation = 0

    def write(self, state):
        # Restrict schema rather than attempting to redact unknown credentials.
        import math
        allowed = {"status", "stage", "elapsed_seconds", "stage_elapsed_seconds",
                   "disk_free_bytes", "memory_available_bytes", "logical_cpus",
                   "threads", "completed_stages"}
        stages = {"download", "prepare", "mask", "application", "cog", "validation"}
        if not isinstance(state, dict) or set(state) != allowed:
            raise ValueError("Unsupported checkpoint fields")
        def seconds(value):
            return type(value) in (int, float) and math.isfinite(value) and value >= 0
        completed = state["completed_stages"]
        if (state["status"] not in {"running", "ok", "failed", "interrupted"}
                or state["stage"] not in stages | {"complete"}
                or not seconds(state["elapsed_seconds"])
                or not seconds(state["stage_elapsed_seconds"])
                or any(state[key] is not None and (type(state[key]) is not int or state[key] < 0)
                       for key in ("disk_free_bytes", "memory_available_bytes"))
                or any(type(state[key]) is not int or state[key] < 1
                       for key in ("logical_cpus", "threads"))
                or not isinstance(completed, dict) or set(completed) - stages
                or any(not seconds(value) for value in completed.values())):
            raise ValueError("Invalid checkpoint state")
        self.blob.upload_from_string(json.dumps(state, allow_nan=False), content_type="application/json",
                                     if_generation_match=self.generation, timeout=5, retry=None)
        if self.blob.generation is None:
            raise RuntimeError("GCS did not return a generation; checkpoint updates disabled")
        self.generation = int(self.blob.generation)


class CheckpointWriter:
    """Periodic checkpoint worker; final result is confirmed, failed or pending."""

    def __init__(self, checkpoint, state, interval=30, directory=None, logical_cpus=None, threads=1):
        self.checkpoint = checkpoint
        self.state = state
        self.interval = interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.result = "pending"
        self.directory = directory
        self.logical_cpus = logical_cpus or (os.cpu_count() or 1)
        self.threads = threads
        self.started = self.stage_started = time.monotonic()
        self.finished_at = None
        self.completed = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_stage(self, stage):
        with self.lock:
            self.state["stage"] = stage
            self.stage_started = time.monotonic()

    def complete_stage(self, stage, seconds):
        with self.lock:
            self.completed[stage] = seconds

    def snapshot(self):
        import psutil
        with self.lock:
            now = self.finished_at if self.finished_at is not None else time.monotonic()
            snapshot = {"status": self.state["status"], "stage": self.state["stage"],
                        "elapsed_seconds": max(0, now - self.started),
                        "stage_elapsed_seconds": max(0, now - self.stage_started),
                        "logical_cpus": self.logical_cpus, "threads": self.threads,
                        "completed_stages": dict(self.completed)}
        try:
            snapshot["disk_free_bytes"] = shutil.disk_usage(self.directory).free if self.directory else None
        except OSError:
            snapshot["disk_free_bytes"] = None
        try:
            snapshot["memory_available_bytes"] = psutil.virtual_memory().available
        except Exception:
            snapshot["memory_available_bytes"] = None
        return snapshot

    def _run(self):
        while True:
            try:
                snapshot = self.snapshot()
                self.checkpoint.write(snapshot)
            except Exception:
                self.result = "failed"
                return
            if self.stop.is_set() and snapshot["status"] != "running":
                self.result = "confirmed"
                return
            self.stop.wait(self.interval)

    def finish(self, status, timeout=7):
        with self.lock:
            self.state["status"] = status
            self.finished_at = time.monotonic()
        self.stop.set()
        self.thread.join(timeout)
        return "pending" if self.thread.is_alive() else self.result


def cleanup_national(work, success, temp_owned=False):
    """Unlink fixed names inside this run's scratch directory only."""
    names = ["source.tif", "country_mask.tif", "masked.tif", "transformed.geojson", "stage_config.json"]
    if not success:
        names.append("result.tif")
    for name in names:
        try:
            (Path(work) / name).unlink(missing_ok=True)
        except OSError:
            pass
    if temp_owned:
        temporary = Path(work) / "gdal_tmp"
        try:
            if temporary.is_symlink():
                temporary.unlink()
            elif temporary.exists():
                shutil.rmtree(temporary)
        except OSError:
            pass


def national_subprocess_env(work, threads, cache_mb):
    """Keep temporary files within this run; do not mutate the parent environment."""
    scratch = str((Path(work) / "gdal_tmp").resolve())
    return dict(os.environ, TMPDIR=scratch, TMP=scratch, TEMP=scratch, CPL_TMPDIR=scratch,
                GDAL_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), GDAL_CACHEMAX=str(cache_mb))


def national_stage(stage, work, config):
    """Local raster stages. Invoked in an independently supervised process."""
    if stage == "download":
        from google.cloud import storage
        from urllib.parse import urlsplit
        parts = urlsplit(config["gcs_uri"])
        blob = storage.Client(project=config["gcs_project"]).bucket(parts.netloc).blob(parts.path.lstrip("/"))
        national_download(blob, Path(work) / "source.tif", config["reserve_bytes"])
        return
    import numpy as np
    import rasterio
    from rasterio.enums import ColorInterp, MaskFlags
    from rasterio.features import geometry_mask
    from rasterio.shutil import copy as rio_copy
    from rasterio.warp import transform_geom
    from rasterio.windows import transform as window_transform

    work = Path(work)
    size = config["window_size"]
    reserve = config["reserve_bytes"]
    source, grid = work / "source.tif", work / "country_mask.tif"
    masked, output = work / "masked.tif", work / "result.tif"
    geometry = work / "transformed.geojson"
    if stage == "prepare":
        with open(config["geometry_path"], encoding="utf-8") as stream:
            shapes = list(polygons(json.load(stream)))
        if not shapes:
            raise ValueError("No polygons")
        with rasterio.open(source) as src:
            if not src.crs:
                raise ValueError("Source has no CRS")
            bands = [i for i, color in enumerate(src.colorinterp, 1) if color != ColorInterp.alpha]
            if len(bands) != 1:
                raise ValueError("Source must have one non-alpha band")
            shapes = [transform_geom(config["geometry_crs"], src.crs, shape) for shape in shapes]
            profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                           crs=src.crs, transform=src.transform, dtype="uint8",
                           tiled=True, blockxsize=size, blockysize=size, BIGTIFF="YES")
            # Three uncompressed rasters plus an output, mask and overview allowance.
            raw = src.width * src.height * (np.dtype(src.dtypes[bands[0]-1]).itemsize + 2)
            check_disk_reserve(work, reserve, raw * 4)
            geometry.write_text(json.dumps({
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": src.crs.to_wkt()}},
                "features": [{"type": "Feature", "properties": {}, "geometry": shape}
                             for shape in shapes]}), encoding="utf-8")
            with rasterio.open(grid, "w", **profile):
                pass
    elif stage == "application":
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(source) as src, rasterio.open(grid) as mask:
            band = next(i for i, color in enumerate(src.colorinterp, 1) if color != ColorInterp.alpha)
            profile = dict(driver="GTiff", width=src.width, height=src.height, count=1,
                           crs=src.crs, transform=src.transform, dtype=src.dtypes[band-1],
                           nodata=src.nodatavals[band-1], tiled=True, blockxsize=size,
                           blockysize=size, compress="LZW", BIGTIFF="YES")
            with rasterio.open(masked, "w", **profile) as dst:
                for win in windows(src.width, src.height, size):
                    check_disk_reserve(work, reserve)
                    data = src.read(band, window=win)
                    valid = (src.dataset_mask(window=win) > 0) & (mask.read(1, window=win) > 0)
                    dst.write(data, 1, window=win)
                    dst.write_mask(valid.astype("uint8") * 255, window=win)
    elif stage == "cog":
        rio_copy(masked, output, driver="COG", compress="LZW", BIGTIFF="YES",
                 NUM_THREADS=str(config["threads"]), BLOCKSIZE=size)
    elif stage == "validation":
        with geometry.open(encoding="utf-8") as stream:
            shapes = list(polygons(json.load(stream)))
        with rasterio.open(source) as src, rasterio.open(grid) as mask, rasterio.open(output) as dst:
            band = next(i for i, color in enumerate(src.colorinterp, 1) if color != ColorInterp.alpha)
            if ((src.width, src.height, src.crs, src.transform, src.dtypes[band-1]) !=
                    (dst.width, dst.height, dst.crs, dst.transform, dst.dtypes[0])):
                raise ValueError("Output grid or dtype differs")
            if (dst.count != 1 or ColorInterp.alpha in dst.colorinterp
                    or MaskFlags.per_dataset not in dst.mask_flag_enums[0]
                    or dst.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG"
                    or Path(str(output) + ".msk").exists()):
                raise ValueError("Expected one-band COG with internal mask and no alpha")
            with output.open("rb") as stream:
                if stream.read(4) not in (b"II+\x00", b"MM\x00+"):
                    raise ValueError("Output is not BigTIFF")
            for win in windows(src.width, src.height, size):
                check_disk_reserve(work, reserve)
                inside = geometry_mask(shapes, out_shape=(int(win.height), int(win.width)),
                                       transform=window_transform(win, src.transform), invert=True)
                if not np.array_equal(inside, mask.read(1, window=win) > 0):
                    raise ValueError("Rasterized mask differs from reference")
                if not np.array_equal(inside & (src.dataset_mask(window=win) > 0),
                                      dst.dataset_mask(window=win) > 0):
                    raise ValueError("Output validity differs")
                if not np.array_equal(src.read(band, window=win), dst.read(1, window=win), equal_nan=True):
                    raise ValueError("Output data differs")
    else:
        raise ValueError("Unknown national stage")


def ensure_no_external_mask(blob):
    """Reject sidecar masks and fail closed when their absence cannot be checked."""
    try:
        exists = blob.bucket.blob(blob.name + ".msk").exists(timeout=30, retry=None)
    except Exception:
        raise RuntimeError("Unable to check external mask sidecar") from None
    if exists:
        raise RuntimeError("Source has an external mask sidecar; standalone TIFF required")


def national_download(blob, path, reserve, chunk_bytes=8 * 1024 * 1024):
    """Download a fixed object generation in bounded requests."""
    import base64
    import binascii
    import google_crc32c

    ensure_no_external_mask(blob)
    blob.reload(timeout=60)
    length, generation = int(blob.size), int(blob.generation)
    if length <= 0 or generation <= 0 or chunk_bytes <= 0:
        raise ValueError("Invalid object metadata or chunk size")
    try:
        expected = base64.b64decode(blob.crc32c, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ValueError("Missing or invalid CRC32C metadata") from exc
    if len(expected) != 4:
        raise ValueError("Invalid CRC32C metadata")
    checksum = google_crc32c.Checksum()
    check_disk_reserve(path.parent, reserve, length)
    with path.open("xb") as stream:
        for start in range(0, length, chunk_bytes):
            check_disk_reserve(path.parent, reserve, min(chunk_bytes, length - start))
            end = min(length, start + chunk_bytes) - 1
            data = blob.download_as_bytes(start=start, end=end, if_generation_match=generation,
                                          timeout=60, raw_download=True)
            if len(data) != end - start + 1:
                raise IOError("Incomplete GCS range")
            stream.write(data)
            checksum.update(data)
    if checksum.digest() != expected:
        raise IOError("CRC32C mismatch")
    ensure_no_external_mask(blob)
    return length


def run_national(config, client=None, stage_runner=None):
    """Experimental single-scenario national run; no EE operations or raster uploads."""
    import math
    import psutil
    import re
    import subprocess
    import sys
    from urllib.parse import urlsplit

    cfg = dict(config)
    if not cfg.get("confirm_national") or cfg.get("label") not in ("FREE", "PRO"):
        raise ValueError("National processing requires FREE/PRO and explicit confirmation")
    if not isinstance(cfg.get("gcs_project"), str) or not cfg["gcs_project"].strip():
        raise ValueError("Set an explicit GCS project")
    parts = urlsplit(cfg["gcs_uri"])
    if (parts.scheme != "gs" or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]", parts.netloc)
            or not parts.path.strip("/") or parts.query or parts.fragment):
        raise ValueError("Supply a specific gs:// object")
    size, threads = int(cfg["window_size"]), int(cfg["threads"])
    reserve = int(float(cfg["disk_reserve_gb"]) * 1024**3)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if (size < 16 or size > 4096 or size % 16 or threads < 1
            or reserve < 0 or not cfg.get("geometry_crs")
            or int(cfg.get("gdal_cache_mb", 128)) < 1
            or float(cfg.get("checkpoint_interval_seconds", 30)) <= 0):
        raise ValueError("Invalid window, thread budget, disk reserve or geometry CRS")
    available_ram = psutil.virtual_memory().available
    threads = min(threads, cpus)
    max_window = int(math.sqrt(max(0, available_ram // (16 * 32)))) // 16 * 16
    if max_window < 16 or available_ram < 32 * 1024 * 1024:
        raise RuntimeError("Insufficient available memory for national processing")
    size = min(size, max_window)
    cache_mb = min(int(cfg.get("gdal_cache_mb", 128)), max(1, available_ram // (8 * 1024**2)))
    if cfg.get("repeats", 1) != 1 or cfg.get("window_workers", [1]) != [1]:
        raise ValueError("National mode supports one scenario and one repetition")
    if not shutil.which("gdal_rasterize") and stage_runner is None:
        raise RuntimeError("gdal_rasterize is required")
    bucket = cfg.get("checkpoint_bucket", "")
    prefix = cfg.get("checkpoint_prefix", "temp/registro_de_testes_de_mosaico").strip("/")
    if bucket and (not prefix or "://" in bucket or "/" in bucket or ".." in prefix.split("/")):
        raise ValueError("Invalid checkpoint bucket or prefix")
    if client is None:
        from google.cloud import storage
        client = storage.Client(project=cfg["gcs_project"])
    run_id = cfg["label"].lower() + "-" + uuid.uuid4().hex
    work = (Path(cfg["work_root"]) / run_id).resolve()
    work.mkdir(parents=True, exist_ok=False)
    reserve = max(reserve, 0)
    state = {"status": "running", "stage": "download"}
    report = {"run_id": run_id, "status": "running", "stages": [],
              "resources": {"window_size": size, "threads": threads, "gdal_cache_mb": cache_mb,
                            "available_memory_at_start": available_ram},
              "notes": "Experimental best effort. No disk or completion guarantee. Polygon mask does not hide pixel values. Standalone source must include its validity mask; external sidecars are not downloaded.",
              "checkpoint": "disabled"}
    checkpoint_writer = None
    if bucket:
        checkpoint = Checkpoint(client.bucket(bucket).blob(prefix + "/" + run_id + "/checkpoint.json"))
        report["checkpoint"] = "pending"
        checkpoint_writer = CheckpointWriter(checkpoint, state, float(cfg.get("checkpoint_interval_seconds", 30)),
                                             directory=work, logical_cpus=cpus, threads=threads)
    stage_config = {"window_size": size, "threads": threads, "reserve_bytes": reserve,
                    "geometry_path": str(Path(cfg["geometry_path"]).resolve()),
                    "geometry_crs": cfg["geometry_crs"], "gcs_uri": cfg["gcs_uri"],
                    "gcs_project": cfg["gcs_project"]}
    stage_config_path = work / "stage_config.json"
    child_env = national_subprocess_env(work, threads, cache_mb)
    def set_stage(stage):
        if checkpoint_writer:
            checkpoint_writer.set_stage(stage)
        else:
            state["stage"] = stage
    def local_stage(stage):
        set_stage(stage)
        started = time.perf_counter()
        if stage_runner is not None:
            stage_runner(stage, work, stage_config)
        else:
            if stage == "mask":
                command = ["gdal_rasterize", "-burn", "1", "-optim", "VECTOR",
                           str(work / "transformed.geojson"), str(work / "country_mask.tif")]
            else:
                command = [sys.executable, str(Path(__file__).resolve()), "--national-stage",
                           stage, str(work), str(stage_config_path)]
            monitored_process(command, work, reserve, env=child_env, log_name=stage)
        duration = time.perf_counter() - started
        report["stages"].append({"stage": stage, "seconds": duration})
        if checkpoint_writer:
            checkpoint_writer.complete_stage(stage, duration)
    temp_owned = False
    try:
        (work / "gdal_tmp").mkdir()
        temp_owned = True
        stage_config_path.write_text(json.dumps(stage_config), encoding="utf-8")
        if checkpoint_writer:
            checkpoint_writer.start()
        started = time.perf_counter()
        if stage_runner is None:
            command = [sys.executable, str(Path(__file__).resolve()), "--national-stage",
                       "download", str(work), str(stage_config_path)]
            monitored_process(command, work, reserve, env=child_env, log_name="download")
            report["input_bytes"] = (work / "source.tif").stat().st_size
        else:
            blob = client.bucket(parts.netloc).blob(parts.path.lstrip("/"))
            report["input_bytes"] = national_download(blob, work / "source.tif", reserve)
        duration = time.perf_counter() - started
        report["stages"].append({"stage": "download", "seconds": duration})
        if checkpoint_writer:
            checkpoint_writer.complete_stage("download", duration)
        for stage in ("prepare", "mask", "application", "cog", "validation"):
            local_stage(stage)
        set_stage("complete")
        report["status"] = "ok"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error_type"] = type(exc).__name__
        report["failed_stage"] = state["stage"]
        raise
    finally:
        if checkpoint_writer and checkpoint_writer.thread.ident is not None:
            report["checkpoint"] = checkpoint_writer.finish(report["status"])
        cleanup_national(work, report["status"] == "ok", temp_owned=temp_owned)
        (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("National run directory:", work)
    return report, work


def worker_rounds(workers, repeats):
    """Rotate the starting worker each round to distribute ordering effects."""
    return [workers[i % len(workers):] + workers[:i % len(workers)]
            for i in range(repeats)]


def list_chunks(client, uri, limit):
    """List at most limit + 1 objects, counting all types, before downloading."""
    from urllib.parse import urlsplit
    parts = urlsplit(uri)
    prefix = parts.path.lstrip("/")
    if (parts.scheme != "gs" or not parts.netloc or not prefix.strip("/")
            or parts.query or parts.fragment or limit < 1):
        raise ValueError("Supply a specific gs://bucket/object-prefix and positive max_chunks")
    blobs = client.list_blobs(parts.netloc, prefix=prefix, max_results=limit + 1,
                              page_size=min(limit + 1, 1000))
    objects = []
    for blob in blobs:
        objects.append(blob)
        if len(objects) > limit:
            raise ValueError("Object limit exceeded (including non-TIFF objects)")
    chunks = sorted((x for x in objects if x.name.lower().endswith((".tif", ".tiff"))),
                    key=lambda x: x.name)
    if not chunks:
        raise ValueError("No TIFF chunks")
    return chunks


def polygons(obj):
    kind = obj.get("type")
    if kind == "FeatureCollection":
        for item in obj.get("features", []):
            yield from polygons(item)
    elif kind == "Feature":
        if obj.get("geometry"):
            yield from polygons(obj["geometry"])
    elif kind == "GeometryCollection":
        for item in obj.get("geometries", []):
            yield from polygons(item)
    elif kind in ("Polygon", "MultiPolygon"):
        yield obj


def windows(width, height, size):
    from rasterio.windows import Window
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


@contextmanager
def measure(name, rows, directory):
    import psutil
    process = psutil.Process()
    start = time.perf_counter()
    cpu = process.cpu_times()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.2):
            peak[0] = max(peak[0], process.memory_info().rss)
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        stop.set()
        worker.join()
        end_cpu = process.cpu_times()
        rows.append(dict(stage=name, status=status, seconds=time.perf_counter() - start,
                         cpu_seconds=(end_cpu.user + end_cpu.system - cpu.user - cpu.system),
                         peak_process_rss_bytes=max(peak[0], process.memory_info().rss),
                         disk_free_bytes=shutil.disk_usage(directory).free))


def run(config):
    """Benchmark a reusable per-grid mask and bounded parallel window application."""
    import collections
    from concurrent.futures import ThreadPoolExecutor
    import statistics
    import subprocess
    import numpy as np
    import psutil
    import rasterio
    from rasterio.enums import ColorInterp, MaskFlags
    from rasterio.features import geometry_mask, rasterize
    from rasterio.shutil import copy as rio_copy
    from rasterio.windows import Window
    from rasterio.warp import transform_geom
    import google.auth
    from google.auth.transport.requests import Request

    cfg = dict(config)
    uri = cfg['gcs_uri']
    if (not uri.startswith('gs://') or len(uri) <= 5 or '?' in uri or '#' in uri
            or cfg['label'] not in ('FREE', 'PRO')):
        raise ValueError('Supply a gs:// object and FREE/PRO label')
    if cfg['mode'] not in ('subset', 'national'):
        raise ValueError('mode must be subset or national')
    if cfg['mode'] == 'national' and not cfg.get('confirm_national'):
        raise ValueError('National processing requires explicit confirmation')
    if cfg['mode'] == 'national':
        return run_national(cfg)
    size, threads = int(cfg['window_size']), int(cfg['threads'])
    workers = cfg.get('window_workers', [1, 2, 4])
    repeats = int(cfg.get('repeats', 1))
    if size < 16 or size > 4096 or size % 16 or threads < 1 or repeats < 1 or repeats > 10:
        raise ValueError('Invalid window size, GDAL threads or repeats')
    if float(cfg['disk_reserve_gb']) < 0:
        raise ValueError('disk_reserve_gb must be non-negative')
    if not workers or len(set(workers)) != len(workers) or any(type(x) is not int or x not in (1, 2, 4) for x in workers):
        raise ValueError('window_workers must be a unique nonempty subset of [1, 2, 4]')
    with open(cfg['geometry_path'], encoding='utf-8') as stream:
        shapes = list(polygons(json.load(stream)))
    if not shapes or not cfg['geometry_crs']:
        raise ValueError('Provide polygon GeoJSON and its CRS')
    run_id = cfg['label'].lower() + '-' + uuid.uuid4().hex
    work = Path(cfg['work_root']) / run_id
    work.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {'run_id': run_id, 'config': cfg, 'stages': rows, 'scenarios': [],
              'method': 'Prototype: one rasterized grid mask reused by bounded window workers; not production M2.',
              'measurement_notes': 'Application includes local reads, mask operations, writing and LZW compression; COG conversion is separate. OS/GDAL caches and prior stages affect timings; rotations do not eliminate cache bias. RSS and CPU cover the parent process only; sampled RSS may miss peaks.',
              'security_notes': 'A validity mask does not protect pixel confidentiality. Reports contain configuration, URIs and potentially private errors; review before sharing.',
              'national_notes': 'GDAL uses a fixed access token refreshed once before download; it can expire during a long national run. National reference uses the same gdal_rasterize method and is not independent.',
              'projection_note': 'VERY UNCERTAIN pixel-linear projection of mask application only; not a prediction or promise.',
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'rasterio': rasterio.__version__, 'gdal': rasterio.__gdal_version__,
                              'numpy': np.__version__, 'logical_cpus': os.cpu_count(),
                              'memory_total_bytes': psutil.virtual_memory().total,
                              'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'geometry_sha256': hashlib.sha256(Path(cfg['geometry_path']).read_bytes()).hexdigest()}}
    source_path, grid_path = work / 'source.tif', work / 'country_mask.tif'
    try:
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/devstorage.read_only'])
        credentials.refresh(Request())
        with rasterio.Env(GS_OAUTH2_ACCESS_TOKEN=credentials.token, GDAL_NUM_THREADS=str(threads),
                          GDAL_CACHEMAX=int(cfg['gdal_cache_mb']) * 1024 * 1024):
            with measure('download', rows, work):
                with rasterio.open('/vsigs/' + uri[5:]) as remote:
                    if not remote.crs:
                        raise ValueError('Source has no CRS')
                    bands = [i for i, c in enumerate(remote.colorinterp, 1) if c != ColorInterp.alpha]
                    if len(bands) != 1:
                        raise ValueError('Input must have exactly one non-alpha data band')
                    total_pixels = remote.width * remote.height
                    if cfg['mode'] == 'subset':
                        col, row = int(cfg['subset_col']), int(cfg['subset_row'])
                        width, height = int(cfg['subset_width']), int(cfg['subset_height'])
                        if min(col, row) < 0 or min(width, height) <= 0 or col + width > remote.width or row + height > remote.height:
                            raise ValueError('Subset window outside source')
                        if width * height > 64_000_000:
                            raise ValueError('Subset exceeds 64 million pixels; choose a smaller subset or opt in to national')
                    else:
                        col, row, width, height = 0, 0, remote.width, remote.height
                    need = width * height * (np.dtype(remote.dtypes[bands[0]-1]).itemsize + 2) * (4 + len(workers) * repeats * 2)
                    if shutil.disk_usage(work).free - need < int(cfg['disk_reserve_gb'] * 1024**3):
                        raise RuntimeError('Insufficient disk for conservative estimated intermediates + reserve')
                    area = Window(col, row, width, height)
                    profile = remote.profile.copy()
                    profile.update(driver='GTiff', count=1, width=width, height=height,
                                   transform=remote.window_transform(area), tiled=True,
                                   blockxsize=size, blockysize=size, compress='LZW', BIGTIFF='YES')
                    profile.pop('photometric', None)
                    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(source_path, 'w', **profile) as local:
                        for win in windows(width, height, size):
                            rwin = Window(col + win.col_off, row + win.row_off, win.width, win.height)
                            local.write(remote.read(bands[0], window=rwin), 1, window=win)
                            local.write_mask(remote.dataset_mask(window=rwin), window=win)
                    shapes = [transform_geom(cfg['geometry_crs'], remote.crs, geom) for geom in shapes]
            mask_profile = dict(profile, dtype='uint8', nodata=None)
            with measure('mask_creation_once', rows, work):
                if cfg['mode'] == 'subset':
                    with rasterio.open(grid_path, 'w', **mask_profile) as dst:
                        dst.write(rasterize(shapes, out_shape=(height, width), transform=profile['transform'],
                                            fill=0, default_value=1, dtype='uint8'), 1)
                else:
                    if not shutil.which('gdal_rasterize'):
                        raise RuntimeError('National mode requires gdal_rasterize on PATH')
                    geometry_path = work / 'polygons.geojson'
                    geometry_path.write_text(json.dumps({
                        'type': 'FeatureCollection',
                        'crs': {'type': 'name', 'properties': {'name': profile['crs'].to_wkt()}},
                        'features': [{'type': 'Feature', 'properties': {}, 'geometry': x} for x in shapes]
                    }), encoding='utf-8')
                    with rasterio.open(grid_path, 'w', **mask_profile):
                        pass
                    subprocess.run(['gdal_rasterize', '-burn', '1', '-optim', 'VECTOR', str(geometry_path), str(grid_path)],
                                   check=True, capture_output=True, text=True)
            # Separate reference file; national mode repeats the same GDAL command,
            # so it is not an independent rasterization method.
            with measure('validation_reference_creation', rows, work):
                reference = work / 'reference_mask.tif'
                with rasterio.open(reference, 'w', **mask_profile) as dst:
                    if cfg['mode'] == 'subset':
                        dst.write(geometry_mask(shapes, out_shape=(height, width), transform=profile['transform'],
                                                invert=True).astype('uint8'), 1)
                if cfg['mode'] == 'national':
                    subprocess.run(['gdal_rasterize', '-burn', '1', '-optim', 'VECTOR', str(geometry_path), str(reference)],
                                   check=True, capture_output=True, text=True)
            # Each thread owns its read handles. Only the calling thread owns the writer.
            def read_window(win):
                with rasterio.open(source_path) as source, rasterio.open(grid_path) as grid:
                    data = source.read(1, window=win)
                    valid = (source.dataset_mask(window=win) > 0) & (grid.read(1, window=win) > 0)
                return win, data, valid.astype('uint8') * 255
            available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
            admitted = []
            times_by_worker = {}
            for n in workers:
                # Conservative admission control, not a hard operating-system limit.
                estimated_ram = n * size * size * (np.dtype(profile['dtype']).itemsize + 8) * 4
                if n > available_cpus or estimated_ram > psutil.virtual_memory().available // 4:
                    report['scenarios'].append({'workers': n, 'status': 'skipped', 'reason': 'CPU/RAM admission control'})
                    continue
                admitted.append(n)
                times_by_worker[n] = []
            rounds = worker_rounds(admitted, repeats) if admitted else []
            report['worker_order_by_round'] = rounds
            for iteration, order in enumerate(rounds, 1):
                for n in order:
                    masked = work / f'masked_w{n}_r{iteration}.tif'
                    cog = work / f'cog_w{n}_r{iteration}.tif'
                    with measure(f'application_w{n}_r{iteration}', rows, work):
                        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(masked, 'w', **profile) as dst:
                            with ThreadPoolExecutor(max_workers=n) as pool:
                                pending = collections.deque()
                                for win in windows(width, height, size):
                                    pending.append(pool.submit(read_window, win))
                                    if len(pending) >= n * 2:
                                        w, data, mask = pending.popleft().result()
                                        dst.write(data, 1, window=w); dst.write_mask(mask, window=w)
                                while pending:
                                    w, data, mask = pending.popleft().result()
                                    dst.write(data, 1, window=w); dst.write_mask(mask, window=w)
                    times_by_worker[n].append(rows[-1]['seconds'])
                    with measure(f'cog_w{n}_r{iteration}', rows, work):
                        rio_copy(masked, cog, driver='COG', compress='LZW', BIGTIFF='YES',
                                 NUM_THREADS=str(threads), BLOCKSIZE=size)
                    with measure(f'validation_w{n}_r{iteration}', rows, work):
                        with cog.open('rb') as stream:
                            if stream.read(4) not in (b'II+\x00', b'MM\x00+'):
                                raise ValueError('Output is not BigTIFF')
                        with rasterio.open(source_path) as src, rasterio.open(reference) as ref, rasterio.open(grid_path) as grid, rasterio.open(cog) as out:
                            if (src.width, src.height, src.crs, src.transform, src.dtypes) != (out.width, out.height, out.crs, out.transform, out.dtypes):
                                raise ValueError('Output grid or dtype differs')
                            if out.count != 1 or ColorInterp.alpha in out.colorinterp or MaskFlags.per_dataset not in out.mask_flag_enums[0]:
                                raise ValueError('Expected one band, no alpha, and a dataset mask')
                            if out.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG' or Path(str(cog) + '.msk').exists():
                                raise ValueError('Expected COG with internal mask')
                            for win in windows(width, height, size):
                                inside = ref.read(1, window=win) > 0
                                if not np.array_equal(inside, grid.read(1, window=win) > 0):
                                    raise ValueError('Cached mask differs from reference rasterization')
                                expected = inside & (src.dataset_mask(window=win) > 0)
                                if not np.array_equal(expected, out.dataset_mask(window=win) > 0):
                                    raise ValueError('Output mask differs from polygon AND original mask')
                                if not np.array_equal(src.read(1, window=win), out.read(1, window=win), equal_nan=True):
                                    raise ValueError('Output pixel values differ')
            for n in admitted:
                times = times_by_worker[n]
                median = statistics.median(times)
                report['scenarios'].append({'workers': n, 'status': 'ok', 'application_seconds': times,
                                            'median_application_seconds': median,
                                            'very_uncertain_projected_national_application_seconds':
                                                median * total_pixels / (width * height) if cfg['mode'] == 'subset' else None})
            report.update(status='ok' if any(x['status'] == 'ok' for x in report['scenarios']) else 'skipped',
                          pixels=width * height, national_pixels=total_pixels)
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (work / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        with (work / 'stages.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['stage', 'status', 'seconds', 'cpu_seconds', 'peak_process_rss_bytes', 'disk_free_bytes'])
            writer.writeheader(); writer.writerows(rows)
        print('Run directory:', work)
    return report, work


def run_chunks(config):
    """Optional local-only VRT/COG benchmark of pre-existing GCS TIFF chunks."""
    import subprocess
    from google.cloud import storage
    import numpy as np
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.windows import Window

    cfg = dict(config)
    if not cfg.get("confirm_chunks"):
        raise ValueError("Downloading and mosaicking chunks requires explicit confirmation")
    if cfg["label"] not in ("FREE", "PRO"):
        raise ValueError("Supply FREE/PRO and a gs:// prefix")
    if not shutil.which("gdalbuildvrt"):
        raise RuntimeError("gdalbuildvrt is required for the optional chunk test")
    threads = int(cfg["threads"])
    if threads < 1:
        raise ValueError("threads must be positive")
    run_id = cfg["label"].lower() + "-chunks-" + uuid.uuid4().hex
    work = Path(cfg["work_root"]) / run_id
    work.mkdir(parents=True, exist_ok=False)
    rows = []
    report = {"run_id": run_id, "config": cfg, "stages": rows,
              "warning": "VRT comparison does not establish geographic mask correctness; chunks may lack nodata. Reports contain URIs and potentially private errors. RSS/CPU cover the parent process only."}
    try:
        with measure("chunk_download", rows, work):
            objects = list_chunks(storage.Client(), cfg["prefix"], int(cfg["max_chunks"]))
            sizes = [int(x.size) for x in objects]
            if shutil.disk_usage(work).free < sum(sizes) * 4 + int(cfg["disk_reserve_gb"] * 1024**3):
                raise RuntimeError("Insufficient disk for approximate chunk estimate + reserve")
            local = []
            for i, obj in enumerate(objects):
                path = work / f"chunk-{i:06d}.tif"
                obj.download_to_filename(str(path))
                local.append(path)
        with measure("vrt", rows, work):
            vrt = work / "mosaic.vrt"
            manifest = work / "chunks.txt"
            manifest.write_text("".join(str(p.resolve()) + "\n" for p in local), encoding="utf-8")
            subprocess.run(["gdalbuildvrt", "-strict", "-input_file_list", str(manifest), str(vrt)],
                           check=True, capture_output=True, text=True)
        with measure("cog", rows, work):
            cog = work / "mosaic.tif"
            rio_copy(vrt, cog, driver="COG", compress="LZW", BIGTIFF="YES", NUM_THREADS=str(threads))
        with measure("validation_samples", rows, work):
            with rasterio.open(vrt) as src, rasterio.open(cog) as dst:
                if (src.width, src.height, src.count, src.crs, src.transform, src.dtypes) != (dst.width, dst.height, dst.count, dst.crs, dst.transform, dst.dtypes):
                    raise ValueError("VRT and output grid/bands differ")
                if dst.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
                    raise ValueError("Output is not a COG")
                for row in (0, max(0, src.height // 2 - 32), max(0, src.height - 64)):
                    for col in (0, max(0, src.width // 2 - 32), max(0, src.width - 64)):
                        win = Window(col, row, min(64, src.width-col), min(64, src.height-row))
                        if not np.array_equal(src.read(window=win), dst.read(window=win), equal_nan=True):
                            raise ValueError("COG differs from VRT in sampled pixels")
                        if not np.array_equal(src.dataset_mask(window=win), dst.dataset_mask(window=win)):
                            raise ValueError("COG differs from VRT in sampled mask")
        report.update(status="ok", chunk_count=len(objects), input_bytes=sum(sizes),
                      output_bytes=cog.stat().st_size, sampled_windows=9)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (work / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        with (work / "stages.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["stage", "status", "seconds", "cpu_seconds", "peak_process_rss_bytes", "disk_free_bytes"])
            writer.writeheader()
            writer.writerows(rows)
        print("Chunk run directory:", work)
    return report, work


def self_test():
    """Offline synthetic regression; run with `python benchmark_m2.py`."""
    import tempfile
    from contextlib import ExitStack
    from unittest.mock import patch
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    assert worker_rounds([1, 2, 4], 4) == [[1, 2, 4], [2, 4, 1], [4, 1, 2], [1, 2, 4]]
    from types import SimpleNamespace
    class ListingClient:
        def __init__(self, names):
            self.names = names
            self.calls = []
        def list_blobs(self, bucket, **kwargs):
            self.calls.append((bucket, kwargs))
            return iter(SimpleNamespace(name=name, size=10) for name in self.names[:kwargs["max_results"]])
    client = ListingClient(["batch/b.tif", "batch/a.TIFF", "batch/readme"])
    assert [x.name for x in list_chunks(client, "gs://bucket/batch/", 3)] == ["batch/a.TIFF", "batch/b.tif"]
    assert client.calls == [("bucket", {"prefix": "batch/", "max_results": 4, "page_size": 4})]
    for uri, limit, names in [("gs://bucket/", 2, []), ("gs://bucket", 2, []),
                              ("gs://bucket/batch", 0, []),
                              ("gs://bucket/batch", 2, ["a", "b", "c"])]:
        try:
            list_chunks(ListingClient(names), uri, limit)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid prefix or excess objects accepted")

    class Credentials:
        token = "offline"
        def refresh(self, request):
            pass

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, geometry = root / "source.tif", root / "geometry.json"
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(source, "w", driver="GTiff", width=48, height=48,
                               count=1, dtype="uint8", crs="EPSG:4326",
                               transform=from_origin(0, 48, 1, 1)) as dst:
                dst.write((np.arange(48 * 48) % 256).astype("uint8").reshape(48, 48), 1)
                mask = np.full((48, 48), 255, dtype="uint8")
                mask[0, 1] = 0
                dst.write_mask(mask)
        geometry.write_text(json.dumps({"type": "Polygon", "coordinates": [
            [[0, 0], [24, 0], [24, 48], [0, 48], [0, 0]]]}), encoding="utf-8")
        config = dict(gcs_uri="gs://offline/source.tif", label="FREE", mode="subset",
                      confirm_national=False, window_size=16, threads=1,
                      geometry_path=str(geometry), geometry_crs="EPSG:4326",
                      work_root=str(root / "runs"), subset_col=0, subset_row=0,
                      subset_width=48, subset_height=48, disk_reserve_gb=0, gdal_cache_mb=32,
                      window_workers=[1, 2, 4], repeats=2)
        original_open = rasterio.open
        def redirect(path, *args, **kwargs):
            return original_open(source if str(path).startswith("/vsigs/") else path, *args, **kwargs)
        import psutil
        with ExitStack() as stack:
            stack.enter_context(patch("google.auth.default", return_value=(Credentials(), None)))
            stack.enter_context(patch("rasterio.open", side_effect=redirect))
            stack.enter_context(patch.object(os, "cpu_count", return_value=4))
            stack.enter_context(patch.object(psutil, "virtual_memory", return_value=SimpleNamespace(available=2**30, total=2**30)))
            if hasattr(os, "sched_getaffinity"):
                stack.enter_context(patch.object(os, "sched_getaffinity", return_value={0, 1, 2, 3}))
            report, work = run(config)
        assert report["status"] == "ok"
        assert report["worker_order_by_round"] == [[1, 2, 4], [2, 4, 1]]
        saved = json.loads((work / "report.json").read_text(encoding="utf-8"))
        assert saved["worker_order_by_round"] == report["worker_order_by_round"]
        assert saved["status"] == "ok" and "confidentiality" in saved["security_notes"]
        with (work / "stages.csv").open(newline="", encoding="utf-8") as stream:
            stages = list(csv.DictReader(stream))
        assert len(stages) == len(report["stages"]) and all(x["status"] == "ok" for x in stages)
        assert [x["stage"] for x in stages if x["stage"].startswith("application_")] == [
            "application_w1_r1", "application_w2_r1", "application_w4_r1",
            "application_w2_r2", "application_w4_r2", "application_w1_r2"]
        assert [x["workers"] for x in report["scenarios"] if x["status"] == "ok"] == [1, 2, 4]
        for scenario in report["scenarios"]:
            if scenario["status"] == "ok":
                assert len(scenario["application_seconds"]) == 2
                for iteration in (1, 2):
                    with original_open(work / f"cog_w{scenario['workers']}_r{iteration}.tif") as output:
                        assert output.read(1)[0, 0] == 0 and output.dataset_mask()[0, 0] == 255
                        assert (output.dataset_mask() > 0).sum() == 1151
        try:
            run(dict(config, mode="national"))
        except ValueError as exc:
            assert "confirmation" in str(exc)
        else:
            raise AssertionError("National opt-in was not enforced")
        class FailedCredentials:
            def refresh(self, request):
                raise RuntimeError("offline failure")
        with patch("google.auth.default", return_value=(FailedCredentials(), None)):
            try:
                run(config)
            except RuntimeError as exc:
                assert str(exc) == "offline failure"
            else:
                raise AssertionError("Failure did not propagate")
        failed_dirs = [p for p in (root / "runs").iterdir() if p != work]
        assert len(failed_dirs) == 1
        failed = json.loads((failed_dirs[0] / "report.json").read_text(encoding="utf-8"))
        assert failed["status"] == "failed" and "offline failure" in failed["error"]
        assert (failed_dirs[0] / "stages.csv").exists()
    print("Offline self-test passed: zero, source mask, polygons, grid, COG and national gate.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "--national-stage":
        with open(sys.argv[4], encoding="utf-8") as stream:
            national_stage(sys.argv[2], sys.argv[3], json.load(stream))
    else:
        self_test()
