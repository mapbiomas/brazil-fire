"""Isolated, read-only GCS benchmark of the M2 country-mask contract."""
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
    size, threads = int(cfg['window_size']), int(cfg['threads'])
    workers = cfg.get('window_workers', [1, 2, 4])
    repeats = int(cfg.get('repeats', 1))
    if size < 16 or size > 4096 or size % 16 or threads < 1 or repeats < 1 or repeats > 10:
        raise ValueError('Invalid window size, GDAL threads or repeats')
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
    self_test()
