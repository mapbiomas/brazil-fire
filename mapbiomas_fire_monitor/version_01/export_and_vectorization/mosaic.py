import os
import subprocess
import time
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config
from . import selection
from .state import _get_fs


def list_tiles(unit_key, context=None):
    fs = _get_fs()
    pattern = (f"{config.BUCKET}/{config.tiles_prefix(context)}/"
               f"{config.tile_pattern_unit(unit_key, context)}*.tif")
    try:
        files = fs.glob(pattern)
        return sorted(files)
    except Exception:
        return []


def check_mosaic_exists(unit_key, context=None):
    fs = _get_fs()
    path = (f"{config.BUCKET}/{config.mosaic_prefix(context)}/"
            f"{config.mosaic_name_unit(unit_key, context)}.tif")
    try:
        return fs.exists(path)
    except Exception:
        return False


def assemble_mosaic(unit_key, force=False, logger=None, context=None):
    ctx = context or config.processing_context()
    # Nunca apaga o COG se nao houver como reconstruir.
    tiles = list_tiles(unit_key, ctx)
    if not tiles:
        if check_mosaic_exists(unit_key, ctx) and not force:
            if logger:
                logger(f"[SKIP] Mosaic for {unit_key} already exists.")
            return True
        if logger:
            logger(f"[WARN] No tiles found for {unit_key}. Cannot rebuild mosaic.")
        return False

    if check_mosaic_exists(unit_key, ctx):
        if force:
            if logger:
                logger(f"[MOSAIC] {unit_key}: force=True — removing COG and rebuilding.")
                logger("[MOSAIC] Note: if the COG was already published to the public bucket, "
                       "re-run Step 3 (Public Mosaic) with FORCE_PUBLISH_MOSAIC=True to update it.")
            try:
                _get_fs().rm(f"{config.BUCKET}/{config.mosaic_prefix(ctx)}/"
                             f"{config.mosaic_name_unit(unit_key, ctx)}.tif")
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Failed to delete COG of {unit_key}: {e}")
        else:
            if logger:
                logger(f"[SKIP] Mosaic for {unit_key} already exists.")
            return True

    if logger:
        logger(f"[MOSAIC] Assembling {len(tiles)} tiles for {unit_key}...")

    vsigs_files = [f"/vsigs/{f}" for f in tiles]

    work_dir = f"/content/temp/mosaic_{unit_key.replace('/', '_')}_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)

    input_list = os.path.join(work_dir, "input_files.txt")
    vrt_file = os.path.join(work_dir, "mosaic.vrt")
    output_file = os.path.join(work_dir, config.mosaic_name_unit(unit_key, ctx) + ".tif")

    try:
        with open(input_list, "w") as f:
            for path in vsigs_files:
                f.write(f"{path}\n")

        build_cmd = ["gdalbuildvrt", "-input_file_list", input_list, vrt_file]
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if logger:
                logger(f"[ERROR] gdalbuildvrt failed: {result.stderr}")
            return False

        save = config.save_options(ctx)
        translate_cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-ot", save["ot"],
            "-a_nodata", str(save["nodata"]),
            "-co", "TILED=YES",
            "-co", f"COMPRESS={save['compression']}",
            "-co", f"PREDICTOR={save['predictor']}",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-co", "BIGTIFF=YES",
            vrt_file,
            output_file,
        ]
        result = subprocess.run(translate_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if logger:
                logger(f"[ERROR] gdal_translate failed: {result.stderr}")
            return False

        fs = _get_fs()
        dest = (f"{config.BUCKET}/{config.mosaic_prefix(ctx)}/"
                f"{config.mosaic_name_unit(unit_key, ctx)}.tif")
        fs.put(output_file, dest)
        if logger:
            logger(f"[OK] Mosaic ({len(tiles)} tiles) uploaded to gs://{dest}")

        return True
    except Exception as e:
        if logger:
            logger(f"[ERROR] Mosaic assembly failed: {e}")
        return False
    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def mosaic_selected(ui, logger=None, force=False):
    items = selection.collect_items(ui)
    if not items:
        if logger:
            logger("[MOSAIC] No unit selected.", "warning")
        return

    workers = min(os.cpu_count() or 4, 4)

    jobs, seen_ctx = selection.build_jobs(items)

    def _process(args):
        unit, ctx = args
        if not list_tiles(unit, ctx):
            return f"[SKIP] {unit} — no tiles in GCS"
        ok = assemble_mosaic(unit, force=force, logger=None, context=ctx)
        return f"[{'OK' if ok else 'FAIL'}] {unit}"

    if logger:
        n_products = len(seen_ctx)
        logger(f"[MOSAIC] Starting mosaic of {len(jobs)} unit(s) "
               f"in {n_products} product(s) ({workers} workers)...", "info")

    verbose = bool(getattr(config, "LOG_VERBOSE", False))
    counts = {"OK": 0, "SKIP": 0, "FAIL": 0}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, j): j[0] for j in jobs}
        for f in as_completed(futures):
            res = f.result()
            if res.startswith("[") and "]" in res:
                tag = res[1:res.index("]")]
                if tag in counts:
                    counts[tag] += 1
            if logger and (verbose or not res.startswith("[SKIP]")):
                logger(res)

    if logger:
        logger(f"[MOSAIC] Done: {counts['OK']} ok, {counts['SKIP']} skipped, "
               f"{counts['FAIL']} failed ({len(jobs)} units). "
               "Click Sync to update the grid.", "success")

    selection.sync_affected(ui, seen_ctx.keys())
