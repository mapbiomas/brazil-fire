import os
import subprocess
import time
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import config
from .state import _get_fs


def list_tiles(year, month):
    fs = _get_fs()
    pattern = f"{config.BUCKET}/{config.tiles_prefix()}/{config.tile_pattern(year, month)}*.tif"
    try:
        files = fs.glob(pattern)
        return sorted(files)
    except Exception:
        return []


def check_mosaic_exists(year, month):
    fs = _get_fs()
    path = f"{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(year, month)}.tif"
    try:
        return fs.exists(path)
    except Exception:
        return False


def assemble_mosaic(year, month, force=False, logger=None):
    if check_mosaic_exists(year, month):
        if force:
            if logger:
                logger(f"[MOSAIC] {year}_{month:02d}: force=True — removing COG and rebuilding.")
            try:
                _get_fs().rm(f"{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(year, month)}.tif")
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Failed to delete COG of {year}_{month:02d}: {e}")
        else:
            if logger:
                logger(f"[SKIP] Mosaic for {year}_{month:02d} already exists.")
            return True

    tiles = list_tiles(year, month)
    if not tiles:
        if logger:
            logger(f"[WARN] No tiles found for {year}_{month:02d}.")
        return False

    if logger:
        logger(f"[MOSAIC] Assembling {len(tiles)} tiles for {year}_{month:02d}...")

    vsigs_files = [f"/vsigs/{f}" for f in tiles]

    work_dir = f"/content/temp/mosaic_{year}_{month:02d}_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)

    input_list = os.path.join(work_dir, "input_files.txt")
    vrt_file = os.path.join(work_dir, f"mosaic_{year}_{month:02d}.vrt")
    output_file = os.path.join(work_dir, config.mosaic_name(year, month) + ".tif")

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

        # Saida Byte 0/1 com 0 = nodata (-a_nodata 0): tiles 100% oceano ficam
        # quase de graca via mascara interna do COG. DEFLATE comprime melhor que
        # LZW em dados 0/1 -> mosaicos bem menores.
        translate_cmd = [
            "gdal_translate",
            "-of", "GTiff",
            "-ot", "Byte",
            "-a_nodata", "0",
            "-co", "TILED=YES",
            "-co", "COMPRESS=DEFLATE",
            "-co", "ZLEVEL=9",
            "-co", "PREDICTOR=2",
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
        dest = f"{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(year, month)}.tif"
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
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[MOSAIC] No month selected.", "warning")
        return

    # Cap de workers evita OOM no Colab com varios gdal_translate simultaneos.
    workers = min(os.cpu_count() or 4, 4)

    def _process(ym):
        y, m = ym
        if not list_tiles(y, m):
            return f"[SKIP] {y}_{m:02d} — no tiles in GCS"
        ok = assemble_mosaic(y, m, force=force, logger=None)
        return f"[{'OK' if ok else 'FAIL'}] {y}_{m:02d}"

    if logger:
        logger(f"[MOSAIC] Starting mosaic of {len(selected)} months ({workers} workers)...", "info")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, ym): ym for ym in selected}
        for f in as_completed(futures):
            if logger:
                logger(f.result())

    if logger:
        logger("[MOSAIC] Done. Click Sync to update the grid.", "success")

    ui.sync()
