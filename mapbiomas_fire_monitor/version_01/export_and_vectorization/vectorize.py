import os
import subprocess
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
from .state import BUCKET, GEE_PROJECT, MOSAIC_PREFIX, VECTOR_PREFIX, VECTOR_ASSET_PREFIX, mosaic_name, vector_name, _get_fs


def check_vector_gcs_exists(year, month):
    fs = _get_fs()
    path = f"{BUCKET}/{VECTOR_PREFIX}/{vector_name(year, month)}.shp"
    try:
        return fs.exists(path)
    except Exception:
        return False


def check_vector_gee_exists(year, month):
    import ee
    asset_id = f"{VECTOR_ASSET_PREFIX}/{vector_name(year, month)}"
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def vectorize_month(year, month, logger=None):
    if check_vector_gcs_exists(year, month):
        if logger:
            logger(f"[SKIP] Vector for {year}_{month:02d} already exists in GCS.")
        return True

    mosaic_path = f"{MOSAIC_PREFIX}/{mosaic_name(year, month)}.tif"
    fs = _get_fs()
    if not fs.exists(f"{BUCKET}/{mosaic_path}"):
        if logger:
            logger(f"[WARN] Mosaic not found for {year}_{month:02d}.")
        return False

    work_dir = f"/content/temp/vectorize_{year}_{month:02d}_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)

    local_raster = os.path.join(work_dir, mosaic_name(year, month) + ".tif")
    local_vector = os.path.join(work_dir, vector_name(year, month))

    try:
        if logger:
            logger(f"[DOWNLOAD] gs://{BUCKET}/{mosaic_path} -> {local_raster}")

        remote_path = f"gs://{BUCKET}/{mosaic_path}"
        download_cmd = ["gcloud", "storage", "cp", remote_path, local_raster]
        result = subprocess.run(download_cmd, capture_output=True, text=True)
        if result.returncode != 0 and not os.path.exists(local_raster):
            raise RuntimeError(f"Download failed: {result.stderr}")

        if logger:
            logger(f"[POLYGONIZE] {local_raster} -> {local_vector}.shp")

        poly_cmd = [
            "gdal_polygonize.py",
            local_raster,
            "-b", "1",
            "-mask", local_raster,
            "-f", "ESRI Shapefile",
            f"{local_vector}.shp",
        ]
        result = subprocess.run(poly_cmd, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(f"{local_vector}.shp"):
            raise RuntimeError(f"Polygonize failed: {result.stderr}")

        if logger:
            logger("[UNIQUE_ID] Adding unique_id column...")

        gdf = gpd.read_file(f"{local_vector}.shp")
        gdf["unique_id"] = range(1, len(gdf) + 1)
        gdf.to_file(f"{local_vector}.shp", driver="ESRI Shapefile")

        if logger:
            logger(f"[UPLOAD] Uploading shapefile to GCS...")

        for ext in [".shp", ".shx", ".dbf", ".prj"]:
            local_file = f"{local_vector}{ext}"
            if os.path.exists(local_file):
                dest = f"{BUCKET}/{VECTOR_PREFIX}/{vector_name(year, month)}{ext}"
                fs.put(local_file, dest)
                if logger:
                    logger(f"[OK] gs://{dest}")

        return True
    except Exception as e:
        if logger:
            logger(f"[ERROR] Vectorization failed: {e}")
        return False
    finally:
        import shutil
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def upload_to_gee(year, month, logger=None):
    if check_vector_gee_exists(year, month):
        if logger:
            logger(f"[SKIP] Asset already in GEE for {year}_{month:02d}.")
        return True

    if not check_vector_gcs_exists(year, month):
        if logger:
            logger(f"[WARN] Vector not in GCS for {year}_{month:02d}. Vectorize first.")
        return False

    asset_id = f"{VECTOR_ASSET_PREFIX}/{vector_name(year, month)}"
    cmd = (
        f"earthengine --project={GEE_PROJECT} upload table "
        f"--asset_id={asset_id} "
        f"gs://{BUCKET}/{VECTOR_PREFIX}/{vector_name(year, month)}.shp"
    )

    if logger:
        logger(f"[UPLOAD GEE] {asset_id}")

    result = os.system(cmd)
    if result == 0:
        if logger:
            logger(f"[OK] Uploaded to GEE: {asset_id}")
        return True
    else:
        if logger:
            logger(f"[ERROR] GEE upload failed (exit code {result})")
        return False


def _check_mosaic_gcs(year, month):
    from .state import _get_fs as _fs
    path = f"{BUCKET}/{MOSAIC_PREFIX}/{mosaic_name(year, month)}.tif"
    try:
        return _fs().exists(path)
    except Exception:
        return False


def vectorize_selected(ui, logger=None):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[VECTORIZE] Nenhum mes selecionado.", "warning")
        return

    workers = os.cpu_count() or 4

    def _process(ym):
        y, m = ym
        if not _check_mosaic_gcs(y, m):
            return f"[SKIP] {y}_{m:02d} — mosaico nao encontrado no GCS"
        ok = vectorize_month(y, m, logger=None)
        return f"{'[OK]' if ok else '[FAIL]'} {y}_{m:02d}"

    if logger:
        logger(f"[VECTORIZE] Iniciando vetorizacao de {len(selected)} meses ({workers} workers)...", "info")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, ym): ym for ym in selected}
        for f in as_completed(futures):
            if logger:
                logger(f.result())

    if logger:
        logger("[VECTORIZE] Concluido. Clique em Sincronizar para atualizar a grid.", "success")

    ui.sync()


def gee_upload_selected(ui, logger=None):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[GEE UPLOAD] Nenhum mes selecionado.", "warning")
        return

    if logger:
        logger(f"[GEE UPLOAD] Iniciando upload de {len(selected)} meses para o GEE...", "info")

    for year, month in selected:
        upload_to_gee(year, month, logger=logger)

    if logger:
        logger("[GEE UPLOAD] Concluido. Clique em Sincronizar para atualizar a grid.", "success")

    ui.sync()
