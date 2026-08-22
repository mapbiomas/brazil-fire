import os
import shlex
import subprocess
import time
import gc
import shutil
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import geopandas as gpd
from . import config
from .state import _get_fs


def check_vector_gcs_exists(year, month):
    fs = _get_fs()
    path = f"{config.BUCKET}/{config.vector_prefix()}/{config.vector_name(year, month)}.zip"
    try:
        return fs.exists(path)
    except Exception:
        return False


def check_vector_gee_exists(year, month):
    import ee
    asset_id = f"{config.vector_asset_prefix()}/{config.vector_name(year, month)}"
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def vectorize_month(year, month, force=False, logger=None):
    if check_vector_gcs_exists(year, month):
        if force:
            if logger:
                logger(f"[VECTORIZE] {year}_{month:02d}: force=True — removing ZIP and re-vectorizing.")
            try:
                _get_fs().rm(f"{config.BUCKET}/{config.vector_prefix()}/{config.vector_name(year, month)}.zip")
            except Exception as e:
                if logger:
                    logger(f"[ERROR] Failed to delete ZIP of {year}_{month:02d}: {e}")
        else:
            if logger:
                logger(f"[SKIP] Vector for {year}_{month:02d} already exists in GCS.")
            return True

    mosaic_path = f"{config.mosaic_prefix()}/{config.mosaic_name(year, month)}.tif"
    fs = _get_fs()
    if not fs.exists(f"{config.BUCKET}/{mosaic_path}"):
        if logger:
            logger(f"[WARN] Mosaic not found for {year}_{month:02d}.")
        return False

    work_dir = f"/content/temp/vectorize_{year}_{month:02d}_{int(time.time())}"
    os.makedirs(work_dir, exist_ok=True)

    local_raster = os.path.join(work_dir, config.mosaic_name(year, month) + ".tif")
    local_vector = os.path.join(work_dir, config.vector_name(year, month))

    try:
        if logger:
            logger(f"[DOWNLOAD] gs://{config.BUCKET}/{mosaic_path} -> {local_raster}")

        remote_path = f"{config.BUCKET}/{mosaic_path}"
        fs.get(remote_path, local_raster)
        if not os.path.exists(local_raster):
            raise RuntimeError("Download via gcsfs failed.")

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
            logger("[ZIP] Compacting shapefile...")

        zip_path = f"{local_vector}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
                p = f"{local_vector}{ext}"
                if os.path.exists(p):
                    zf.write(p, arcname=os.path.basename(p))

        if logger:
            logger(f"[UPLOAD] Uploading zip to GCS...")

        dest = f"{config.BUCKET}/{config.vector_prefix()}/{config.vector_name(year, month)}.zip"
        fs.put(zip_path, dest)
        if logger:
            logger(f"[OK] gs://{dest}")

        return True
    except Exception as e:
        if logger:
            logger(f"[ERROR] Vectorization failed: {e}")
        return False
    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        gc.collect()


def _has_active_upload(asset_id):
    import ee
    try:
        for t in ee.data.getTaskList():
            desc = t.get("description") or ""
            if asset_id in desc and t.get("state") in ("READY", "RUNNING", "PENDING"):
                return True
    except Exception:
        pass
    return False


def _ensure_folder(folder, logger=None):
    """Cria a pasta no GEE se nao existir e deixa a ACL publica (all_users_can_read)."""
    import ee
    try:
        ee.data.getAsset(folder)
    except Exception:
        if logger:
            logger(f"[GEE FOLDER] Creating folder: {folder}")
        try:
            ee.data.createAsset({'type': 'Folder', 'name': folder}, folder)
        except Exception as e:
            if logger:
                logger(f"[WARN] Could not create folder {folder}: {e}")
    try:
        ee.data.setAssetAcl(folder, {'all_users_can_read': True})
        if logger:
            logger(f"[GEE FOLDER] Public ACL ensured on: {folder}")
    except Exception as e:
        if logger:
            logger(f"[WARN] Could not set public ACL on {folder}: {e}")


def make_vectors_public(logger=None):
    """Seta all_users_can_read=True em todos os assets da pasta de vetores (idempotente).

    Rode apos as tasks de upload concluirem para garantir a visibilidade por asset
    (a pasta ja herda publica para os novos).
    """
    import ee
    folder = config.vector_asset_prefix()
    _ensure_folder(folder, logger=logger)
    _log = logger or (lambda *_: None)
    changed = 0
    try:
        assets = ee.data.listAssets({"parent": folder})
        page_token = assets.get("nextPageToken")
        to_scan = list(assets.get("assets", []))
        while page_token:
            assets = ee.data.listAssets({"parent": folder, "pageToken": page_token})
            to_scan.extend(assets.get("assets", []))
            page_token = assets.get("nextPageToken")
    except Exception as e:
        _log(f"[WARN] Listing assets of {folder}: {e}")
        return 0

    for a in to_scan:
        asset_id = a.get("name")
        try:
            ee.data.setAssetAcl(asset_id, {'all_users_can_read': True})
            changed += 1
            _log(f"[PUBLIC] {asset_id.split('/')[-1]}")
        except Exception as e:
            _log(f"[WARN] ACL failed on {asset_id}: {e}")

    _log(f"[PUBLIC] {changed} assets made public (all_users_can_read).", "success")
    return changed


def _run_upload(asset_id, source, logger=None):
    cmd = [
        "earthengine",
        f"--project={config.GEE_PROJECT}",
        "upload", "table",
        f"--asset_id={asset_id}",
        source,
    ]
    if logger:
        logger(f"[UPLOAD GEE] {asset_id} ({source})")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        if logger:
            logger(f"[OK] Upload submitted: {asset_id}")
        return True

    if logger:
        err = (result.stderr or result.stdout or "").strip()
        logger(f"[ERROR] GEE upload failed (exit {result.returncode}): {err}")
    return False


def _fallback_upload(asset_id, zip_remote, logger=None):
    """Se a CLI nao aceitar o .zip, extrai localmente e envia o .shp."""
    work_dir = tempfile.mkdtemp(prefix="gee_upload_")
    try:
        fs = _get_fs()
        zip_local = os.path.join(work_dir, "vector.zip")
        fs.get(zip_remote.replace("gs://", ""), zip_local)
        with zipfile.ZipFile(zip_local) as zf:
            zf.extractall(work_dir)
        shp_files = [p for p in os.listdir(work_dir) if p.endswith(".shp")]
        if not shp_files:
            if logger:
                logger("[ERROR] Fallback upload: no .shp inside the zip.")
            return False
        return _run_upload(asset_id, os.path.join(work_dir, shp_files[0]), logger)
    except Exception as e:
        if logger:
            logger(f"[ERROR] Fallback upload failed: {e}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def upload_to_gee(year, month, force=False, logger=None):
    import ee
    if check_vector_gee_exists(year, month):
        if force:
            if logger:
                logger(f"[UPLOAD GEE] {year}_{month:02d}: force=True — removing asset and re-uploading.")
            try:
                asset_id_existing = f"{config.vector_asset_prefix()}/{config.vector_name(year, month)}"
                ee.data.deleteAsset(asset_id_existing)
            except Exception as e:
                if logger:
                    logger(f"[WARN] Failed to delete existing asset: {e}")
        else:
            if logger:
                logger(f"[SKIP] Asset already in GEE for {year}_{month:02d}.")
            return True

    if not check_vector_gcs_exists(year, month):
        if logger:
            logger(f"[WARN] Vector not in GCS for {year}_{month:02d}. Vectorize first.")
        return False

    asset_id = f"{config.vector_asset_prefix()}/{config.vector_name(year, month)}"

    if _has_active_upload(asset_id):
        if logger:
            logger(f"[WARN] Upload already in progress for {asset_id}. Wait for it to finish.")
        return False

    _ensure_folder(config.vector_asset_prefix(), logger=logger)

    zip_remote = f"gs://{config.BUCKET}/{config.vector_prefix()}/{config.vector_name(year, month)}.zip"

    if _run_upload(asset_id, zip_remote, logger):
        return True

    if logger:
        logger("[UPLOAD GEE] Trying fallback with locally extracted .shp...")
    return _fallback_upload(asset_id, zip_remote, logger)


def _check_mosaic_gcs(year, month):
    path = f"{config.BUCKET}/{config.mosaic_prefix()}/{config.mosaic_name(year, month)}.tif"
    try:
        return _get_fs().exists(path)
    except Exception:
        return False


def vectorize_selected(ui, logger=None, force=False):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[VECTORIZE] No month selected.", "warning")
        return

    # Cap de workers evita OOM no Colab com varios polygonize simultaneos.
    workers = min(os.cpu_count() or 4, 4)

    def _process(ym):
        y, m = ym
        if not _check_mosaic_gcs(y, m):
            return f"[SKIP] {y}_{m:02d} — mosaic not found in GCS"
        ok = vectorize_month(y, m, force=force, logger=None)
        return f"[{'OK' if ok else 'FAIL'}] {y}_{m:02d}"

    if logger:
        logger(f"[VECTORIZE] Starting vectorization of {len(selected)} months ({workers} workers)...", "info")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, ym): ym for ym in selected}
        for f in as_completed(futures):
            if logger:
                logger(f.result())

    if logger:
        logger("[VECTORIZE] Done. Click Sync to update the grid.", "success")

    ui.sync()


def gee_upload_selected(ui, logger=None, force=False):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[GEE UPLOAD] No month selected.", "warning")
        return

    if logger:
        logger(f"[GEE UPLOAD] Starting upload of {len(selected)} months to GEE...", "info")

    for year, month in selected:
        upload_to_gee(year, month, force=force, logger=logger)

    # Visibilidade publica: pasta ja e publica; garante tambem por asset.
    try:
        make_vectors_public(logger=logger)
    except Exception as e:
        if logger:
            logger(f"[WARN] make_vectors_public failed: {e}")

    if logger:
        logger("[GEE UPLOAD] Done. After GEE tasks finish, run make_vectors_public to ensure per-asset ACL. Click Sync.", "success")

    ui.sync()
