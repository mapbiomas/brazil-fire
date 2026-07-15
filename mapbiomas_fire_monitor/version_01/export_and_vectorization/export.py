import ee
from .state import IMAGE_COLLECTION, BUCKET, TILES_PREFIX, tile_pattern, _get_fs

EXPORT_FLAG = ""


def get_image_for_month(year, month):
    end_month = month + 1
    end_year = year
    if end_month > 12:
        end_month = 1
        end_year += 1
    start = f"{year}-{month:02d}-01"
    end = f"{end_year}-{end_month:02d}-01"
    filtered = ee.ImageCollection(IMAGE_COLLECTION).filterDate(start, end)
    if filtered.size().getInfo() == 0:
        return None
    return filtered.first()


def check_tiles_exist(year, month):
    fs = _get_fs()
    pattern = f"{BUCKET}/{TILES_PREFIX}/{tile_pattern(year, month)}*.tif"
    try:
        files = fs.glob(pattern)
        return len(files) > 0
    except Exception:
        return False


def start_export(year, month, logger=None):
    if check_tiles_exist(year, month):
        if logger:
            logger(f"[SKIP] Tiles for {year}_{month:02d} already exist in GCS.")
        return True

    image = get_image_for_month(year, month)
    if image is None:
        if logger:
            logger(f"[WARN] No image found for {year}_{month:02d} in ImageCollection.")
        return False

    prefix = tile_pattern(year, month)
    task_desc = f"{EXPORT_FLAG}MONITOR_EXPORT_{year}_{month:02d}"

    if logger:
        logger(f"[EXPORT] Starting export: {task_desc} -> gs://{BUCKET}/{TILES_PREFIX}/{prefix}_*.tif")

    task = ee.batch.Export.image.toCloudStorage(
        image=image.selfMask().toByte(),
        description=task_desc,
        bucket=BUCKET,
        fileNamePrefix=f"{TILES_PREFIX}/{prefix}_",
        scale=30,
        region=image.geometry().bounds(),
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()

    if logger:
        logger(f"[EXPORT] Task submitted: {task_desc}")

    return task


def export_selected(ui, logger=None):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[EXPORT] Nenhum mes selecionado.", "warning")
        return

    if logger:
        logger(f"[EXPORT] Iniciando export de {len(selected)} meses...", "info")

    for year, month in selected:
        start_export(year, month, logger=logger)

    if logger:
        logger("[EXPORT] Todos os exports foram submetidos. Aguarde as tasks do GEE finalizarem, depois clique em Sincronizar.", "success")

    ui.sync()
