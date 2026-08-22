import ee
from . import config
from .state import _get_fs

EXPORT_FLAG = ""


def get_image_for_month(year, month):
    end_month = month + 1
    end_year = year
    if end_month > 12:
        end_month = 1
        end_year += 1
    start = f"{year}-{month:02d}-01"
    end = f"{end_year}-{end_month:02d}-01"
    filtered = ee.ImageCollection(config.image_collection()).filterDate(start, end)
    if filtered.size().getInfo() == 0:
        return None
    return filtered.first()


def count_tiles(year, month):
    fs = _get_fs()
    pattern = f"{config.BUCKET}/{config.tiles_prefix()}/{config.tile_pattern(year, month)}*.tif"
    try:
        return len(fs.glob(pattern))
    except Exception:
        return 0


def check_tiles_exist(year, month):
    return count_tiles(year, month) > 0


def delete_tiles(year, month, logger=None):
    fs = _get_fs()
    pattern = f"{config.BUCKET}/{config.tiles_prefix()}/{config.tile_pattern(year, month)}*.tif"
    deleted = 0
    try:
        for t in fs.glob(pattern):
            fs.rm(t)
            deleted += 1
    except Exception as e:
        if logger:
            logger(f"[ERROR] Falha ao apagar tiles de {year}_{month:02d}: {e}")
        return 0
    if logger:
        logger(f"[EXPORT] {deleted} tile(s) excluidos de temp/ para {year}_{month:02d}.")
    return deleted


def start_export(year, month, force=False, logger=None):
    if check_tiles_exist(year, month):
        if force:
            if logger:
                logger(f"[EXPORT] {year}_{month:02d}: force=True — excluindo tiles e reexportando.")
            delete_tiles(year, month, logger=logger)
        else:
            if logger:
                n = count_tiles(year, month)
                if n == 0:
                    logger(f"[SKIP] Tiles for {year}_{month:02d} already exist in GCS.")
                else:
                    logger(f"[WARN] {year}_{month:02d}: {n} tile(s) ja no GCS. Export pode estar incompleto. Use force=True para refazer.")
            return True

    image = get_image_for_month(year, month)
    if image is None:
        if logger:
            logger(f"[WARN] No image found for {year}_{month:02d} in ImageCollection.")
        return False

    # Exporta Byte puro 0/1 (sem selfMask/nodata) — 1 banda, maxima compressao.
    bounds = image.geometry().bounds()
    if logger:
        try:
            info = bounds.getInfo()
            coords = info["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            area_km2 = (max(xs) - min(xs)) * (max(ys) - min(ys)) * 111.32 * 111.32
            logger(f"[EXPORT] Bounds area aprox: {area_km2:,.0f} km² — verifique se nao e bbox mundial.")
        except Exception:
            pass

    prefix = config.tile_pattern(year, month)
    task_desc = f"{EXPORT_FLAG}MONITOR_EXPORT_{year}_{month:02d}"

    if logger:
        logger(f"[EXPORT] Starting export: {task_desc} -> gs://{config.BUCKET}/{config.tiles_prefix()}/{prefix}_*.tif")

    task = ee.batch.Export.image.toCloudStorage(
        image=image.toByte(),
        description=task_desc,
        bucket=config.BUCKET,
        fileNamePrefix=f"{config.tiles_prefix()}/{prefix}_",
        scale=config.scale(),
        region=bounds,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()

    if logger:
        logger(f"[EXPORT] Task submitted: {task_desc}")

    return task


def export_selected(ui, logger=None, force=False):
    selected = ui.get_selected_months()
    if not selected:
        if logger:
            logger("[EXPORT] Nenhum mes selecionado.", "warning")
        return

    if logger:
        logger(f"[EXPORT] Iniciando export de {len(selected)} meses...", "info")

    for year, month in selected:
        start_export(year, month, force=force, logger=logger)

    if logger:
        logger("[EXPORT] Todos os exports foram submetidos. Aguarde as tasks do GEE finalizarem, depois clique em Sincronizar.", "success")

    ui.sync()
