import ee
from state import IMAGE_COLLECTION, BUCKET, GEE_PROJECT, TILES_PREFIX, tile_pattern, _get_fs


def get_image_for_month(year, month):
    col = ee.ImageCollection(IMAGE_COLLECTION)
    date_str = f"{year}_{month:02d}"
    filtered = col.filter(ee.Filter.eq("system:index", date_str))
    count = filtered.size().getInfo()
    if count == 0:
        filtered = col.filterDate(f"{year}-{month:02d}-01", f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01")
        count2 = filtered.size().getInfo()
        if count2 == 0:
            return None
    return filtered.mosaic()


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
    task_desc = f"MONITOR_EXPORT_{year}_{month:02d}"

    if logger:
        logger(f"[EXPORT] Starting export: {task_desc} -> gs://{BUCKET}/{TILES_PREFIX}/{prefix}_*.tif")

    task = ee.batch.Export.image.toCloudStorage(
        image=image.toByte(),
        description=task_desc,
        bucket=BUCKET,
        fileNamePrefix=f"{TILES_PREFIX}/{prefix}_",
        scale=30,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
    )
    task.start()

    if logger:
        logger(f"[EXPORT] Task submitted: {task_desc}")

    return task
