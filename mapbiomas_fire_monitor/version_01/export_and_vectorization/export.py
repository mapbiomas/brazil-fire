import datetime
import ee
from . import config
from . import selection
from .state import _get_fs

EXPORT_FLAG = ""


def _resolve_ic_image(assetid, unit_key, kind):
    """Resolve a imagem de uma ImageCollection cuja unidade derivada == unit_key."""
    col = ee.ImageCollection(assetid)
    try:
        idx = col.aggregate_array("system:index").getInfo() or []
        ts = col.aggregate_array("system:time_start").getInfo() or []
    except Exception:
        return None
    target = None
    for i, t in zip(idx, ts):
        if t and config.unit_key_for_image(kind, t) == unit_key:
            target = i
            break
    if target is None:
        return None
    return col.filter(ee.Filter.eq("system:index", target)).first()


def get_image_for_unit(unit_key, context=None):
    """Retorna o ee.Image da unidade: banda (multibanda) ou imagem (IC)."""
    ctx = context or config.processing_context()
    assetid = ctx["assetid"]
    try:
        atype = ee.data.getAsset(assetid).get("type")
    except Exception:
        atype = None
    if atype == "IMAGE":
        return ee.Image(assetid).select(unit_key).rename(unit_key)
    return _resolve_ic_image(assetid, unit_key,
                             ctx.get("kind") or config.product_kind(ctx["product"]))


def count_tiles(unit_key, context=None):
    fs = _get_fs()
    pattern = (f"{config.BUCKET}/{config.tiles_prefix(context)}/"
               f"{config.tile_pattern_unit(unit_key, context)}*.tif")
    try:
        return len(fs.glob(pattern))
    except Exception:
        return 0


def check_tiles_exist(unit_key, context=None):
    return count_tiles(unit_key, context)


def delete_tiles(unit_key, logger=None, context=None):
    fs = _get_fs()
    pattern = (f"{config.BUCKET}/{config.tiles_prefix(context)}/"
               f"{config.tile_pattern_unit(unit_key, context)}*.tif")
    deleted = 0
    try:
        for t in fs.glob(pattern):
            fs.rm(t)
            deleted += 1
    except Exception as e:
        if logger:
            logger(f"[ERROR] Failed to delete tiles of {unit_key}: {e}")
        return 0
    if logger:
        logger(f"[EXPORT] {deleted} tile(s) removed from temp/ for {unit_key}.")
    return deleted


def start_export(unit_key, force=False, logger=None, context=None):
    ctx = context or config.processing_context()

    if check_tiles_exist(unit_key, ctx):
        if force:
            if logger:
                logger(f"[EXPORT] {unit_key}: force=True — removing tiles and re-exporting.")
            delete_tiles(unit_key, logger=logger, context=ctx)
        else:
            if logger:
                n = count_tiles(unit_key, ctx)
                if n == 0:
                    logger(f"[SKIP] Tiles for {unit_key} already exist in GCS.")
                else:
                    logger(f"[WARN] {unit_key}: {n} tile(s) already in GCS. "
                           "Export may be incomplete. Use force=True to redo.")
            return True

    image = get_image_for_unit(unit_key, ctx)
    if image is None:
        if logger:
            logger(f"[WARN] No image found for unit '{unit_key}'.")
        return False

    # Exporta a imagem nativa (sem toByte/selfMask — o mosaico converte via -ot).
    bounds = image.geometry().bounds()
    if logger:
        try:
            info = bounds.getInfo()
            coords = info["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            area_km2 = (max(xs) - min(xs)) * (max(ys) - min(ys)) * 111.32 * 111.32
            logger(f"[EXPORT] Bounds area approx: {area_km2:,.0f} km² — check it is not a world-wide bbox.")
        except Exception:
            pass

    prefix = config.tile_pattern_unit(unit_key, ctx)
    tag = config._sanitize(f"{ctx['product']}_{ctx['storage_country']}_{unit_key}")
    task_desc = f"{EXPORT_FLAG}MONITOR_EXPORT_{tag}"

    if logger:
        logger(f"[EXPORT] Starting export: {task_desc} -> gs://{config.BUCKET}/{config.tiles_prefix(ctx)}/{prefix}_*.tif")

    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=task_desc,
        bucket=config.BUCKET,
        fileNamePrefix=f"{config.tiles_prefix(ctx)}/{prefix}_",
        scale=config.scale(ctx),
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
    items = selection.collect_items(ui)
    if not items:
        if logger:
            logger("[EXPORT] No unit selected.", "warning")
        return

    groups = selection.group_items(items)

    if logger:
        logger(f"[EXPORT] Starting export of {len(items)} unit(s) "
               f"in {len(groups)} product(s)...", "info")

    for key, units in sorted(groups.items()):
        country, theme, collection, product = key
        ctx = config.processing_context(country, theme, collection, product)
        if logger and len(groups) > 1:
            logger(f"[EXPORT] {country}/{theme}/{collection}/{product}: {len(units)} unit(s)")
        for unit in units:
            start_export(unit, force=force, logger=logger, context=ctx)

    if logger:
        logger("[EXPORT] All exports submitted. Wait for the GEE tasks to finish, then click Sync.", "success")

    selection.sync_affected(ui, groups.keys())
