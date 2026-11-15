"""
U-Net experiment - publish the classified maps to Google Earth Engine.

Same approach as the production catalog M6 (``M6_publisher.upload_to_gee``):
the ``earthengine upload image`` CLI ingests the COG straight from GCS into an
ImageCollection, one image per region and period, with time and model
properties. An asset that already exists is replaced.

    {GEE_PREFIX}/{CAMPAIGN}/{EXPERIMENT}/{biome}/{version}/{region}_{period}
    e.g. projects/mapbiomas-fire-485203/assets/FIRE/CATALOG_01/MONITOR_01/UNET_01/
         pantanal/training_001_pantanal_r1_sentinel2/pantanal_r1_2026_10
"""
import calendar
import datetime
import re
import subprocess
import time

from unet_config import biome_slug, paths, period_info, region_slug


def gee_prefix(cfg):
    return cfg.get("GEE_PREFIX") or f"projects/{cfg['GEE_EXPORT_PROJECT']}/assets/FIRE/CATALOG_01"


def gee_folders(cfg):
    """Folders to create (in order) and the version ImageCollection."""
    base = f"{gee_prefix(cfg)}/{cfg['CAMPAIGN']}"
    exp = f"{base}/{cfg['EXPERIMENT']}"
    biome = f"{exp}/{biome_slug(cfg)}"
    return [base, exp, biome], f"{biome}/{cfg['MODEL_NAME']}"


def asset_id(cfg, period):
    return f"{gee_folders(cfg)[1]}/{region_slug(cfg)}_{period_info(period)[0]}"


def map_path(cfg, period):
    """GCS path (gs://) of the classified map, as unet_predict writes it."""
    P = paths(cfg)
    return f"gs://{P['bucket']}/{P['predict']}/{region_slug(cfg)}_{period_info(period)[0]}_prob.tif"


def time_props(period):
    """time_start / time_end (ms, UTC) and year / month of a period."""
    key, periodicity = period_info(period)
    year = int(key[:4])
    if periodicity == "monthly":
        month = int(key[5:])
        start = datetime.datetime(year, month, 1, tzinfo=datetime.timezone.utc)
        end = start + datetime.timedelta(days=calendar.monthrange(year, month)[1])
    else:
        month = 0
        start = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        end = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
    ms = lambda d: int(d.timestamp() * 1000)  # noqa: E731
    return {"time_start": ms(start), "time_end": ms(end), "year": year, "month": month,
            "periodicity": periodicity}


def map_threshold(cfg, period):
    """Threshold used in the map (reports/predict_{period}.json); None if unknown."""
    from unet_io import read_json
    try:
        return read_json(f"{paths(cfg)['reports']}/predict_{period_info(period)[0]}.json").get("threshold")
    except Exception:
        return None


def upload_command(cfg, period, threshold=None):
    """`earthengine upload image` command (as the production M6); bands burned, prob, doy."""
    from unet_predict import BAND_NAMES
    t = time_props(period)
    props = {
        "source": "mapbiomas-fire", "model": "unet", "model_id": cfg["MODEL_NAME"],
        "region": cfg["REGION"], "biome": biome_slug(cfg), "period": period_info(period)[0],
        "year": t["year"], "month": t["month"], "periodicity": t["periodicity"],
        "campaign": cfg["CAMPAIGN"], "experiment": cfg["EXPERIMENT"],
    }
    if threshold is not None:
        props["threshold"] = round(float(threshold), 4)
    prop_args = " ".join(f"--property {k}={v}" for k, v in props.items())
    return (f"earthengine --project {cfg['GEE_EXPORT_PROJECT']} upload image "
            f"--asset_id={asset_id(cfg, period)} --pyramiding_policy=mode --nodata_value=-9999 "
            f"--band_names={','.join(BAND_NAMES)} "
            f"--time_start {t['time_start']} --time_end {t['time_end']} "
            f"{prop_args} {map_path(cfg, period)}")


def classified_periods(cfg, ls=None):
    """Periods with a map in {version}/predict/ ({region}_{period}_prob.tif)."""
    if ls is None:
        import unet_env
        ls = unet_env.ls
    P = paths(cfg)
    pat = re.compile(rf"^{re.escape(region_slug(cfg))}_(\d{{4}}(?:_\d{{2}})?)_prob\.tif$")
    return sorted(m.group(1) for m in (pat.match(n) for n in ls(f"{P['bucket']}/{P['predict']}")) if m)


def gee_assets(cfg):
    """Names of the images already in the version collection (empty set if none)."""
    import ee
    try:
        res = ee.data.listAssets({"parent": gee_folders(cfg)[1]})
    except Exception:
        return set()
    return {a["name"].split("/")[-1] for a in res.get("assets", [])}


def publish(cfg, periods, logger=print, run=None):
    """Create the folders, replace existing images and submit one ingestion per period.

    Returns {period: True/False} (True = task submitted; the ingestion itself
    runs in the GEE Tasks tab).
    """
    import ee
    run = run or (lambda cmd: subprocess.run(cmd, shell=True).returncode)
    folders, collection = gee_folders(cfg)
    for f in folders:
        try:
            ee.data.createAsset({"type": "Folder"}, f)
        except Exception:
            pass  # já existe
    try:
        ee.data.createAsset({"type": "ImageCollection"}, collection)
    except Exception:
        pass
    done = {}
    for period in periods:
        aid = asset_id(cfg, period)
        try:
            ee.data.getAsset(aid)
            logger(f"   ♻️ {aid.split('/')[-1]} já existe no GEE: substituindo")
            ee.data.deleteAsset(aid)
            time.sleep(2)
        except Exception:
            pass
        ok = run(upload_command(cfg, period, map_threshold(cfg, period))) == 0
        done[period] = ok
        logger(f"   {'✅ enviado' if ok else '❌ falhou'}: {period} → {aid}")
    if any(done.values()):
        logger("💡 A ingestão roda no GEE: acompanhe na aba Tasks do Code Editor (leva alguns minutos).")
    return done
