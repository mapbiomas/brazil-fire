"""
U-Net experiment - publish the classified maps to Google Earth Engine.

Same approach as the production catalog M6 (``M6_publisher.upload_to_gee``):
the ``earthengine upload_manifest`` CLI ingests the COG straight from GCS into an
ImageCollection, one image per region and period, with time and model
properties. An asset that already exists is replaced.

    {GEE_PREFIX}/{CAMPAIGN}/{EXPERIMENT}/{biome}/{version}/{region}_{period}
    e.g. projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/MONITOR_01/UNET_01/
         pantanal/training_001_pantanal_r1_sentinel2/pantanal_r1_2026_10
"""
import calendar
import datetime
import os
import re
import subprocess
import time

from unet_config import biome_slug, paths, period_info, region_slug


def gee_prefix(cfg):
    """Where the maps are stored: projects/mapbiomas-{country}/assets/FIRE/CATALOG_01 (not the project that
    runs the tasks, GEE_EXPORT_PROJECT)."""
    return cfg.get("GEE_PREFIX") or f"projects/mapbiomas-{cfg['COUNTRY']}/assets/FIRE/CATALOG_01"


def gee_folders(cfg):
    """Every folder to create, in order, from the first level under assets/ down to the biome, and the
    version ImageCollection. The parents (FIRE, FIRE/CATALOG_01, ...) may not exist yet in a new project."""
    biome = f"{gee_prefix(cfg)}/{cfg['CAMPAIGN']}/{cfg['EXPERIMENT']}/{biome_slug(cfg)}"
    root, sep, rest = biome.partition("/assets/")
    parts = rest.split("/")
    folders = [f"{root}{sep}{'/'.join(parts[:n])}" for n in range(1, len(parts) + 1)]
    return folders, f"{biome}/{cfg['MODEL_NAME']}"


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


def manifest(cfg, period, threshold=None):
    """Manifesto de ingestão (`earthengine upload_manifest`): bandas burned, prob, doy com nome, MODE, nodata."""
    from unet_predict import BAND_NAMES
    t = time_props(period)
    iso = lambda ms: datetime.datetime.fromtimestamp(ms / 1000, datetime.timezone.utc).strftime(  # noqa: E731
        "%Y-%m-%dT%H:%M:%SZ")
    props = {
        "source": "mapbiomas-fire", "model": "unet", "model_id": cfg["MODEL_NAME"],
        "region": cfg["REGION"], "biome": biome_slug(cfg), "period": period_info(period)[0],
        "year": t["year"], "month": t["month"], "periodicity": t["periodicity"],
        "campaign": cfg["CAMPAIGN"], "experiment": cfg["EXPERIMENT"],
    }
    if threshold is not None:
        props["threshold"] = round(float(threshold), 4)
    return {
        "name": asset_id(cfg, period),
        "tilesets": [{"id": "t0", "sources": [{"uris": [map_path(cfg, period)]}]}],
        "bands": [{"id": nm, "tilesetId": "t0", "tilesetBandIndex": i} for i, nm in enumerate(BAND_NAMES)],
        "properties": props,
        "startTime": iso(t["time_start"]), "endTime": iso(t["time_end"]),
        "pyramidingPolicy": "MODE",
        "missingData": {"values": [-9999]},
    }


def upload_command(cfg, manifest_path):
    """`earthengine upload_manifest` (o `upload image` do CLI não aceita nomes de banda)."""
    return f"earthengine --project {cfg['GEE_EXPORT_PROJECT']} upload_manifest --force {manifest_path}"


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
    def default_run(cmd):
        # captura a saída: no Colab a mensagem de erro do CLI não aparece na célula
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        if r.returncode and out:
            logger("      motivo: " + out[-1500:].replace("\n", "\n      "))
        return r.returncode

    run = run or default_run
    folders, collection = gee_folders(cfg)
    for kind, path in [("Folder", f) for f in folders] + [("ImageCollection", collection)]:
        try:
            ee.data.createAsset({"type": kind}, path)
        except Exception as e:
            if "already exists" not in str(e).lower():  # "já existe" é normal; o resto (permissão, projeto) importa
                logger(f"   ⚠️ não foi possível criar {path}: {e}")
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
        import json
        import tempfile
        mpath = os.path.join(tempfile.mkdtemp(), f"{region_slug(cfg)}_{period_info(period)[0]}.json")
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest(cfg, period, map_threshold(cfg, period)), fh, indent=1)
        ok = run(upload_command(cfg, mpath)) == 0
        done[period] = ok
        logger(f"   {'✅ enviado' if ok else '❌ falhou'}: {period} → {aid}")
    if not all(done.values()):
        logger("💡 Se o motivo for credencial ou permissão: o CLI `earthengine` usa as credenciais do Colab, "
               "então rode `earthengine authenticate` (ou `!earthengine --project "
               f"{cfg['GEE_EXPORT_PROJECT']} ls`) para testar o acesso ao projeto.")
    if any(done.values()):
        logger("💡 A ingestão roda no GEE: acompanhe na aba Tasks do Code Editor (leva alguns minutos).")
    return done
