"""
U-Net experiment - sliding-window inference over the whole region (spec 5.12).

Output follows the M5 tile format, so M6/M7 and the GEE post-processing can
read it without changes:
  band 1 `probability`: round(p * 1000), int16 0-1000
  band 2 `dayOfYear`:   day of the burn where probability > 0 (M5 rule)
  nodata -9999 outside the region or where the mosaic is invalid.
Same CRS, pixel size and pixel alignment as the mosaic.
"""
import math
import os
import time

import numpy as np

from unet_config import paths, period_info, region_slug
from unet_features import add_indices, normalize, scale_bands
from unet_io import MosaicReader, _log, local_dir, region_geometry, upload, write_json

NODATA = -9999


def region_window(reader, region_geom):
    """Pixel window of the region bbox (whole image if no region)."""
    from rasterio.windows import Window, from_bounds
    if region_geom is None:
        return Window(0, 0, reader.width, reader.height)
    w = from_bounds(*region_geom.bounds, transform=reader.transform)
    c0 = max(0, int(math.floor(w.col_off)))
    r0 = max(0, int(math.floor(w.row_off)))
    c1 = min(reader.width, int(math.ceil(w.col_off + w.width)))
    r1 = min(reader.height, int(math.ceil(w.row_off + w.height)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("A regiao nao cruza o mosaico.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def core_tiles(win, tile):
    """(row, col, h, w) of the core tiles covering a window (absolute pixels)."""
    r0, c0 = int(win.row_off), int(win.col_off)
    H, W = int(win.height), int(win.width)
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            yield r0 + r, c0 + c, min(tile, H - r), min(tile, W - c)


def to_output(prob, valid, doy):
    """M5 encoding: (probability int16 0-1000, dayOfYear where probability > 0)."""
    p = np.full(prob.shape, NODATA, dtype=np.int16)
    p[valid] = np.round(np.clip(prob[valid], 0, 1) * 1000).astype(np.int16)
    d = np.full(prob.shape, NODATA, dtype=np.int16)
    if doy is not None:  # no dayOfYear band -> band 2 stays NODATA
        burned = valid & (p > 0)
        d[burned] = doy[burned].astype(np.int16)
    return p, d


def predict_mosaic(cfg, period, model, stats, logger=None, upload_result=True):
    """Classify the region of one period and upload a COG. Returns the GCS path."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform
    from shapely.geometry import box
    from shapely.prepared import prep

    period = period_info(period)[0]
    P = paths(cfg)
    T, O = cfg["INFER_TILE"], cfg["OVERLAP"]
    reader = MosaicReader(cfg, period, logger)
    try:
        region = region_geometry(cfg, reader.crs, logger)
        region_p = prep(region) if region is not None else None
        win = region_window(reader, region)
        out_transform = win_transform(win, reader.transform)
        name = f"{region_slug(cfg)}_{period}_prob"
        tmp = local_dir("predict")
        local_tif = os.path.join(tmp, f"{name}_tmp.tif")
        profile = {
            "driver": "GTiff", "width": int(win.width), "height": int(win.height),
            "count": 2, "dtype": "int16", "crs": reader.crs, "transform": out_transform,
            "nodata": NODATA, "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "deflate", "BIGTIFF": "IF_SAFER",
        }
        tiles = list(core_tiles(win, T))
        _log(logger, f"[CLASSIFICAR] {period}: {int(win.width)}x{int(win.height)} px, "
                     f"{len(tiles)} janelas de {T}+2x{O}")
        bands, indices = cfg["BANDS"], cfg["INDICES"]
        n_valid = n_burned = 0
        t0 = time.time()

        with rasterio.open(local_tif, "w", **profile) as dst:
            dst.set_band_description(1, "probability")
            dst.set_band_description(2, "dayOfYear")
            batch = []

            def flush():
                nonlocal n_valid, n_burned
                if not batch:
                    return
                xb = np.stack([b["x"] for b in batch])
                prob = np.asarray(model.predict_on_batch(xb))[..., 0]
                for b, pr in zip(batch, prob):
                    h, w = b["h"], b["w"]
                    core = pr[O:O + h, O:O + w]
                    p, d = to_output(core, b["valid"], b["doy"])
                    dwin = Window(b["col"] - int(win.col_off), b["row"] - int(win.row_off), w, h)
                    dst.write(p, 1, window=dwin)
                    dst.write(d, 2, window=dwin)
                    n_valid += int(b["valid"].sum())
                    n_burned += int((p > 0).sum())
                batch.clear()

            for k, (row, col, h, w) in enumerate(tiles):
                core_win = Window(col, row, w, h)
                core_tf = win_transform(core_win, reader.transform)
                if region_p is not None:
                    from rasterio.windows import bounds as wb
                    if not region_p.intersects(box(*wb(core_win, reader.transform))):
                        continue
                in_win = Window(col - O, row - O, T + 2 * O, T + 2 * O)
                raw, valid = reader.read(in_win)
                core_valid = valid[O:O + h, O:O + w]
                if region is not None:
                    core_valid &= geometry_mask([region], (h, w), core_tf, invert=True)
                if not core_valid.any():
                    continue
                x = normalize(add_indices(scale_bands(raw, cfg["SCALE"], valid), bands, indices), stats)
                doy = reader.read_doy(core_win)
                batch.append({"x": x, "valid": core_valid, "doy": doy,
                              "row": row, "col": col, "h": h, "w": w})
                if len(batch) >= cfg["INFER_BATCH"]:
                    flush()
                if (k + 1) % 200 == 0:
                    el = time.time() - t0
                    _log(logger, f"  [CLASSIFICAR] {k + 1}/{len(tiles)} janelas | "
                                 f"{el / 60:.1f} min | faltam ~{el / (k + 1) * (len(tiles) - k - 1) / 60:.1f} min")
            flush()

        cog = os.path.join(tmp, f"{name}.tif")
        from rasterio.shutil import copy as rio_copy
        rio_copy(local_tif, cog, driver="COG", COMPRESS="DEFLATE", BIGTIFF="IF_SAFER")
        os.remove(local_tif)
        rel = f"{P['predict']}/{name}.tif"
        summary = {"period": period, "model": cfg["MODEL_NAME"], "gcs": f"gs://{P['bucket']}/{rel}",
                   "valid_pixels": n_valid, "burned_pixels_prob_gt_0": n_burned,
                   "encoding": ("b1 probability int16 0-1000; b2 dayOfYear where probability > 0"
                                if cfg.get("VALID_BAND") else
                                "b1 probability int16 0-1000; b2 empty (-9999): no dayOfYear band")
                               + "; nodata -9999",
                   "minutes": round((time.time() - t0) / 60, 1)}
        if upload_result:
            upload(cfg, cog, rel)
            write_json(cfg, f"{P['reports']}/predict_{period}.json", summary)
        _log(logger, f"[CLASSIFICAR] {period} -> {summary['gcs']} ({summary['minutes']} min)")
        summary["local"] = cog
        return summary
    finally:
        reader.close()


def gee_upload_command(cfg, period, asset_id):
    """`earthengine upload image` command for the prediction COG (optional)."""
    P = paths(cfg)
    name = f"{region_slug(cfg)}_{period}_prob"
    return (f"earthengine upload image --asset_id={asset_id} "
            f"--nodata_value={NODATA} --pyramiding_policy=mean "
            f"gs://{P['bucket']}/{P['predict']}/{name}.tif")
