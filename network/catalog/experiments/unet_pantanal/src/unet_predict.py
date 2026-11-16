"""
U-Net experiment - sliding-window inference over the whole region (spec 5.12).

Output: three int16 bands, nodata -9999 outside the region or where the mosaic is invalid:
  burned  1 where the probability is >= the validation threshold, else 0
  prob    probability x 1000 (0-1000)
  doy     day of the year where prob > 0 (empty when there is no dayOfYear band)
Same CRS, pixel size and pixel alignment as the mosaic.

The region is classified in horizontal strips; each finished strip is saved in GCS
({predict}/parts/{name}/part_NNN.tif), so a Colab disconnection only loses the strip in
progress: running the step again resumes from the next strip.
"""
import math
import os
import time

import numpy as np

from unet_config import paths, period_info, region_slug
from unet_features import add_indices, normalize, scale_bands
import unet_env as gcs
from unet_io import (MosaicReader, _log, delete_files, local_dir, read_json, region_geometry, upload,
                     write_json)

NODATA = -9999
# Output bands: burned (1 = probability >= validation threshold), prob (x 1000), doy (day of year).
BAND_NAMES = ("burned", "prob", "doy")


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
        raise ValueError("A região não cruza o mosaico.")
    return Window(c0, r0, c1 - c0, r1 - r0)


def core_tiles(win, tile):
    """(row, col, h, w) of the core tiles covering a window (absolute pixels)."""
    r0, c0 = int(win.row_off), int(win.col_off)
    H, W = int(win.height), int(win.width)
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            yield r0 + r, c0 + c, min(tile, H - r), min(tile, W - c)


def get_threshold(cfg, logger=None):
    """Cut of the burned band: THRESHOLD, else the best-F1 threshold of the validation, else 0.5."""
    if cfg.get("THRESHOLD") is not None:
        return float(cfg["THRESHOLD"])
    P = paths(cfg)
    for name in ["all"] + list(cfg["PERIODS"]):
        try:
            return float(read_json(f"{P['reports']}/metrics_{name}.json")["val_best_threshold"])
        except Exception:
            continue
    _log(logger, "[CLASSIFICAR][AVISO] sem métricas de validação (passo 5): usando limiar 0.5", "warning")
    return 0.5


def to_output(prob, valid, doy, threshold=0.5):
    """(burned 0/1, prob int16 0-1000, doy where prob > 0); NODATA where not valid."""
    b = np.full(prob.shape, NODATA, dtype=np.int16)
    b[valid] = (prob[valid] >= threshold).astype(np.int16)
    p = np.full(prob.shape, NODATA, dtype=np.int16)
    p[valid] = np.round(np.clip(prob[valid], 0, 1) * 1000).astype(np.int16)
    d = np.full(prob.shape, NODATA, dtype=np.int16)
    if doy is not None:  # no dayOfYear band -> band 2 stays NODATA
        burned = valid & (p > 0)
        d[burned] = doy[burned].astype(np.int16)
    return b, p, d


STRIP_ROWS = 4  # linhas de janelas por faixa salva (checkpoint no GCS)


def strip_plan(win, tile, strip_rows=STRIP_ROWS):
    """Faixas horizontais da janela: [(k, linha_absoluta, altura)], cada uma com `strip_rows` linhas de janelas."""
    r0, H = int(win.row_off), int(win.height)
    step = tile * strip_rows
    return [(k, r0 + off, min(step, H - off)) for k, off in enumerate(range(0, H, step))]


def _model_signature(model):
    """Impressão digital curta dos pesos (para não reaproveitar faixas de outro modelo)."""
    import hashlib
    try:
        return hashlib.md5(np.ascontiguousarray(model.get_weights()[0]).tobytes()).hexdigest()[:12]
    except Exception:
        return ""


def part_name(k):
    return f"part_{k:03d}.tif"


def resume_state(cfg, parts_rel, manifest, logger=None):
    """Faixas já salvas no GCS para ESTE modelo e estas opções ({k}); apaga as de outra execução."""
    names = {os.path.basename(f): f for f in gcs.list_files(parts_rel)}
    old = None
    if "manifest.json" in names:
        try:
            old = read_json(f"{parts_rel}/manifest.json")
        except Exception:
            old = None
    if old == manifest:
        done = {int(n[5:8]) for n in names if n.startswith("part_") and n.endswith(".tif")}
        if done:
            _log(logger, f"[CLASSIFICAR] retomando: {len(done)} faixa(s) já salva(s) no GCS")
        return done
    if names:
        _log(logger, "[CLASSIFICAR] faixas salvas de outro modelo ou de outras opções: descartando", "warning")
        delete_files(cfg, list(names.values()))
    write_json(cfg, f"{parts_rel}/manifest.json", manifest)
    return set()


def assemble_parts(final_path, profile, win, strips, part_path, tile):
    """Junta as faixas (arquivos locais) no raster final, em blocos. Devolve (válidos, queimados, prob > 0)."""
    import rasterio
    from rasterio.windows import Window
    total = [0, 0, 0]
    with rasterio.open(final_path, "w", **profile) as dst:
        for k, srow, sh in strips:
            with rasterio.open(part_path(k)) as src:
                tags = src.tags()
                for i, key in enumerate(("valid_pixels", "burned_pixels", "prob_pixels")):
                    total[i] += int(tags.get(key, 0))
                for r in range(0, sh, tile):
                    h = min(tile, sh - r)
                    dst.write(src.read(window=Window(0, r, src.width, h)),
                              window=Window(0, srow - int(win.row_off) + r, src.width, h))
    return tuple(total)


def predict_mosaic(cfg, period, model, stats, logger=None, upload_result=True, overwrite=False,
                   strip_rows=STRIP_ROWS):
    """Classify the region of one period and upload a COG. Returns the summary dict.

    The result of every strip is saved in GCS as it finishes, and an existing final map is not
    redone unless overwrite=True.
    """
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform
    from shapely.geometry import box
    from shapely.prepared import prep

    period = period_info(period)[0]
    P = paths(cfg)
    name = f"{region_slug(cfg)}_{period}_prob"
    rel = f"{P['predict']}/{name}.tif"
    if upload_result and not overwrite and gcs.blob_exists(f"{P['bucket']}/{rel}"):
        _log(logger, f"[CLASSIFICAR] {period}: o mapa já existe no GCS, pulando "
                     f"(REFAZER_MAPAS=True para classificar de novo)")
        try:
            summary = read_json(f"{P['reports']}/predict_{period}.json")
        except Exception:
            summary = {"period": period, "model": cfg["MODEL_NAME"], "gcs": f"gs://{P['bucket']}/{rel}"}
        summary["skipped"] = True
        return summary

    T, O = cfg["INFER_TILE"], cfg["OVERLAP"]
    threshold = get_threshold(cfg, logger)
    reader = MosaicReader(cfg, period, logger)
    try:
        region = region_geometry(cfg, reader.crs, logger)
        region_p = prep(region) if region is not None else None
        win = region_window(reader, region)
        out_transform = win_transform(win, reader.transform)
        tmp = local_dir("predict")
        W = int(win.width)
        profile = {
            "driver": "GTiff", "width": W, "height": int(win.height),
            "count": 3, "dtype": "int16", "crs": reader.crs, "transform": out_transform,
            "nodata": NODATA, "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "deflate", "BIGTIFF": "IF_SAFER",
        }
        strips = strip_plan(win, T, strip_rows)
        by_strip = {k: [] for k, _, _ in strips}
        for row, col, h, w in core_tiles(win, T):
            by_strip[(row - int(win.row_off)) // (T * strip_rows)].append((row, col, h, w))
        n_tiles = sum(len(v) for v in by_strip.values())

        parts_rel = f"{P['predict']}/parts/{name}"
        manifest = {"model": cfg["MODEL_NAME"], "weights": _model_signature(model), "threshold": threshold,
                    "tile": T, "overlap": O, "strip_rows": strip_rows, "width": W, "height": int(win.height),
                    "row_off": int(win.row_off), "col_off": int(win.col_off),
                    "channels": list(cfg["BANDS"]) + list(cfg["INDICES"]), "valid_band": cfg.get("VALID_BAND")}
        done = resume_state(cfg, parts_rel, manifest, logger) if upload_result else set()
        todo_tiles = sum(len(by_strip[k]) for k, _, _ in strips if k not in done)
        _log(logger, f"[CLASSIFICAR] {period}: {W}x{int(win.height)} px, {n_tiles} janelas de {T}+2x{O} "
                     f"em {len(strips)} faixas ({len(done)} já prontas)")
        bands, indices = cfg["BANDS"], cfg["INDICES"]
        t0 = time.time()
        n_done_tiles = 0

        def part_local(k):
            return os.path.join(tmp, f"{name}_{part_name(k)}")

        for k, srow, sh in strips:
            if k in done:
                continue
            counts = [0, 0, 0]  # válidos, queimados (>= limiar), prob > 0
            sprofile = dict(profile, height=sh,
                            transform=win_transform(Window(int(win.col_off), srow, W, sh), reader.transform))
            with rasterio.open(part_local(k), "w", **sprofile) as dst:
                for i, nm in enumerate(BAND_NAMES, 1):
                    dst.set_band_description(i, nm)
                batch = []

                def flush():
                    if not batch:
                        return
                    xb = np.stack([b["x"] for b in batch])
                    prob = np.asarray(model.predict_on_batch(xb))[..., 0]
                    for b, pr in zip(batch, prob):
                        h, w = b["h"], b["w"]
                        core = pr[O:O + h, O:O + w]
                        bu, p, d = to_output(core, b["valid"], b["doy"], threshold)
                        dwin = Window(b["col"] - int(win.col_off), b["row"] - srow, w, h)
                        dst.write(bu, 1, window=dwin)
                        dst.write(p, 2, window=dwin)
                        dst.write(d, 3, window=dwin)
                        counts[0] += int(b["valid"].sum())
                        counts[1] += int((bu == 1).sum())
                        counts[2] += int((p > 0).sum())
                    batch.clear()

                for row, col, h, w in by_strip[k]:
                    n_done_tiles += 1
                    if n_done_tiles % 200 == 0:
                        el = time.time() - t0
                        _log(logger, f"  [CLASSIFICAR] {n_done_tiles}/{todo_tiles} janelas | {el / 60:.1f} min | "
                                     f"faltam ~{el / n_done_tiles * (todo_tiles - n_done_tiles) / 60:.1f} min")
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
                flush()
                dst.update_tags(valid_pixels=counts[0], burned_pixels=counts[1], prob_pixels=counts[2])
            if upload_result:
                upload(cfg, part_local(k), f"{parts_rel}/{part_name(k)}")
                _log(logger, f"[CLASSIFICAR] faixa {k + 1}/{len(strips)} salva no GCS "
                             f"({(time.time() - t0) / 60:.1f} min)")

        # junta as faixas (as que já estavam no GCS são baixadas)
        for k, _, _ in strips:
            if not os.path.exists(part_local(k)):
                gcs.download(f"{P['bucket']}/{parts_rel}/{part_name(k)}", part_local(k))
        local_tif = os.path.join(tmp, f"{name}_tmp.tif")
        n_valid, n_burned, n_prob = assemble_parts(local_tif, profile, win, strips, part_local, T)
        cog = os.path.join(tmp, f"{name}.tif")
        from rasterio.shutil import copy as rio_copy
        rio_copy(local_tif, cog, driver="COG", COMPRESS="DEFLATE", BIGTIFF="IF_SAFER")
        os.remove(local_tif)
        summary = {"period": period, "model": cfg["MODEL_NAME"], "gcs": f"gs://{P['bucket']}/{rel}",
                   "valid_pixels": n_valid, "burned_pixels": n_burned, "pixels_prob_gt_0": n_prob,
                   "threshold": threshold, "bands": list(BAND_NAMES),
                   "encoding": (f"burned: 1 if prob >= {threshold:.2f}, else 0; prob: probability int16 0-1000; "
                                + ("doy: dayOfYear where prob > 0" if cfg.get("VALID_BAND") else
                                   "doy: empty (-9999), no dayOfYear band")) + "; nodata -9999",
                   "minutes": round((time.time() - t0) / 60, 1)}
        if upload_result:
            upload(cfg, cog, rel)
            write_json(cfg, f"{P['reports']}/predict_{period}.json", summary)
            delete_files(cfg, gcs.list_files(parts_rel))  # só depois do mapa final salvo
        for k, _, _ in strips:
            if os.path.exists(part_local(k)):
                os.remove(part_local(k))
        _log(logger, f"[CLASSIFICAR] {period} -> {summary['gcs']} ({summary['minutes']} min nesta sessão)")
        summary["local"] = cog
        return summary
    finally:
        reader.close()
