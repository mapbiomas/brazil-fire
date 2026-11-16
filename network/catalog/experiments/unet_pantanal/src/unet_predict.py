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

Speed: the mosaic is read one row of windows at a time (one big read per band, bands in
parallel threads, the next row prefetched) and the windows are prepared in worker threads
while the GPU predicts the previous batch.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

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
        for i, nm in enumerate(BAND_NAMES, 1):
            dst.set_band_description(i, nm)
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


def split_rows(tiles):
    """Agrupa as janelas por linha: [(row, [(row, col, h, w), ...]), ...], de cima para baixo."""
    rows = {}
    for tl in tiles:
        rows.setdefault(tl[0], []).append(tl)
    return sorted(rows.items())


def block_geometry(row, tiles, tile, overlap):
    """Bloco de leitura de UMA linha de janelas: (c_min, col_off, row_off, largura, altura).

    Cobre as janelas dadas com a borda de sobreposição; cada janela é um recorte de
    (tile + 2*overlap) colunas a partir de `col - c_min`.
    """
    c_min = min(tl[1] for tl in tiles)
    c_max = max(tl[1] for tl in tiles)
    return c_min, c_min - overlap, row - overlap, (c_max - c_min) + tile + 2 * overlap, tile + 2 * overlap


def _gdal_tuning():
    """Opções do GDAL para ler COGs do GCS (/vsigs/); o que o usuário já definiu é mantido."""
    for k, v in {"GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
                 "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE",
                 "VSI_CACHE_SIZE": str(256 * 1024 * 1024), "GDAL_CACHEMAX": "512"}.items():
        os.environ.setdefault(k, v)


def _finish(cfg, period, name, rel, parts_rel, profile, win, strips, T, threshold, t0, logger, upload_result, tmp):
    """Junta as faixas (baixando as que faltam), faz o COG, sobe ao GCS e apaga as faixas."""
    from rasterio.shutil import copy as rio_copy
    P = paths(cfg)

    def part_local(k):
        return os.path.join(tmp, f"{name}_{part_name(k)}")

    missing = [k for k, _, _ in strips if not os.path.exists(part_local(k))]
    for n, k in enumerate(missing, 1):
        _log(logger, f"[JUNTAR] baixando a faixa {k + 1}/{len(strips)} do GCS ({n}/{len(missing)})")
        gcs.download(f"{P['bucket']}/{parts_rel}/{part_name(k)}", part_local(k))
    _log(logger, f"[JUNTAR] juntando {len(strips)} faixas em um raster de {profile['width']}x{profile['height']} px")
    local_tif = os.path.join(tmp, f"{name}_tmp.tif")
    t1 = time.time()
    n_valid, n_burned, n_prob = assemble_parts(local_tif, profile, win, strips, part_local, T)
    _log(logger, f"[JUNTAR] faixas juntadas em {(time.time() - t1) / 60:.1f} min; convertendo para COG "
                 f"(a parte mais demorada, sem mensagens até terminar)")
    cog = os.path.join(tmp, f"{name}.tif")
    t1 = time.time()
    rio_copy(local_tif, cog, driver="COG", COMPRESS="DEFLATE", BIGTIFF="IF_SAFER")
    os.remove(local_tif)
    _log(logger, f"[JUNTAR] COG pronto em {(time.time() - t1) / 60:.1f} min "
                 f"({os.path.getsize(cog) / 1e9:.2f} GB)")
    summary = {"period": period, "model": cfg["MODEL_NAME"], "gcs": f"gs://{P['bucket']}/{rel}",
               "valid_pixels": n_valid, "burned_pixels": n_burned, "pixels_prob_gt_0": n_prob,
               "threshold": threshold, "bands": list(BAND_NAMES), "assembled": True,
               "encoding": (f"burned: 1 if prob >= {threshold:.2f}, else 0; prob: probability int16 0-1000; "
                            + ("doy: dayOfYear where prob > 0" if cfg.get("VALID_BAND") else
                               "doy: empty (-9999), no dayOfYear band")) + "; nodata -9999",
               "minutes": round((time.time() - t0) / 60, 1)}
    if upload_result:
        _log(logger, "[JUNTAR] enviando o COG ao GCS")
        upload(cfg, cog, rel)
        write_json(cfg, f"{P['reports']}/predict_{period}.json", summary)
        delete_files(cfg, gcs.list_files(parts_rel))  # só depois do mapa final salvo
    for k, _, _ in strips:
        if os.path.exists(part_local(k)):
            os.remove(part_local(k))
    _log(logger, f"[JUNTAR] {period} -> {summary['gcs']}")
    summary["local"] = cog
    return summary


def assembly_status(cfg, files=None, manifests=None):
    """Meses com faixas no GCS: [(período, faixas prontas, faixas esperadas)], só os que têm manifesto."""
    import re
    P = paths(cfg)
    parts_root = f"{P['predict']}/parts"
    if files is None:
        files = gcs.list_files(parts_root)
    pat = re.compile(rf"/{re.escape(region_slug(cfg))}_(\d{{4}}(?:_\d{{2}})?)_prob/([^/]+)$")
    found = {}
    for f in files:
        m = pat.search(f)
        if m:
            found.setdefault(m.group(1), set()).add(m.group(2))
    out = []
    for period in sorted(found):
        names = found[period]
        if "manifest.json" not in names:
            continue
        man = (manifests or {}).get(period)
        if man is None:
            man = read_json(f"{parts_root}/{region_slug(cfg)}_{period}_prob/manifest.json")
        from rasterio.windows import Window
        win = Window(man["col_off"], man["row_off"], man["width"], man["height"])
        total = len(strip_plan(win, man["tile"], man["strip_rows"]))
        done = len([n for n in names if n.startswith("part_") and n.endswith(".tif")])
        out.append((period, done, total))
    return out


def assemble_map(cfg, period, logger=None, overwrite=False):
    """Passo 6.3: junta as faixas salvas no GCS no COG final do mês (sem precisar do modelo nem da GPU)."""
    from rasterio.windows import Window
    from rasterio.windows import transform as win_transform
    period = period_info(period)[0]
    P = paths(cfg)
    name = f"{region_slug(cfg)}_{period}_prob"
    rel = f"{P['predict']}/{name}.tif"
    parts_rel = f"{P['predict']}/parts/{name}"
    if not overwrite and gcs.blob_exists(f"{P['bucket']}/{rel}"):
        _log(logger, f"[JUNTAR] {period}: o mapa já existe no GCS, pulando")
        return {"period": period, "gcs": f"gs://{P['bucket']}/{rel}", "skipped": True}
    try:
        man = read_json(f"{parts_rel}/manifest.json")
    except Exception:
        raise RuntimeError(f"{period}: não há faixas salvas em {parts_rel}. Rode o passo 6.2 antes.")
    win = Window(man["col_off"], man["row_off"], man["width"], man["height"])
    strips = strip_plan(win, man["tile"], man["strip_rows"])
    have = {os.path.basename(f) for f in gcs.list_files(parts_rel)}
    absent = [k + 1 for k, _, _ in strips if part_name(k) not in have]
    if absent:
        raise RuntimeError(f"{period}: faltam as faixas {absent} de {len(strips)}. Rode o passo 6.2 de novo "
                           f"para terminar a classificação.")
    reader = MosaicReader(cfg, period, logger)
    try:
        profile = {
            "driver": "GTiff", "width": int(win.width), "height": int(win.height),
            "count": 3, "dtype": "int16", "crs": reader.crs, "transform": win_transform(win, reader.transform),
            "nodata": NODATA, "tiled": True, "blockxsize": 512, "blockysize": 512,
            "compress": "deflate", "BIGTIFF": "IF_SAFER",
        }
    finally:
        reader.close()
    return _finish(cfg, period, name, rel, parts_rel, profile, win, strips, man["tile"],
                   man.get("threshold", 0.5), time.time(), logger, True, local_dir("predict"))


def predict_mosaic(cfg, period, model, stats, logger=None, upload_result=True, overwrite=False,
                   strip_rows=STRIP_ROWS, assemble=True):
    """Classify the region of one period and upload a COG. Returns the summary dict.

    The result of every strip is saved in GCS as it finishes, and an existing final map is not
    redone unless overwrite=True. With assemble=False the strips stay in GCS and the final COG is
    made later by assemble_map() (notebook step 6.3).
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
    _gdal_tuning()
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
        wait = {"leitura": 0.0, "preparo": 0.0, "gpu": 0.0}
        strip_info = {k: (srow, sh) for k, srow, sh in strips}
        work = [(k, row, tl) for k, _, _ in strips if k not in done for row, tl in split_rows(by_strip[k])]
        n_cpu = max(2, min(4, os.cpu_count() or 2))
        io_pool = ThreadPoolExecutor(max_workers=len(reader.all_bands)) if not reader.multiband else None
        cpu_pool = ThreadPoolExecutor(max_workers=n_cpu)
        row_exec = ThreadPoolExecutor(max_workers=1)
        cur = {"k": None, "dst": None, "counts": None}

        def part_local(k):
            return os.path.join(tmp, f"{name}_{part_name(k)}")

        def start_part(k):
            srow, sh = strip_info[k]
            sprofile = dict(profile, height=sh,
                            transform=win_transform(Window(int(win.col_off), srow, W, sh), reader.transform))
            dst = rasterio.open(part_local(k), "w", **sprofile)
            for i, nm in enumerate(BAND_NAMES, 1):
                dst.set_band_description(i, nm)
            cur.update(k=k, dst=dst, counts=[0, 0, 0])  # válidos, queimados (>= limiar), prob > 0

        def finish_part():
            if cur["dst"] is None:
                return
            c = cur["counts"]
            cur["dst"].update_tags(valid_pixels=c[0], burned_pixels=c[1], prob_pixels=c[2])
            cur["dst"].close()
            cur["dst"] = None
            if upload_result:
                upload(cfg, part_local(cur["k"]), f"{parts_rel}/{part_name(cur['k'])}")
                _log(logger, f"[CLASSIFICAR] faixa {cur['k'] + 1}/{len(strips)} salva no GCS "
                             f"({(time.time() - t0) / 60:.1f} min)")

        def load_row(row, tiles):
            """Lê de uma vez o bloco de uma linha de janelas (só as que tocam a região)."""
            active = tiles
            if region_p is not None:
                from rasterio.windows import bounds as wb
                active = [tl for tl in tiles if region_p.intersects(
                    box(*wb(Window(tl[1], tl[0], tl[3], tl[2]), reader.transform)))]
            if not active:
                return {"tiles": []}
            c_min, co, ro, bw, bh = block_geometry(row, active, T, O)
            raw, valid, doy = reader.read_all(Window(co, ro, bw, bh), pool=io_pool)
            return {"tiles": active, "c_min": c_min, "raw": raw, "valid": valid, "doy": doy}

        def prep_tile(blk, tl):
            """Recorta a janela do bloco e prepara a entrada da rede (roda em thread)."""
            row, col, h, w = tl
            x0 = col - blk["c_min"]
            raw_t = blk["raw"][:, :, x0:x0 + T + 2 * O]
            valid_t = blk["valid"][:, x0:x0 + T + 2 * O]
            core_valid = valid_t[O:O + h, O:O + w].copy()
            if region is not None:
                from rasterio.windows import bounds as wb
                core_win = Window(col, row, w, h)
                if not region_p.contains(box(*wb(core_win, reader.transform))):  # só nas bordas da região
                    core_valid &= geometry_mask([region], (h, w), win_transform(core_win, reader.transform),
                                                invert=True)
            if not core_valid.any():
                return None
            x = normalize(add_indices(scale_bands(raw_t, cfg["SCALE"], valid_t), bands, indices), stats)
            doy = None if blk["doy"] is None else blk["doy"][O:O + h, x0 + O:x0 + O + w]
            return {"x": x, "valid": core_valid, "doy": doy, "row": row, "col": col, "h": h, "w": w}

        def predict_write(batch):
            t1 = time.time()
            srow = strip_info[cur["k"]][0]
            xb = np.stack([b["x"] for b in batch])
            prob = np.asarray(model.predict_on_batch(xb))[..., 0]
            for b, pr in zip(batch, prob):
                h, w = b["h"], b["w"]
                bu, p, d = to_output(pr[O:O + h, O:O + w], b["valid"], b["doy"], threshold)
                dwin = Window(b["col"] - int(win.col_off), b["row"] - srow, w, h)
                cur["dst"].write(bu, 1, window=dwin)
                cur["dst"].write(p, 2, window=dwin)
                cur["dst"].write(d, 3, window=dwin)
                cur["counts"][0] += int(b["valid"].sum())
                cur["counts"][1] += int((bu == 1).sum())
                cur["counts"][2] += int((p > 0).sum())
            wait["gpu"] += time.time() - t1

        def run_row(blk):
            """Prepara as janelas em threads enquanto a GPU prediz o lote anterior."""
            tiles, B = blk["tiles"], cfg["INFER_BATCH"]
            groups = [tiles[i:i + B] for i in range(0, len(tiles), B)]
            futs = [cpu_pool.submit(prep_tile, blk, tl) for tl in groups[0]]
            for gi in range(len(groups)):
                t1 = time.time()
                items = [f.result() for f in futs]
                wait["preparo"] += time.time() - t1
                futs = ([cpu_pool.submit(prep_tile, blk, tl) for tl in groups[gi + 1]]
                        if gi + 1 < len(groups) else [])
                batch = [i for i in items if i is not None]
                if batch:
                    predict_write(batch)

        try:
            fut = row_exec.submit(load_row, work[0][1], work[0][2]) if work else None
            for i, (k, row, tl) in enumerate(work):
                t1 = time.time()
                blk = fut.result()
                wait["leitura"] += time.time() - t1
                fut = row_exec.submit(load_row, work[i + 1][1], work[i + 1][2]) if i + 1 < len(work) else None
                if cur["k"] != k:
                    finish_part()
                    start_part(k)
                if blk["tiles"]:
                    run_row(blk)
                del blk
                n_done_tiles += len(tl)
                el = time.time() - t0
                _log(logger, f"  [CLASSIFICAR] {n_done_tiles}/{todo_tiles} janelas | {el / 60:.1f} min | "
                             f"faltam ~{el / n_done_tiles * (todo_tiles - n_done_tiles) / 60:.1f} min | espera: "
                             f"leitura {wait['leitura']:.0f}s, preparo {wait['preparo']:.0f}s, GPU {wait['gpu']:.0f}s")
            finish_part()
        finally:
            if cur["dst"] is not None:
                cur["dst"].close()
            row_exec.shutdown(wait=True, cancel_futures=True)
            cpu_pool.shutdown(wait=True, cancel_futures=True)
            if io_pool is not None:
                io_pool.shutdown(wait=True, cancel_futures=True)

        if not assemble and upload_result:
            _log(logger, f"[CLASSIFICAR] {period}: {len(strips)} faixas prontas no GCS "
                         f"({(time.time() - t0) / 60:.1f} min nesta sessão). Falta juntar (passo 6.3).")
            return {"period": period, "model": cfg["MODEL_NAME"], "assembled": False,
                    "strips": len(strips), "minutes": round((time.time() - t0) / 60, 1)}
        return _finish(cfg, period, name, rel, parts_rel, profile, win, strips, T, threshold, t0, logger,
                       upload_result, tmp)
    finally:
        reader.close()
