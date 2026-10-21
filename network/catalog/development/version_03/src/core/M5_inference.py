import os
import math
import time
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from affine import Affine
from shapely.geometry import shape, mapping
from pyproj import Transformer
from osgeo import gdal
from M_lang import L as Lang
from M0_auth_config import CONFIG, mosaic_name_for_period, period_cog_path, get_temp_dir, _gcs_download, _gcs_upload
from M4_data_extractor import normalize
from M_periods import parse_period

BLOCK_SIZE = 4096

def _fmt_time(s):
    h, r = divmod(int(s), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def load_model_from_gcs(model_dir, fs, logger=None):
    """Carga pesos + metadatos desde GCS y devuelve un predictor NumPy.

    Returns:
        (predict_fn, meta, bands_config, norm_stats, band_order)
    """
    import json

    meta_path = f"{model_dir}/metadata.json"
    if not fs.exists(meta_path):
        raise ValueError(Lang.ERR_MODEL_NO_METADATA.format(path=meta_path))

    with fs.open(meta_path, 'r') as f:
        meta = json.load(f)

    bands_config = meta.get('bands_config', {})
    if not bands_config:
        raise ValueError(Lang.ERR_MODEL_NO_BANDS_CONFIG)

    band_order = meta.get('band_order')
    if not band_order:
        raise ValueError(Lang.ERR_MODEL_NO_BAND_ORDER)

    norm_stats = {int(k): tuple(v) for k, v in meta.get('norm_stats', {}).items()}
    num_input = meta['num_input']
    layers_cfg = meta['layers']

    # Validação: consistência entre band_order, norm_stats, num_input e bands_config
    n_bands = len(band_order)
    n_stats = len(norm_stats)
    n_cfg = len(bands_config)
    mismatches = []
    if n_bands != num_input:
        mismatches.append(f"band_order ({n_bands}) != num_input ({num_input})")
    if n_stats != num_input:
        mismatches.append(f"norm_stats ({n_stats}) != num_input ({num_input})")
    if n_cfg != num_input:
        mismatches.append(f"bands_config ({n_cfg}) != num_input ({num_input})")
    # Verifica se os nomes das bandas em band_order existem em bands_config
    missing = [b for b in band_order if b not in bands_config]
    if missing:
        mismatches.append(f"band_order contém bandas ausentes em bands_config: {missing}")
    if mismatches:
        raise ValueError(Lang.ERR_MODEL_INCONSISTENCY.format(details='; '.join(mismatches)))

    local_npz = os.path.join(get_temp_dir('weights'), f"{meta.get('training_id', 'model')}_weights.npz")
    _gcs_download(f"{model_dir}/weights.npz", local_npz)

    saved_vars = dict(np.load(local_npz))
    os.remove(local_npz)

    from M_dnn import predict_chunked

    def predict_fn(x_normalized, _vars=saved_vars, _layers=list(layers_cfg)):
        return predict_chunked(_vars, _layers, x_normalized)

    if logger:
        logger(f"    Model '{meta.get('training_id')}' loaded ({num_input} bands, NumPy)")

    return predict_fn, meta, bands_config, norm_stats, band_order

def build_band_paths(bands_config, period):
    """Construye diccionario {banda: gcs_path} para un periodo."""
    from M0_auth_config import CONFIG
    info = parse_period(period)
    paths = {}
    for b, cfg in bands_config.items():
        s_name = cfg.get('sensor', 'sentinel2').lower()
        m_type = cfg.get('mosaic', 'minnbr').lower()

        rel_folder = period_cog_path(info.key, mosaic=m_type, sensor=s_name)
        b_correct = 'dayOfYear' if b.lower() == 'dayofyear' else b
        cog_name = f"{mosaic_name_for_period(info.key, band=b_correct, mosaic=m_type, sensor=s_name)}_cog.tif"
        paths[b] = f"{CONFIG['bucket']}/{rel_folder}/{cog_name}"

    # dayOfYear sempre incluso para banda de saida, independente de ser feature
    if 'dayOfYear' not in paths:
        first_cfg = next(iter(bands_config.values()), {})
        s_name = first_cfg.get('sensor', 'sentinel2').lower()
        doy_folder = period_cog_path(info.key, mosaic='minnbr', sensor=s_name)
        doy_name = f"{mosaic_name_for_period(info.key, band='dayOfYear', mosaic='minnbr', sensor=s_name)}_cog.tif"
        paths['dayOfYear'] = f"{CONFIG['bucket']}/{doy_folder}/{doy_name}"

    return paths

def _reproject_geometry(geom_coords, src_crs):
    """Reprojecta geometria EPSG:4326 -> src_crs. Retorna (clip_geom, geom_bounds)."""
    geom = shape(geom_coords)
    if not (src_crs and str(src_crs).upper() != 'EPSG:4326'):
        return [mapping(geom)], geom.bounds
    transformer = Transformer.from_crs('EPSG:4326', src_crs, always_xy=True)
    if hasattr(geom, 'geoms'):
        xs, ys = [], []
        for part in geom.geoms:
            px, py = zip(*part.exterior.coords)
            nx, ny = transformer.transform(list(px), list(py))
            xs.extend(nx); ys.extend(ny)
    else:
        xs, ys = zip(*geom.exterior.coords)
        nx, ny = transformer.transform(list(xs), list(ys))
        xs, ys = nx, ny
    return [mapping(geom)], (min(xs), min(ys), max(xs), max(ys))

def classify_cell_with_cogs(cell_id, predict_fn, bands_config, norm_stats, out_gcs_path, band_paths, band_order, logger=None, worker_id=None):
    """Clasifica un tile cim-world usando COGs via /vsigs/ streaming en bloques.

    Lee los COGs directamente desde GCS (sin descargar) y procesa en
    bloques de BLOCK_SIZE x BLOCK_SIZE para evitar OOM e o limite de
    2 GB do TensorProto.

    Args:
        band_paths: dict {banda: 'bucket/.../cog.tif'} com paths GCS relativos.
        band_order: orden de las bandas (del modelo), ej: ['nir','red','swir1','swir2'].
        worker_id: opcional, prefixo [W1] nos logs.
    """
    import ee

    cim = ee.FeatureCollection("projects/mapbiomas-workspace/AUXILIAR/cim-world-1-250000")
    cell_feature = cim.filter(ee.Filter.eq('name', cell_id)).first()
    cell_geom = cell_feature.geometry()

    sources = {}
    try:
        geom_coords = cell_geom.getInfo()

        for b in band_order:
            vsig = f"/vsigs/{band_paths[b]}"
            sources[b] = rasterio.open(vsig)

        # Abre COG dayOfYear para banda de saida mesmo se nao for feature
        if 'dayOfYear' in band_paths and 'dayOfYear' not in sources:
            try:
                sources['dayOfYear'] = rasterio.open(f"/vsigs/{band_paths['dayOfYear']}")
            except Exception:
                pass

        src_crs = sources[band_order[0]].crs
        src_transform = sources[band_order[0]].transform
        src_nodata = sources[band_order[0]].nodata or -9999
        src_width = sources[band_order[0]].width
        src_height = sources[band_order[0]].height
        clip_geom, geom_bounds = _reproject_geometry(geom_coords, src_crs)

        window = from_bounds(*geom_bounds, transform=src_transform)
        win_col = max(0, int(math.floor(window.col_off)))
        win_row = max(0, int(math.floor(window.row_off)))
        win_w = min(int(math.ceil(window.width)), src_width - win_col)
        win_h = min(int(math.ceil(window.height)), src_height - win_row)

        if win_w <= 0 or win_h <= 0:
            if logger:
                logger(f"    [WARN] {cell_id} is outside the image extent")
            return None

        out_transform = src_transform * Affine.translation(win_col, win_row)
        profile = sources[band_order[0]].profile.copy()
        profile.update({
            'height': win_h,
            'width': win_w,
            'transform': out_transform,
            'dtype': rasterio.int16,
            'count': 2,
            'compress': 'lzw',
            'nodata': -9999,
        })

        local_tmp = os.path.join(get_temp_dir('tiles'), f"{cell_id}.tif")
        total_valid = 0
        total_burned = 0
        conf_sum = 0.0
        conf_count = 0

        _tile_t0 = time.time()
        with rasterio.open(local_tmp, 'w', **profile) as dst:
            n_blocks_h = math.ceil(win_h / BLOCK_SIZE)
            n_blocks_w = math.ceil(win_w / BLOCK_SIZE)
            n_blocks = n_blocks_h * n_blocks_w
            block_idx = 0
            if logger:
                logger(f"    {cell_id}: {win_w}x{win_h} px in {n_blocks} blocks of {BLOCK_SIZE}")

            for row in range(0, win_h, BLOCK_SIZE):
                for col in range(0, win_w, BLOCK_SIZE):
                    bh = min(BLOCK_SIZE, win_h - row)
                    bw = min(BLOCK_SIZE, win_w - col)
                    block_idx += 1

                    src_win = Window(win_col + col, win_row + row, bw, bh)
                    dst_win = Window(col, row, bw, bh)

                    bands_block = {}
                    for b in band_order:
                        arr = sources[b].read(1, window=src_win)
                        bands_block[b] = np.ascontiguousarray(arr)
                    if 'dayOfYear' in sources:
                        doy_arr = sources['dayOfYear'].read(1, window=src_win)
                        bands_block['dayOfYear'] = np.ascontiguousarray(doy_arr)

                    hb, wb = bands_block[band_order[0]].shape
                    stack = np.stack([bands_block[b] for b in band_order], axis=-1)
                    if 'dayOfYear' in bands_block:
                        valid_mask = bands_block['dayOfYear'] > 0
                    else:
                        valid_mask = np.all(stack > src_nodata, axis=-1)
                    n_valid = int(valid_mask.sum())

                    if n_valid == 0:
                        dst.write(np.full((hb, wb), -9999, dtype=np.int16), 1, window=dst_win)
                        dst.write(np.full((hb, wb), -9999, dtype=np.int16), 2, window=dst_win)
                        continue

                    X_norm = normalize(stack[valid_mask], norm_stats)
                    if logger and n_valid > 0:
                        t0 = time.time()
                    probs = predict_fn(X_norm)
                    if logger and n_valid > 0:
                        wid = f"[W{worker_id}] " if worker_id else ""
                        logger(f"    {wid}{cell_id}: block {block_idx}/{n_blocks} ({hb}x{wb}, {n_valid} px) predicted in {_fmt_time(time.time() - t0)}")
                    if probs.ndim > 1 and probs.shape[1] == 1:
                        probs = probs.flatten()

                    output_prob = np.full((hb, wb), -9999, dtype=np.int16)
                    output_doy = np.full((hb, wb), -9999, dtype=np.int16)
                    output_prob[valid_mask] = np.round(probs * 1000).astype(np.int16)
                    if 'dayOfYear' in bands_block:
                        burned_2d = np.zeros((hb, wb), dtype=bool)
                        burned_2d[valid_mask] = np.round(probs * 1000) > 0
                        output_doy[burned_2d] = bands_block['dayOfYear'][burned_2d].astype(np.int16)

                    dst.write(output_prob, 1, window=dst_win)
                    dst.write(output_doy, 2, window=dst_win)

                    total_valid += n_valid
                    # Product convention: any positive probability counts as
                    # burned, using the same rounded int16 value stored in the
                    # raster (band 1) and the dayOfYear mask (band 2).
                    burned = np.round(probs * 1000) > 0
                    n_burned = int(burned.sum())
                    total_burned += n_burned
                    if n_burned > 0:
                        conf_sum += float(probs[burned].sum())
                        conf_count += n_burned

        if total_valid == 0:
            if logger:
                logger(f"    [WARN] {cell_id}: all pixels are nodata")
            os.remove(local_tmp)
            return None

        # Nomeia as bandas no GeoTIFF
        try:
            ds = gdal.Open(local_tmp, 1)
            ds.GetRasterBand(1).SetDescription('probability')
            ds.GetRasterBand(2).SetDescription('dayOfYear')
            ds = None
        except Exception:
            pass

        _gcs_upload(local_tmp, out_gcs_path)
        os.remove(local_tmp)

        mean_conf = float(conf_sum / conf_count) if conf_count > 0 else 0.0

        return {
            'total_pixels': total_valid,
            'burned_pixels': total_burned,
            'mean_confidence': mean_conf,
            'tile_elapsed': time.time() - _tile_t0,
            'n_blocks': n_blocks,
        }

    except Exception as e:
        if logger:
            logger(f"    [ERROR] {cell_id}: {e}")
        raise
    finally:
        for s in sources.values():
            s.close()

