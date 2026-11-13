"""
U-Net experiment - chips from the M3 polygons (spec 5.3-5.7).

For each year:
  1. spatial blocks (BLOCK_KM) split into train/val/test;
  2. chip centers sampled inside each polygon, balanced fire / not fire;
  3. CHIP x CHIP window of the mosaic + label raster (1 fire, 0 not fire,
     255 no information / nodata);
  4. TFRecord shards in GCS: {EXPERIMENT}/chips/{year}_{split}_{shard}.tfrecord.gz

The chips keep RAW features (reflectance 0-1 + indices, NaN = nodata).
Normalization statistics come later from the train chips only.
"""
import os

import numpy as np

from unet_config import IGNORE, SPLITS, channel_names, paths
from unet_features import PixelReservoir, add_indices, compute_norm_stats, scale_bands
from unet_geo import (BlockGrid, balance_by_class, block_size_in_crs, chip_quality,
                      chip_window, rasterize_labels, sample_points_in_polygon, window_box)
from unet_io import (MosaicReader, _log, load_samples, local_dir, region_geometry,
                     upload, write_json)


# ---------------------------------------------------------------------------
# TFRecord encoding
# ---------------------------------------------------------------------------

def encode_chip(x, y, year, split, cx, cy):
    import tensorflow as tf

    def _bytes(v):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _int(v):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    def _float(v):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v)]))

    h, w, c = x.shape
    feat = {
        "x": _bytes(np.ascontiguousarray(x, dtype=np.float32).tobytes()),
        "y": _bytes(np.ascontiguousarray(y, dtype=np.uint8).tobytes()),
        "h": _int(h), "w": _int(w), "c": _int(c),
        "year": _int(year),
        "split": _bytes(split.encode()),
        "center_x": _float(cx), "center_y": _float(cy),
    }
    return tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString()


def decode_chip_numpy(record):
    """Decode one serialized chip to (x, y, meta) with NumPy (tests, stats)."""
    import tensorflow as tf
    ex = tf.train.Example.FromString(record)
    f = ex.features.feature
    h, w, c = f["h"].int64_list.value[0], f["w"].int64_list.value[0], f["c"].int64_list.value[0]
    x = np.frombuffer(f["x"].bytes_list.value[0], dtype=np.float32).reshape(h, w, c)
    y = np.frombuffer(f["y"].bytes_list.value[0], dtype=np.uint8).reshape(h, w)
    meta = {"year": f["year"].int64_list.value[0],
            "split": f["split"].bytes_list.value[0].decode(),
            "center_x": f["center_x"].float_list.value[0],
            "center_y": f["center_y"].float_list.value[0]}
    return x, y, meta


class ShardWriter:
    """Writes GZIP TFRecord shards of ~SHARD_MB and uploads each one to GCS."""

    def __init__(self, cfg, year, split, logger=None):
        self.cfg, self.year, self.split, self.logger = cfg, year, split, logger
        self.max_bytes = int(cfg["SHARD_MB"]) * 1024 * 1024
        self.dir = local_dir("chips_out")
        self.shard = -1
        self.writer = None
        self.bytes = 0
        self.count = 0
        self.files = []

    def _open(self):
        import tensorflow as tf
        self.shard += 1
        name = f"{self.year}_{self.split}_{self.shard:03d}.tfrecord.gz"
        self.local = os.path.join(self.dir, name)
        self.writer = tf.io.TFRecordWriter(self.local, options="GZIP")
        self.bytes = 0

    def write(self, record):
        if self.writer is None or self.bytes >= self.max_bytes:
            self.close_shard()
            self._open()
        self.writer.write(record)
        self.bytes += len(record)
        self.count += 1

    def close_shard(self):
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        rel = f"{paths(self.cfg)['chips']}/{os.path.basename(self.local)}"
        upload(self.cfg, self.local, rel)
        self.files.append(rel)
        os.remove(self.local)


# ---------------------------------------------------------------------------
# Chip generation
# ---------------------------------------------------------------------------

def _block_grid(cfg, samples, region_geom, crs):
    bounds = region_geom.bounds if region_geom is not None else tuple(samples.total_bounds)
    # Grid covers both the region and the samples.
    sb = samples.total_bounds
    bounds = (min(bounds[0], sb[0]), min(bounds[1], sb[1]),
              max(bounds[2], sb[2]), max(bounds[3], sb[3]))
    lat_center = 0.0
    try:
        import geopandas as gpd
        from shapely.geometry import box
        c = gpd.GeoSeries([box(*bounds)], crs=crs).to_crs("EPSG:4326").iloc[0].centroid
        lat_center = c.y
    except Exception:
        pass
    dx, dy = block_size_in_crs(cfg["BLOCK_KM"], crs, lat_center)
    grid = BlockGrid(bounds, dx, dy)
    cent = samples.geometry.representative_point()
    grid.assign(list(zip(cent.x, cent.y)), cfg["SPLIT"], cfg["SEED"])
    return grid


def sample_centers(cfg, samples, grid):
    """Chip centers inside each polygon; split = split of the polygon's block."""
    import geopandas as gpd
    rng = np.random.default_rng(cfg["SEED"])
    rep = samples.geometry.representative_point()
    rows = []
    for idx, (geom, cls, rp) in enumerate(zip(samples.geometry, samples["class"], rep)):
        split = grid.split_of(rp.x, rp.y)
        for p in sample_points_in_polygon(geom, cfg["POINTS_PER_POLYGON"], rng):
            rows.append({"poly": idx, "cls": int(cls), "split": split,
                         "x": p.x, "y": p.y, "geometry": p})
    centers = gpd.GeoDataFrame(rows, geometry="geometry", crs=samples.crs)
    centers = centers[centers["split"].isin(SPLITS)]
    return balance_by_class(centers, "cls", "split", cfg["SEED"])


def build_chips(cfg, year, logger=None, overwrite=False, n_preview=16):
    """Generate and upload the chips of one year. Returns a summary dict."""
    import M_gcs
    P = paths(cfg)
    existing = M_gcs.list_files(P["chips"], suffix=".tfrecord.gz")
    existing = [f for f in existing if f.split("/")[-1].startswith(f"{year}_")]
    if existing and not overwrite:
        _log(logger, f"[CHIPS] {year}: {len(existing)} shard(s) already in GCS; skipping "
                     "(overwrite=True to rebuild).", "warning")
        return {"year": year, "skipped": True, "shards": existing}
    if existing and overwrite:
        M_gcs.rm_files(existing)

    reader = MosaicReader(cfg, year, logger)
    try:
        inventory = reader.inventory()
        samples, sample_report = load_samples(cfg, year, reader.crs, logger)
        region = region_geometry(cfg, reader.crs, logger)
        grid = _block_grid(cfg, samples, region, reader.crs)
        centers = sample_centers(cfg, samples, grid)
        _log(logger, f"[SPLIT] blocks {grid.counts()} | centers per split: "
                     f"{centers.groupby(['split', 'cls']).size().to_dict()}")

        sindex = samples.sindex
        writers = {s: ShardWriter(cfg, year, s, logger) for s in SPLITS}
        counts = {s: {"chips": 0, "fire_px": 0, "notfire_px": 0} for s in SPLITS}
        dropped = {"cross_block": 0, "few_labels": 0, "nodata": 0}
        overlaps = []
        chip_boxes = {s: [] for s in SPLITS}
        preview = []
        rng = np.random.default_rng(cfg["SEED"])
        bands, indices = cfg["BANDS"], cfg["INDICES"]
        chip = cfg["CHIP"]

        for k, c in enumerate(centers.itertuples(index=False)):
            win = chip_window(reader.transform, c.x, c.y, chip)
            box = window_box(reader.transform, win)
            if grid.splits_of_box(*box) != {c.split}:
                dropped["cross_block"] += 1
                continue
            raw, valid = reader.read(win)
            from rasterio.windows import transform as win_transform
            wt = win_transform(win, reader.transform)
            hits = samples.iloc[list(sindex.intersection(box))]
            y, n_overlap = rasterize_labels(
                hits.geometry[hits["class"] == 1].tolist(),
                hits.geometry[hits["class"] == 0].tolist(), (chip, chip), wt)
            y[~valid] = IGNORE
            keep, lab_frac, nod_frac = chip_quality(
                y, valid, cfg["MIN_LABELED_FRACTION"], cfg["MAX_NODATA_FRACTION"])
            if not keep:
                dropped["nodata" if nod_frac > cfg["MAX_NODATA_FRACTION"] else "few_labels"] += 1
                continue
            if n_overlap:
                overlaps.append({"center_x": c.x, "center_y": c.y, "pixels": n_overlap})
            x = add_indices(scale_bands(raw, cfg["SCALE"], valid), bands, indices)
            writers[c.split].write(encode_chip(x, y, year, c.split, c.x, c.y))
            counts[c.split]["chips"] += 1
            counts[c.split]["fire_px"] += int((y == 1).sum())
            counts[c.split]["notfire_px"] += int((y == 0).sum())
            chip_boxes[c.split].append(box)
            if c.split == "train":
                if len(preview) < n_preview:
                    preview.append((x, y))
                elif rng.random() < 0.05:
                    preview[int(rng.integers(n_preview))] = (x, y)
            if (k + 1) % 500 == 0:
                _log(logger, f"  [CHIPS] {k + 1}/{len(centers)} centers processed")

        for w in writers.values():
            w.close_shard()

        # Acceptance rule: no val/test chip touches a train block (and vice versa).
        for s in ("val", "test"):
            for b in chip_boxes[s]:
                assert "train" not in grid.splits_of_box(*b), f"{s} chip overlaps a train block: {b}"
        for b in chip_boxes["train"]:
            assert grid.splits_of_box(*b) == {"train"}, f"train chip outside train blocks: {b}"

        # Blocks + logs
        tmp = local_dir("splits")
        gpkg = os.path.join(tmp, f"blocks_{year}.gpkg")
        grid.to_geodataframe(reader.crs).to_file(gpkg, driver="GPKG")
        upload(cfg, gpkg, f"{P['splits']}/blocks_{year}.gpkg")

        fig_rel = None
        if preview:
            fig_rel = save_preview(cfg, year, preview)

        summary = {
            "year": year, "inventory": inventory, "samples": sample_report,
            "blocks": grid.counts(), "centers": int(len(centers)),
            "chips": counts, "dropped": dropped,
            "fire_notfire_overlaps": {"chips": len(overlaps), "pixels": int(sum(o["pixels"] for o in overlaps)),
                                       "items": overlaps[:200]},
            "shards": {s: w.files for s, w in writers.items()},
            "channels": channel_names(cfg), "preview": fig_rel,
        }
        write_json(cfg, f"{P['reports']}/chips_{year}.json", summary)
        _log(logger, f"[CHIPS] {year}: " + ", ".join(f"{s} {counts[s]['chips']}" for s in SPLITS)
                     + f" | dropped {dropped} | fire/notfire overlaps in {len(overlaps)} chip(s)")
        return summary
    finally:
        reader.close()


def save_preview(cfg, year, preview):
    """Figure with up to 16 chips: false color SWIR1/NIR/RED + labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = cfg["BANDS"]
    rgb_idx = [bands.index(b) for b in ("swir1", "nir", "red")]
    n = len(preview)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5, rows * 2.6))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (x, y) in enumerate(preview):
        r, c = divmod(i, cols)
        rgb = np.nan_to_num(x[..., rgb_idx])
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
        axes[r, 2 * c].imshow(rgb)
        lab = np.zeros(y.shape + (4,))
        lab[y == 1] = (1, 0, 0, 0.6)
        lab[y == 0] = (0, 0.3, 1, 0.6)
        axes[r, 2 * c + 1].imshow(rgb)
        axes[r, 2 * c + 1].imshow(lab)
    fig.suptitle(f"{cfg['REGION']} {year} - train chips (SWIR1/NIR/RED | red = fire, blue = not fire)")
    fig.tight_layout()
    local = os.path.join(local_dir("reports"), f"chips_preview_{year}.png")
    fig.savefig(local, dpi=110)
    plt.close(fig)
    rel = f"{paths(cfg)['reports']}/chips_preview_{year}.png"
    upload(cfg, local, rel)
    return rel


# ---------------------------------------------------------------------------
# Normalization statistics (train chips only, spec 5.6)
# ---------------------------------------------------------------------------

def compute_stats(cfg, logger=None):
    """Read the train shards of every year and save norm_stats.json."""
    import tensorflow as tf
    from unet_io import download_prefix

    P = paths(cfg)
    files = download_prefix(P["chips"], local_dir("chips"), suffix=".tfrecord.gz")
    train = [f for f in files if "_train_" in os.path.basename(f)
             and int(os.path.basename(f).split("_")[0]) in cfg["YEARS"]]
    if not train:
        raise FileNotFoundError("No train chips found; run build_chips first.")
    names = channel_names(cfg)
    res = PixelReservoir(cfg["STATS_MAX_PIXELS"], len(names), cfg["SEED"])
    for rec in tf.data.TFRecordDataset(train, compression_type="GZIP").as_numpy_iterator():
        x, _, meta = decode_chip_numpy(rec)
        assert meta["split"] == "train"
        res.add(x)
    stats = compute_norm_stats(res.data, names, tuple(cfg["CLIP_PERCENTILES"]))
    stats["train_pixels_seen"] = int(res.seen)
    stats["years"] = list(cfg["YEARS"])
    stats["source"] = "train chips only"
    write_json(cfg, P["stats"], stats)
    _log(logger, f"[STATS] {len(names)} channels from {res.seen} valid train pixels "
                 f"(sample {len(res.data)}) -> gs://{P['bucket']}/{P['stats']}")
    return stats
