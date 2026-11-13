"""
U-Net experiment - geometry helpers (no GCS/GEE access).

- BlockGrid: spatial blocks for the train/val/test split.
- Chip centers sampled inside the M3 polygons.
- Chip windows and 1/0/255 label rasters.
"""
import math

import numpy as np

from unet_config import IGNORE, SPLITS


# ---------------------------------------------------------------------------
# Spatial blocks
# ---------------------------------------------------------------------------

def block_size_in_crs(block_km, crs, lat_center):
    """Block size (dx, dy) in CRS units. Geographic CRS -> degrees."""
    is_geographic = False
    try:
        from pyproj import CRS
        is_geographic = CRS.from_user_input(crs).is_geographic
    except Exception:
        is_geographic = str(crs).upper() in ("EPSG:4326", "OGC:CRS84")
    if is_geographic:
        dy = block_km / 111.32
        dx = block_km / (111.32 * max(math.cos(math.radians(lat_center)), 1e-6))
        return dx, dy
    return block_km * 1000.0, block_km * 1000.0


class BlockGrid:
    """Regular grid of blocks; each block gets one split (or 'unused')."""

    def __init__(self, bounds, dx, dy):
        minx, miny, maxx, maxy = bounds
        self.x0, self.y0 = float(minx), float(miny)
        self.dx, self.dy = float(dx), float(dy)
        self.nx = max(1, int(math.ceil((maxx - minx) / dx)))
        self.ny = max(1, int(math.ceil((maxy - miny) / dy)))
        self.split = {}

    def index(self, x, y):
        i = int(math.floor((x - self.x0) / self.dx))
        j = int(math.floor((y - self.y0) / self.dy))
        return i, j

    def split_of(self, x, y):
        return self.split.get(self.index(x, y), "unused")

    def blocks_of_box(self, minx, miny, maxx, maxy):
        i0, j0 = self.index(minx, miny)
        # Max edge belongs to the previous block when it lies exactly on a line.
        i1, j1 = self.index(maxx - 1e-12 * self.dx, maxy - 1e-12 * self.dy)
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def splits_of_box(self, minx, miny, maxx, maxy):
        return {self.split.get(b, "unused") for b in self.blocks_of_box(minx, miny, maxx, maxy)}

    def assign(self, occupied_xy, fractions, seed):
        """Shuffle the blocks that contain samples and split them by fractions.

        Blocks without samples stay 'unused' (no chips come from them).
        """
        occupied = sorted({self.index(x, y) for x, y in occupied_xy})
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(occupied))
        n = len(occupied)
        n_train = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        self.split = {}
        for rank, k in enumerate(order):
            b = occupied[k]
            if rank < n_train:
                self.split[b] = "train"
            elif rank < n_train + n_val:
                self.split[b] = "val"
            else:
                self.split[b] = "test"
        return self

    def block_box(self, b):
        i, j = b
        return (self.x0 + i * self.dx, self.y0 + j * self.dy,
                self.x0 + (i + 1) * self.dx, self.y0 + (j + 1) * self.dy)

    def counts(self):
        return {s: sum(1 for v in self.split.values() if v == s) for s in SPLITS}

    def to_geodataframe(self, crs):
        import geopandas as gpd
        from shapely.geometry import box
        rows = [
            {"block": f"{i}_{j}", "split": s, "geometry": box(*self.block_box((i, j)))}
            for (i, j), s in sorted(self.split.items())
        ]
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


# ---------------------------------------------------------------------------
# Chip centers inside polygons
# ---------------------------------------------------------------------------

def sample_points_in_polygon(geom, n, rng, max_tries=50):
    """Up to n random points inside a polygon (rejection sampling).

    Tiny polygons that reject every try fall back to one representative point.
    """
    from shapely.geometry import Point
    from shapely.prepared import prep

    if geom is None or geom.is_empty:
        return []
    minx, miny, maxx, maxy = geom.bounds
    pg = prep(geom)
    pts = []
    tries = 0
    while len(pts) < n and tries < max_tries * n:
        batch = max(8, 2 * (n - len(pts)))
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        for x, y in zip(xs, ys):
            p = Point(x, y)
            if pg.contains(p):
                pts.append(p)
                if len(pts) == n:
                    break
        tries += batch
    if not pts:
        pts = [geom.representative_point()]
    return pts


def balance_by_class(df, class_col, split_col, seed):
    """Same number of fire and not-fire centers inside each split."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(split_col):
        n = g[class_col].value_counts()
        if len(n) < 2:
            continue  # a split with one class only cannot be balanced
        k = int(n.min())
        for _, gc in g.groupby(class_col):
            idx = rng.choice(len(gc), k, replace=False)
            parts.append(gc.iloc[np.sort(idx)])
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Windows and labels
# ---------------------------------------------------------------------------

def chip_window(transform, x, y, chip):
    """Window of chip x chip pixels centered on the pixel that contains (x, y)."""
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    row, col = rowcol(transform, x, y)
    row, col = int(row), int(col)
    half = chip // 2
    return Window(col - half, row - half, chip, chip)


def window_box(transform, window):
    """(minx, miny, maxx, maxy) of a window in CRS units."""
    from rasterio.windows import bounds
    left, bottom, right, top = bounds(window, transform)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def rasterize_labels(fire_geoms, notfire_geoms, out_shape, transform):
    """Label raster: 1 fire, 0 not fire, 255 no information.

    Where fire and not-fire overlap, fire wins; the number of overlapping
    pixels is returned so the caller can log it.
    """
    from rasterio.features import rasterize

    def _mask(geoms):
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            return np.zeros(out_shape, dtype=bool)
        return rasterize(
            [(g, 1) for g in geoms], out_shape=out_shape, transform=transform,
            fill=0, dtype="uint8", all_touched=False,
        ).astype(bool)

    fire = _mask(fire_geoms)
    notfire = _mask(notfire_geoms)
    y = np.full(out_shape, IGNORE, dtype=np.uint8)
    y[notfire] = 0
    y[fire] = 1
    return y, int((fire & notfire).sum())


def chip_quality(y, valid, min_labeled, max_nodata):
    """(keep, labeled_fraction, nodata_fraction) for one chip."""
    total = y.size
    labeled = float(((y != IGNORE) & valid).sum()) / total
    nodata = 1.0 - float(valid.sum()) / total
    return (labeled >= min_labeled and nodata <= max_nodata), labeled, nodata
