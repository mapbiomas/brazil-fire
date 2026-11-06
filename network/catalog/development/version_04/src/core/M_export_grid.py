"""Deterministic pixel-aligned grids for experimental exports.

The caller supplies bounds already transformed to the target projected CRS.
This module deliberately does not choose a CRS or export region: those choices
must be validated against existing mosaics before an exporter uses this grid.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Tile:
    column: int
    row: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def id(self):
        return f"c{self.column:+d}_r{self.row:+d}"


def tiles_for_bounds(bounds, pixel_size, tile_pixels, origin=(0, 0)):
    """Cover projected bounds with full, non-overlapping, origin-anchored tiles.

    Bounds are (xmin, ymin, xmax, ymax), in the same units as pixel_size.
    A tile only touching the maximum edge is excluded. Geometry intersection
    and masking are the caller's responsibility.
    """
    xmin, ymin, xmax, ymax = map(float, bounds)
    ox, oy = map(float, origin)
    pixel_size = float(pixel_size)
    if (not all(math.isfinite(v) for v in (xmin, ymin, xmax, ymax, ox, oy, pixel_size))
            or pixel_size <= 0 or xmin >= xmax or ymin >= ymax
            or isinstance(tile_pixels, bool) or not isinstance(tile_pixels, int)
            or tile_pixels <= 0):
        raise ValueError('Invalid bounds, origin, pixel size or tile dimensions')
    width = pixel_size * tile_pixels
    if not math.isfinite(width):
        raise ValueError('Tile size must be finite')
    first_col = math.floor((xmin - ox) / width)
    last_col = math.ceil((xmax - ox) / width)
    first_row = math.floor((ymin - oy) / width)
    last_row = math.ceil((ymax - oy) / width)
    for row in range(first_row, last_row):
        for col in range(first_col, last_col):
            left, bottom = ox + col * width, oy + row * width
            yield Tile(col, row, left, bottom, left + width, bottom + width)


def filter_intersecting_tiles(tiles, geojson):
    """Drop tiles far outside a country, keeping a conservative boundary buffer.

    Coordinates must use the same geographic CRS as the tile grid. A full tile
    of padding reduces the chance of dropping boundary pixels when translating
    Earth Engine's geodesic geometry into planar geometry. Never repairs an
    invalid geometry silently.
    """
    from shapely.geometry import shape, box
    from shapely.prepared import prep
    tiles = list(tiles)
    if not tiles:
        return []
    area = shape(geojson)
    if area.is_empty or not area.is_valid or area.geom_type not in ('Polygon', 'MultiPolygon', 'GeometryCollection'):
        raise ValueError('Country geometry must be a valid polygon or geometry collection')
    padding = max(tiles[0].xmax - tiles[0].xmin, tiles[0].ymax - tiles[0].ymin)
    buffered = prep(area.buffer(padding))
    return [tile for tile in tiles
            if buffered.intersects(box(tile.xmin, tile.ymin, tile.xmax, tile.ymax))]
