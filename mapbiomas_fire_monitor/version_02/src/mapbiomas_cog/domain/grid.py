"""Grid de tiles deterministico (inspirado no datensee).

O grid e ancorado na origem do CRS e endereçado por coordenadas inteiras de
pixel, garantindo alinhamento entre exports e resumibilidade. A matematica do
grid e Python puro (testavel sem EE); apenas ``plan_tiles`` toca o EE, para
projetar o bounding box no CRS alvo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

# CRS equal-area global (metros), bom default para tiling multi-pais.
DEFAULT_CRS = "EPSG:6933"
DEFAULT_TILE_SIZE_PX = 4096


@dataclass(frozen=True)
class Tile:
    """Um tile do grid, endereçado por (row, col) e com bounds no CRS."""

    row: int
    col: int
    bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)

    @property
    def key(self) -> str:
        return f"r{self.row:04d}_c{self.col:04d}"

    @property
    def xmin(self) -> float:
        return self.bounds[0]

    @property
    def ymin(self) -> float:
        return self.bounds[1]

    @property
    def xmax(self) -> float:
        return self.bounds[2]

    @property
    def ymax(self) -> float:
        return self.bounds[3]


@dataclass(frozen=True)
class TileGrid:
    """Especificacao do grid de tiles de uma unidade."""

    crs: str
    scale: float
    tile_size_px: int
    tiles: Tuple[Tile, ...] = field(default_factory=tuple)

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    def keys(self) -> list:
        return [t.key for t in self.tiles]


def compute_grid(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    scale: float,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
    crs: str = DEFAULT_CRS,
) -> TileGrid:
    """Decompoe um bounding box projetado em tiles alinhados a origem (0, 0).

    Python puro: recebe coordenadas ja no CRS alvo (metros).
    """
    if scale <= 0:
        raise ValueError("scale deve ser > 0")
    if tile_size_px <= 0:
        raise ValueError("tile_size_px deve ser > 0")
    tile_units = scale * tile_size_px

    col0 = math.floor(xmin / tile_units)
    col1 = math.floor(xmax / tile_units)
    row0 = math.floor(ymin / tile_units)
    row1 = math.floor(ymax / tile_units)

    tiles = []
    for row in range(row0, row1 + 1):
        for col in range(col0, col1 + 1):
            x0 = col * tile_units
            y0 = row * tile_units
            tiles.append(Tile(row=row, col=col, bounds=(x0, y0, x0 + tile_units, y0 + tile_units)))

    return TileGrid(crs=crs, scale=scale, tile_size_px=tile_size_px, tiles=tuple(tiles))


def tile_crs_transform(tile: Tile, grid: TileGrid) -> list:
    """crsTransform (GDAL/EE, 6 valores) do tile: [s, 0, x0, 0, -s, y_top]."""
    s = grid.scale
    return [s, 0, tile.xmin, 0, -s, tile.ymax]


def tile_dimensions(grid: TileGrid) -> str:
    """Dimensoes EE do tile: 'WxH'."""
    return f"{grid.tile_size_px}x{grid.tile_size_px}"


def projected_bounds(region, crs: str = DEFAULT_CRS) -> Tuple[float, float, float, float]:
    """Bounding box de ``region`` no CRS alvo, via EE (retorna em metros)."""
    import ee

    geom = region if isinstance(region, ee.Geometry) else ee.Geometry(region)
    coords = geom.transform(ee.Projection(crs), maxError=1).bounds().coordinates().getInfo()[0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def plan_tiles(
    region,
    scale: float,
    crs: str = DEFAULT_CRS,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
) -> TileGrid:
    """Projeta ``region`` no CRS alvo e devolve o TileGrid alinhado."""
    xmin, ymin, xmax, ymax = projected_bounds(region, crs)
    return compute_grid(xmin, ymin, xmax, ymax, scale, tile_size_px, crs)
