"""Planner: decompoe uma unidade em um TileGrid deterministico."""

from __future__ import annotations

from ..domain.grid import DEFAULT_CRS, DEFAULT_TILE_SIZE_PX, TileGrid, plan_tiles
from ..domain.models import TaskSpec
from .images import resolve_image


def plan(
    task: TaskSpec,
    tile_size_px: int = DEFAULT_TILE_SIZE_PX,
    crs: str = DEFAULT_CRS,
) -> TileGrid:
    """Planeja o grid de tiles de uma TaskSpec.

    Resolve a imagem da unidade no EE, projeta o footprint no CRS alvo e
    devolve o grid alinhado a origem. Se a task ja traz um grid, respeita-o.
    """
    if task.grid is not None:
        return task.grid
    image = resolve_image(task.product.assetid, task.unit_key, task.product.kind)
    if image is None:
        raise ValueError(f"Sem imagem para a unidade {task.unit_key!r}.")
    return plan_tiles(
        image.geometry(),
        scale=task.product.scale,
        crs=crs,
        tile_size_px=tile_size_px,
    )
