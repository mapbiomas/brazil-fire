"""Status por unidade/produto derivado do manifest (M5).

Sem glob no GCS: o estado vem do ``_manifest.json`` (``state.manifest``).
"""

from __future__ import annotations

from ..domain import paths
from ..state.manifest import Manifest


def unit_status(manifest_path: str, unit: str) -> dict:
    state = Manifest(manifest_path).get(unit)
    if state is None:
        return {"unit": unit, "status": "unknown"}
    return {
        "unit": state.unit,
        "status": state.status,
        "tiles_total": state.tiles_total,
        "tiles_done": state.tiles_done,
        "failed_tiles": state.failed_tiles(),
        "cog": state.cog,
    }


def product_status(root_path: str, units=None, state_dir: str | None = None) -> dict:
    """Status das unidades de um produto (a partir do manifest local)."""
    data = Manifest(paths.local_state_path(root_path, state_dir)).data.get("units", {})
    if units is None:
        return data
    return {u: data.get(u, {"status": "unknown"}) for u in units}


def summary(root_path: str, state_dir: str | None = None) -> dict:
    """Contagem por status de um produto."""
    counts: dict = {}
    for entry in product_status(root_path, state_dir=state_dir).values():
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts
