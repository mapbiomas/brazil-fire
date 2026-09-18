"""Runner default: export tiled via ``ee.batch.Export.image.toCloudStorage``.

Mantem a infra/custo do version_01, mas com:
- tiling deterministico (um tile por task, alinhado por crsTransform);
- manifest/estado por unidade e por tile;
- ``sync`` (poll das tasks) e ``retry`` (reenvia apenas tiles falhos).

Os imports de ``ee`` sao lazy para manter o pacote importavel sem GEE.
"""

from __future__ import annotations

from ..base import RunResult
from .. import images, planner
from ...domain.grid import tile_crs_transform, tile_dimensions
from ...domain.models import TaskSpec
from ...domain import paths
from ...state.manifest import Manifest, UnitState

_TERMINAL_OK = {"COMPLETED"}
_TERMINAL_FAIL = {"FAILED", "CANCELLED", "CANCEL_REQUESTED"}


class EeBatchTiledRunner:
    name = "ee_batch_tiled"

    def __init__(self, bucket: str | None = None, state_dir: str | None = None,
                 export_flag: str = "COG_", log=None):
        self.bucket = bucket or paths.BUCKET
        self.state_dir = state_dir
        self.export_flag = export_flag
        self.log = log or (lambda _msg: None)

    # -- helpers -------------------------------------------------------------
    def _manifest(self, task: TaskSpec) -> Manifest:
        return Manifest(paths.local_state_path(task.product.root, self.state_dir))

    def _tile_prefix(self, task: TaskSpec) -> str:
        return f"{paths.tiles_dir(task.product.root)}/{task.unit_key}"

    def _description(self, task: TaskSpec, tile_key: str) -> str:
        tag = f"{task.product.country}_{task.product.product}_{task.unit_key}_{tile_key}"
        return f"{self.export_flag}{tag}"

    def _state(self, manifest: Manifest, task: TaskSpec) -> UnitState:
        return manifest.get(task.unit_key) or UnitState(unit=task.unit_key)

    # -- API -----------------------------------------------------------------
    def run(self, task: TaskSpec) -> RunResult:
        """Submete as tasks dos tiles pendentes e atualiza o manifest."""
        import ee

        grid = task.grid or planner.plan(task)
        image = images.resolve_image(task.product.assetid, task.unit_key, task.product.kind)
        if image is None:
            raise ValueError(f"Sem imagem para a unidade {task.unit_key!r}.")

        manifest = self._manifest(task)
        state = self._state(manifest, task)

        submitted = 0
        for tile in grid.tiles:
            status = state.tiles.get(tile.key)
            if status in ("submitted", "running", "done") and not task.force:
                continue
            prefix = f"{self._tile_prefix(task)}_{tile.key}_"
            job = ee.batch.Export.image.toCloudStorage(
                image=image,
                description=self._description(task, tile.key),
                bucket=self.bucket,
                fileNamePrefix=prefix,
                crs=grid.crs,
                crsTransform=tile_crs_transform(tile, grid),
                dimensions=tile_dimensions(grid),
                maxPixels=1e13,
                fileFormat="GeoTIFF",
                formatOptions={"cloudOptimized": True},
            )
            job.start()
            state.mark_tile(tile.key, "submitted", job.id)
            submitted += 1
            self.log(f"[EE] {task.unit_key}/{tile.key} -> {job.id}")

        state.status = "submitted" if submitted else state.status
        manifest.upsert(state)
        manifest.save()
        return RunResult(
            unit=task.unit_key,
            ok=True,
            submitted=submitted,
            done=state.tiles_done,
            failed=len(state.failed_tiles()),
            detail=f"{state.tiles_done}/{state.tiles_total} tiles done",
        )

    def sync(self, task: TaskSpec) -> RunResult:
        """Consulta o estado das tasks no EE e atualiza o manifest."""
        import ee

        manifest = self._manifest(task)
        state = self._state(manifest, task)
        ids = list(state.tile_tasks.values())
        statuses = ee.data.getTaskStatus(ids) if ids else []
        by_id = {s.get("id"): s for s in statuses}

        for tile_key, task_id in state.tile_tasks.items():
            task_state = by_id.get(task_id, {}).get("state", "")
            if task_state in _TERMINAL_OK:
                state.mark_tile(tile_key, "done")
            elif task_state in _TERMINAL_FAIL:
                state.mark_tile(tile_key, "failed")
                state.error = by_id.get(task_id, {}).get("error_message", state.error)
            elif task_state == "RUNNING":
                state.mark_tile(tile_key, "running")

        if state.tiles and all(s == "done" for s in state.tiles.values()):
            state.status = "done"
        elif state.failed_tiles():
            state.status = "failed"
        manifest.upsert(state)
        manifest.save()
        return RunResult(
            unit=task.unit_key,
            ok=state.status != "failed",
            submitted=0,
            done=state.tiles_done,
            failed=len(state.failed_tiles()),
            detail=f"status={state.status} ({state.tiles_done}/{state.tiles_total})",
        )

    def retry(self, task: TaskSpec) -> RunResult:
        """Remove os tiles falhos do manifest e reenvia (via ``run``)."""
        manifest = self._manifest(task)
        state = self._state(manifest, task)
        for tile_key in state.failed_tiles():
            state.tiles.pop(tile_key, None)
            state.tile_tasks.pop(tile_key, None)
        state.error = ""
        manifest.upsert(state)
        manifest.save()
        return self.run(task)
