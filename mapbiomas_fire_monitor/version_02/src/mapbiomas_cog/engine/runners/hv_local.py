"""Runner ``hv_local``: High Volume API em Python puro (tecnica do datensee).

Busca cada tile via ``ee.data.computePixels`` (GEO_TIFF) em um ThreadPool, com
backoff exponencial em 429/erros transitorios, grava o tile no GCS e registra
progresso/falhas no manifest. Sem Java/Dataflow.

O COG final e montado depois pelo ``assemble`` (M3), de forma uniforme com o
runner ``ee_batch_tiled``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..base import RunResult, RunnerUnavailable
from .. import images, planner
from ...domain import paths
from ...domain.models import TaskSpec
from ...state.manifest import FailureJournal, Manifest, UnitState

_RETRYABLE_MARKERS = ("429", "rate limit", "too many requests", "quota", "503", "502")


def is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def with_backoff(fn, *, retries: int = 5, base_delay: float = 1.0,
                 sleep=time.sleep, retryable=is_retryable, log=None):
    """Executa ``fn`` com backoff exponencial para erros transitorios."""
    delay = base_delay
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries or not retryable(exc):
                raise
            if log:
                log(f"[HV] retry {attempt + 1}/{retries} em {delay:.1f}s: {exc}")
            sleep(delay)
            delay *= 2
    raise RuntimeError("inalcancavel")  # pragma: no cover


class HvLocalRunner:
    name = "hv_local"

    def __init__(self, bucket: str | None = None, state_dir: str | None = None,
                 max_workers: int = 8, retries: int = 5, base_delay: float = 1.0,
                 log=None, fs=None, sleep=None):
        self.bucket = bucket or paths.BUCKET
        self.state_dir = state_dir
        self.max_workers = max(1, int(max_workers))
        self.retries = retries
        self.base_delay = base_delay
        self.log = log or (lambda _m: None)
        self.fs = fs
        self.sleep = sleep or time.sleep

    # -- helpers -------------------------------------------------------------
    def _get_fs(self):
        if self.fs is None:
            import gcsfs

            self.fs = gcsfs.GCSFileSystem(token="google_default")
        return self.fs

    def _manifest(self, task: TaskSpec) -> Manifest:
        return Manifest(paths.local_state_path(task.product.root, self.state_dir))

    def _tile_object(self, task: TaskSpec, tile_key: str) -> str:
        return f"{paths.tiles_dir(task.product.root)}/{task.unit_key}_{tile_key}.tif"

    def _compute_tile(self, image, tile, grid) -> bytes:
        import ee

        request = {
            "expression": image,
            "fileFormat": "GEO_TIFF",
            "grid": {
                "dimensions": {"width": grid.tile_size_px, "height": grid.tile_size_px},
                "affineTransform": {
                    "scaleX": grid.scale,
                    "shearX": 0,
                    "translateX": tile.xmin,
                    "shearY": 0,
                    "scaleY": -grid.scale,
                    "translateY": tile.ymax,
                },
                "crsCode": grid.crs,
            },
        }
        return ee.data.computePixels(request)

    # -- API -----------------------------------------------------------------
    def run(self, task: TaskSpec) -> RunResult:
        import ee

        if not hasattr(ee.data, "computePixels"):
            raise RunnerUnavailable("ee.data.computePixels indisponivel nesta versao do EE.")

        grid = task.grid or planner.plan(task)
        image = images.resolve_image(task.product.assetid, task.unit_key, task.product.kind)
        if image is None:
            raise ValueError(f"Sem imagem para a unidade {task.unit_key!r}.")

        manifest = self._manifest(task)
        state = manifest.get(task.unit_key) or UnitState(unit=task.unit_key)
        journal = FailureJournal(paths.local_failures_path(task.product.root, self.state_dir))
        fs = self._get_fs()

        pending = [t for t in grid.tiles
                   if task.force or state.tiles.get(t.key) != "done"]

        def _work(tile):
            data = with_backoff(
                lambda: self._compute_tile(image, tile, grid),
                retries=self.retries,
                base_delay=self.base_delay,
                sleep=self.sleep,
                log=self.log,
            )
            fs.pipe(f"{self.bucket}/{self._tile_object(task, tile.key)}", data)
            return tile.key

        done = 0
        if pending:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = {pool.submit(_work, t): t for t in pending}
                for future in as_completed(futures):
                    tile = futures[future]
                    try:
                        key = future.result()
                        state.mark_tile(key, "done")
                        done += 1
                        self.log(f"[HV] {task.unit_key}/{key} ok")
                    except Exception as exc:  # noqa: BLE001
                        state.mark_tile(tile.key, "failed")
                        state.error = str(exc)
                        journal.record(task.unit_key, str(exc), tile.key)
                        self.log(f"[ERROR] {task.unit_key}/{tile.key}: {exc}")

        journal.save()
        if state.tiles and all(s == "done" for s in state.tiles.values()):
            state.status = "done"
        elif state.failed_tiles():
            state.status = "failed"
        else:
            state.status = "done"
        manifest.upsert(state)
        manifest.save()

        return RunResult(
            unit=task.unit_key,
            ok=not state.failed_tiles(),
            submitted=done,
            done=state.tiles_done,
            failed=len(state.failed_tiles()),
            detail=f"{state.tiles_done}/{state.tiles_total} tiles",
        )

    def sync(self, task: TaskSpec) -> RunResult:
        """hv_local e sincrono: apenas reporta o estado atual do manifest."""
        state = self._manifest(task).get(task.unit_key)
        if state is None:
            return RunResult(unit=task.unit_key, ok=True, detail="sem estado")
        return RunResult(
            unit=task.unit_key,
            ok=not state.failed_tiles(),
            done=state.tiles_done,
            failed=len(state.failed_tiles()),
            detail=f"status={state.status} ({state.tiles_done}/{state.tiles_total})",
        )

    def retry(self, task: TaskSpec) -> RunResult:
        manifest = self._manifest(task)
        state = manifest.get(task.unit_key)
        if state is not None:
            for tile_key in state.failed_tiles():
                state.tiles.pop(tile_key, None)
                state.tile_tasks.pop(tile_key, None)
            state.error = ""
            manifest.upsert(state)
            manifest.save()
        return self.run(task)
