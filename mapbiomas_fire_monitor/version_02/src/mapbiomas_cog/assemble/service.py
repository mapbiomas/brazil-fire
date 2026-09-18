"""Orquestracao do assembly de uma unidade (GCS -> COG -> GCS)."""

from __future__ import annotations

import os
import shutil
import tempfile

from ..domain import paths
from ..domain.models import TaskSpec
from ..state.manifest import Manifest, UnitState
from .cog import assemble_cog


def assemble_unit(task: TaskSpec, work_dir: str | None = None, force: bool = False,
                  log=None, state_dir: str | None = None) -> str:
    """Baixa os tiles da unidade, monta o COG e sobe para o bucket.

    Atualiza o manifest (``state.cog``/``status``). Idempotente: se o COG ja
    existe no manifest e ``force`` e False, retorna sem refazer.
    """
    import gcsfs

    _log = log or (lambda _m: None)
    root = task.product.root
    manifest = Manifest(paths.local_state_path(root, state_dir))
    state = manifest.get(task.unit_key) or UnitState(unit=task.unit_key)
    if state.cog and not force:
        _log(f"[ASSEMBLE] {task.unit_key}: COG ja registrado ({state.cog}).")
        return state.cog

    fs = gcsfs.GCSFileSystem(token="google_default")
    pattern = f"{paths.BUCKET}/{paths.tiles_dir(root)}/{task.unit_key}_*.tif"
    tiles = sorted(fs.glob(pattern))
    if not tiles:
        raise FileNotFoundError(f"Nenhum tile encontrado: {pattern}")

    work = tempfile.mkdtemp(dir=work_dir, prefix=f"cog_{task.unit_key}_")
    try:
        local = []
        for remote in tiles:
            dst = os.path.join(work, os.path.basename(remote))
            _log(f"[ASSEMBLE] download {remote}")
            fs.get(remote, dst)
            local.append(dst)

        output = os.path.join(work, f"{task.unit_key}.tif")
        assemble_cog(local, output, nodata=0)
        _log(f"[ASSEMBLE] COG montado ({len(local)} tiles) -> {output}")

        dest = f"{paths.BUCKET}/{paths.cog_object(root, task.unit_key)}"
        fs.put(output, dest)
        _log(f"[ASSEMBLE] gs://{dest}")

        state.cog = paths.cog_object(root, task.unit_key)
        if state.tiles_total and state.tiles_done >= state.tiles_total:
            state.status = "done"
        manifest.upsert(state)
        manifest.save()
        return state.cog
    finally:
        shutil.rmtree(work, ignore_errors=True)
