"""Facade do engine e registro de runners.

Um ``Runner`` recebe um ``TaskSpec`` e e responsavel por produzir os tiles da
unidade, atualizando o manifest. Runners:

- ``ee_batch_tiled`` (default) — ``Export.image.toCloudStorage`` por tile.
- ``hv_local``       — High Volume API (``computePixels``) em Python puro.
- ``dataflow``       — plugin opcional (futuro).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..domain.models import TaskSpec


class RunnerUnavailable(RuntimeError):
    """Runner nao pode executar aqui (permite fallback para outro runner)."""


@dataclass
class RunResult:
    unit: str
    ok: bool = True
    submitted: int = 0
    done: int = 0
    failed: int = 0
    detail: str = ""


@runtime_checkable
class Runner(Protocol):
    name: str

    def run(self, task: TaskSpec) -> RunResult:  # pragma: no cover - protocolo
        ...


_RUNNERS: dict = {}


def register_runner(name: str):
    """Decorator para registrar um runner no registry global."""

    def _wrap(cls):
        _RUNNERS[name] = cls
        return cls

    return _wrap


def get_runner(name: str = "ee_batch_tiled", **kwargs):
    """Instancia um runner pelo nome (import lazy para evitar deps no import)."""
    if name in ("ee_batch_tiled", "ee_batch"):
        from .runners.ee_batch import EeBatchTiledRunner

        return EeBatchTiledRunner(**kwargs)
    if name == "hv_local":
        from .runners.hv_local import HvLocalRunner

        return HvLocalRunner(**kwargs)
    if name in _RUNNERS:
        return _RUNNERS[name](**kwargs)
    raise ValueError(
        f"Runner '{name}' desconhecido. Disponiveis: ee_batch_tiled, hv_local "
        "(dataflow previsto para fase futura)."
    )


class Engine:
    """Facade estavel consumido pela UI/CLI/notebook."""

    def __init__(self, runner: str = "ee_batch_tiled", manifest_store=None,
                 fallback: str | None = None, **runner_kwargs):
        self.runner_name = runner
        self.fallback = fallback
        self._runner_kwargs = runner_kwargs
        self.runner = get_runner(runner, **runner_kwargs)
        self.manifest_store = manifest_store

    def _switch(self, name: str) -> None:
        self.runner_name = name
        self.runner = get_runner(name, **self._runner_kwargs)

    def _call(self, method: str, task: TaskSpec) -> RunResult:
        fn = getattr(self.runner, method, None)
        if fn is None:
            raise NotImplementedError(f"Runner '{self.runner_name}' nao suporta {method}.")
        try:
            return fn(task)
        except RunnerUnavailable:
            if self.fallback and self.fallback != self.runner_name:
                self._switch(self.fallback)
                fn = getattr(self.runner, method, None)
                if fn is None:
                    raise
                return fn(task)
            raise

    def run(self, task: TaskSpec) -> RunResult:
        return self._call("run", task)

    def sync(self, task: TaskSpec) -> RunResult:
        return self._call("sync", task)

    def retry(self, task: TaskSpec) -> RunResult:
        return self._call("retry", task)
