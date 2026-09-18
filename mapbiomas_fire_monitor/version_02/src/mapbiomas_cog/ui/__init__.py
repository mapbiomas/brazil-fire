"""UI ipywidgets do mapbiomas_cog (import lazy para nao exigir ipywidgets)."""

from __future__ import annotations

__all__ = ["run_ui"]


def run_ui(countries=None, themes=None, runner: str = "ee_batch_tiled",
           config_path: str | None = None):
    """Abre a interface de navegacao + engine (export/sync/assemble)."""
    from .app import run_ui as _run_ui

    return _run_ui(countries=countries, themes=themes, runner=runner, config_path=config_path)
