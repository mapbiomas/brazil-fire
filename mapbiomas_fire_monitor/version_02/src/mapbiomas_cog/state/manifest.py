"""Manifest por unidade + journal de falhas (stdlib apenas).

``_manifest.json`` guarda o progresso de cada unidade (tiles previstos/feitos,
COG, status). ``_failures.json`` e o journal de falhas usado pelo ``retry``.
Esta e a fonte de estado do mapbiomas_cog — substitui o scan por glob do
version_01.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field


@dataclass
class UnitState:
    unit: str
    status: str = "pending"  # pending | submitted | done | failed
    tiles_total: int = 0
    tiles_done: int = 0
    cog: str = ""
    error: str = ""
    updated_at: str = ""
    tiles: dict = field(default_factory=dict)       # tile_key -> status
    tile_tasks: dict = field(default_factory=dict)  # tile_key -> ee task id

    def touch(self) -> "UnitState":
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self

    def mark_tile(self, key: str, status: str, task_id: str | None = None) -> "UnitState":
        self.tiles[key] = status
        if task_id:
            self.tile_tasks[key] = task_id
        self.tiles_total = len(self.tiles)
        self.tiles_done = sum(1 for s in self.tiles.values() if s == "done")
        return self

    def failed_tiles(self) -> list:
        return [k for k, s in self.tiles.items() if s == "failed"]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Manifest:
    """Leitura/escrita do manifest e do journal de falhas (arquivos JSON)."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {"units": {}, "updated_at": _now()}
        if os.path.exists(path):
            self.load()

    def load(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                self.data = loaded
                self.data.setdefault("units", {})
        except (OSError, json.JSONDecodeError):
            pass
        return self.data

    def save(self) -> None:
        self.data["updated_at"] = _now()
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)

    def get(self, unit: str) -> UnitState | None:
        entry = self.data.get("units", {}).get(unit)
        return UnitState(**entry) if entry else None

    def upsert(self, state: UnitState) -> UnitState:
        state.touch()
        self.data.setdefault("units", {})[state.unit] = asdict(state)
        return state

    def pending_units(self) -> list:
        return [
            u
            for u, e in self.data.get("units", {}).items()
            if e.get("status") != "done"
        ]


class FailureJournal:
    """Journal de falhas (``_failures.json``) para ``retry`` idempotente."""

    def __init__(self, path: str):
        self.path = path
        self.entries: list = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    self.entries = loaded
            except (OSError, json.JSONDecodeError):
                pass

    def record(self, unit: str, reason: str, tile: str = "") -> None:
        self.entries.append(
            {"unit": unit, "tile": tile, "reason": reason, "at": _now()}
        )

    def save(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, ensure_ascii=False)

    def clear(self, unit: str | None = None) -> None:
        if unit is None:
            self.entries = []
        else:
            self.entries = [e for e in self.entries if e.get("unit") != unit]
