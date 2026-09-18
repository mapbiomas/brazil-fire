"""Fakes para testar os runners sem rede (M7).

``install_fake_ee`` injeta um modulo ``ee`` falso em ``sys.modules``; ``FakeFS``
implementa o subconjunto de ``gcsfs`` usado pelos runners.
"""

from __future__ import annotations

import contextlib
import sys
import types


class FakeFS:
    def __init__(self, files=None):
        self.files = dict(files or {})
        self.blobs = {}

    def info(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return {"size": self.files[path]}

    def copy(self, src, dst):
        self.files[dst] = self.files[src]

    def pipe(self, path, data):
        self.blobs[path] = data
        self.files[path] = len(data)


class _FakeJob:
    def __init__(self, job_id: str):
        self.id = job_id

    def start(self):
        return self


class _FakeImage:
    def __init__(self, assetid):
        self.assetid = assetid

    def select(self, *_a, **_k):
        return self

    def rename(self, *_a, **_k):
        return self

    def geometry(self):
        return self


class _FakeData:
    def __init__(self, owner):
        self._owner = owner

    def getAsset(self, assetid):
        return {"type": "IMAGE"}

    def getTaskStatus(self, ids):
        return [{"id": i, "state": self._owner.states.get(i, "READY")} for i in ids]

    def computePixels(self, request):
        return b"GEO_TIFF_BYTES"


class FakeEE:
    def __init__(self):
        self._counter = 0
        self.states = {}
        self.data = _FakeData(self)
        self.batch = types.SimpleNamespace(
            Export=types.SimpleNamespace(
                image=types.SimpleNamespace(toCloudStorage=self._to_cloud_storage)
            )
        )

    def _next_id(self) -> str:
        self._counter += 1
        return f"task{self._counter}"

    def _to_cloud_storage(self, **_kwargs):
        job_id = self._next_id()
        self.states[job_id] = "READY"
        return _FakeJob(job_id)

    def set_states(self, mapping):
        self.states.update(mapping)

    def Image(self, assetid):
        return _FakeImage(assetid)

    def ImageCollection(self, assetid):  # pragma: no cover - nao usado nos testes
        raise NotImplementedError


def make_task(unit: str = "1985"):
    """TaskSpec de teste com grid fixo (6 tiles), sem tocar o EE."""
    from mapbiomas_cog.domain.grid import compute_grid
    from mapbiomas_cog.domain.models import Product, TaskSpec, Unit

    product = Product(
        country="ecuador",
        theme="lulc",
        collection="collection_03",
        product="coverage",
        assetid="projects/mapbiomas-public/assets/ecuador/lulc/collection3/coverage_v3",
        dtype="byte",
        scale=1000,
        kind="annual",
    )
    grid = compute_grid(0, 0, 200_000, 100_000, scale=1000, tile_size_px=100)
    return TaskSpec(product=product, unit=Unit(unit, "annual"), grid=grid)


@contextlib.contextmanager
def install_fake_ee():
    fake = FakeEE()
    previous = sys.modules.get("ee")
    sys.modules["ee"] = fake
    try:
        yield fake
    finally:
        if previous is None:
            sys.modules.pop("ee", None)
        else:
            sys.modules["ee"] = previous
