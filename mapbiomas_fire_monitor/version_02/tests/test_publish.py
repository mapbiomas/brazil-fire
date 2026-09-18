"""Testes da publicacao no bucket publico (M5, com fs fake)."""

from mapbiomas_cog.domain import paths
from mapbiomas_cog.publish.gcs import publish_cog

ROOT = "initiatives/ecuador/lulc/collection_03/coverage"
SRC = f"{paths.BUCKET}/{paths.cog_object(ROOT, '1985')}"
DST = f"{paths.PUBLIC_BUCKET}/{paths.cog_object(ROOT, '1985')}"


class FakeFS:
    def __init__(self, files=None):
        self.files = dict(files or {})

    def info(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return {"size": self.files[path]}

    def copy(self, src, dst):
        self.files[dst] = self.files[src]


def test_publish_copies():
    fs = FakeFS({SRC: 123})
    assert publish_cog(ROOT, "1985", fs=fs) == "copied"
    assert fs.files[DST] == 123


def test_publish_exists():
    fs = FakeFS({SRC: 10, DST: 10})
    assert publish_cog(ROOT, "1985", fs=fs) == "exists"


def test_publish_force_overwrites():
    fs = FakeFS({SRC: 10, DST: 5})
    assert publish_cog(ROOT, "1985", fs=fs) == "copied"
    assert fs.files[DST] == 10


def test_publish_missing_source():
    assert publish_cog(ROOT, "1985", fs=FakeFS()) == "error"
