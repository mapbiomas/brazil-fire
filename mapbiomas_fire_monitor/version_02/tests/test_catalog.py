"""Testes do catalogo e status (M5)."""

import os

from mapbiomas_cog.catalog import build_inventory, product_status, summary
from mapbiomas_cog.domain import paths
from mapbiomas_cog.state.manifest import Manifest, UnitState


def test_inventory_shape():
    inv = build_inventory(countries=["ecuador"])
    assert list(inv) == ["ecuador"]
    products = inv["ecuador"]["lulc"]["collection_03"]
    names = {p["product"] for p in products}
    assert {"coverage", "deforestation_secondary_vegetation"} <= names
    assert all(p["root"].startswith("initiatives/ecuador/lulc/collection_03/") for p in products)


def test_status_from_manifest(tmp_path):
    root = "initiatives/ecuador/lulc/collection_03/coverage"
    path = paths.local_state_path(root, state_dir=str(tmp_path))
    manifest = Manifest(path)
    manifest.upsert(UnitState(unit="1985", status="done", tiles_total=2, tiles_done=2))
    manifest.upsert(UnitState(unit="1986", status="submitted", tiles_total=2, tiles_done=1))
    manifest.save()

    status = product_status(root, state_dir=str(tmp_path))
    assert status["1985"]["status"] == "done"
    assert summary(root, state_dir=str(tmp_path)) == {"done": 1, "submitted": 1}
    assert os.path.exists(path)
