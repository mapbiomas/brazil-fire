"""Testes do scaffold (M0): import, paths, manifest e catalogo."""

import os

import mapbiomas_cog
from mapbiomas_cog.domain import config as cog_config
from mapbiomas_cog.domain import paths
from mapbiomas_cog.domain.models import Product, TaskSpec, Unit
from mapbiomas_cog.state.manifest import FailureJournal, Manifest, UnitState


def test_version():
    assert mapbiomas_cog.__version__


def test_paths_layout():
    root = paths.root("ecuador", "lulc", "collection_03", "coverage")
    assert root == "initiatives/ecuador/lulc/collection_03/coverage"
    assert paths.tiles_dir(root) == f"{root}/temp"
    assert paths.cog_object(root, "1985") == f"{root}/1985.tif"
    assert paths.manifest_object(root).endswith(paths.MANIFEST_NAME)
    assert paths.public_uri(root).startswith("gs://mapbiomas-public/")


def test_manifest_roundtrip(tmp_path):
    path = os.path.join(str(tmp_path), paths.MANIFEST_NAME)
    manifest = Manifest(path)
    manifest.upsert(UnitState(unit="2024_07", status="done", tiles_total=4, tiles_done=4))
    manifest.save()

    reloaded = Manifest(path)
    state = reloaded.get("2024_07")
    assert state is not None
    assert state.status == "done"
    assert state.tiles_total == 4
    assert reloaded.pending_units() == []


def test_failure_journal(tmp_path):
    path = os.path.join(str(tmp_path), paths.FAILURES_NAME)
    journal = FailureJournal(path)
    journal.record("2024_07", "429", tile="r0000_c0000")
    journal.save()
    assert len(FailureJournal(path).entries) == 1
    journal.clear()
    assert journal.entries == []


def test_task_spec_root():
    product = Product(
        country="ecuador",
        theme="lulc",
        collection="collection_03",
        product="coverage",
        assetid="projects/mapbiomas-public/assets/ecuador/lulc/collection3/coverage_v3",
    )
    task = TaskSpec(product=product, unit=Unit(key="1985", kind="annual"))
    assert task.unit_key == "1985"
    assert task.product.root == "initiatives/ecuador/lulc/collection_03/coverage"


def test_catalog_adapter():
    if not os.path.exists(cog_config.DEFAULT_CONFIG_PATH):
        return  # catalogo do version_01 ausente: teste opcional
    obj = cog_config.load_obj()
    assert "brasil" in obj
    countries = cog_config.resolve_countries(obj)
    assert countries and countries[0] == "brasil"
