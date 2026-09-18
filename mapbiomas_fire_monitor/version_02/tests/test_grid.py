"""Testes do grid deterministico e do estado por tile (M1/M2, sem EE)."""

from mapbiomas_cog.domain.grid import compute_grid, tile_crs_transform, tile_dimensions
from mapbiomas_cog.domain.models import product_kind, unit_key_from_time_start
from mapbiomas_cog.state.manifest import UnitState


def test_compute_grid_alignment():
    # 1000m/px * 100px = 100_000m por tile; box cobre 2.5 tiles em x e 1.5 em y.
    grid = compute_grid(0, 0, 250_000, 150_000, scale=1000, tile_size_px=100, crs="EPSG:6933")
    assert grid.crs == "EPSG:6933"
    assert grid.tile_size_px == 100
    assert grid.n_tiles == 3 * 2  # colunas 0..2, linhas 0..1
    assert grid.keys()[0] == "r0000_c0000"
    # bounds alinhados a origem
    t = grid.tiles[0]
    assert t.bounds == (0.0, 0.0, 100_000.0, 100_000.0)


def test_compute_grid_negative_origin():
    grid = compute_grid(-250_000, -150_000, 0, 0, scale=1000, tile_size_px=100)
    keys = set(grid.keys())
    assert "r-002_c-003" in keys  # {-2:04d} -> "-002"
    assert "r0000_c0000" in keys


def test_tile_crs_transform():
    grid = compute_grid(0, 0, 100_000, 100_000, scale=1000, tile_size_px=100)
    tile = grid.tiles[0]
    assert tile_crs_transform(tile, grid) == [1000, 0, 0.0, 0, -1000, 100_000.0]
    assert tile_dimensions(grid) == "100x100"


def test_product_kind_and_unit_key():
    assert product_kind("monthly_burned") == "monthly"
    assert product_kind("annual_burned") == "annual"
    assert product_kind("coverage") == "period"
    # 2024-07-01T00:00:00Z em ms
    assert unit_key_from_time_start("monthly", 1719792000000) == "2024_07"
    assert unit_key_from_time_start("annual", 1719792000000) == "2024"


def test_unit_state_tiles():
    state = UnitState(unit="2024_07")
    state.mark_tile("r0000_c0000", "done")
    state.mark_tile("r0000_c0001", "failed")
    state.mark_tile("r0001_c0000", "submitted", task_id="abc")
    assert state.tiles_total == 3
    assert state.tiles_done == 1
    assert state.failed_tiles() == ["r0000_c0001"]
    assert state.tile_tasks["r0001_c0000"] == "abc"
