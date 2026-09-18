"""Testes da matematica de assembly do COG (M3, sem rasterio)."""

from mapbiomas_cog.assemble.cog import output_geometry


def test_output_geometry_union():
    bounds = [
        (0.0, 0.0, 100.0, 100.0),
        (100.0, 0.0, 200.0, 100.0),
        (0.0, 100.0, 100.0, 200.0),
    ]
    xmin, ymax, width, height = output_geometry(bounds, scale=1.0)
    assert (xmin, ymax) == (0.0, 200.0)
    assert (width, height) == (200, 200)


def test_output_geometry_scale():
    bounds = [(0.0, 0.0, 30000.0, 30000.0)]
    xmin, ymax, width, height = output_geometry(bounds, scale=30.0)
    assert (width, height) == (1000, 1000)
