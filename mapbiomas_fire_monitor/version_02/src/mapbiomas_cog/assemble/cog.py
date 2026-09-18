"""Monta o COG final de uma unidade a partir dos tiles (M3).

Substitui ``gdalbuildvrt`` + ``gdal_translate`` do version_01 por rasterio:
escreve os tiles em uma GTiff temporaria e a converte para COG. A matematica do
bounding box de saida e Python puro (``output_geometry``), testavel sem rasterio.
"""

from __future__ import annotations

import os

DEFAULT_COMPRESS = "DEFLATE"
DEFAULT_PREDICTOR = 2  # 2 = horizontal (int); 3 = floating point


def output_geometry(bounds_list, scale: float):
    """Uniao dos bounds -> (xmin, ymax, width, height) no grid de ``scale``.

    ``bounds_list``: iteravel de (xmin, ymin, xmax, ymax).
    """
    if not bounds_list:
        raise ValueError("bounds_list vazio")
    if scale <= 0:
        raise ValueError("scale deve ser > 0")
    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)
    width = int(round((xmax - xmin) / scale))
    height = int(round((ymax - ymin) / scale))
    return xmin, ymax, width, height


def assemble_cog(
    tiles,
    output: str,
    *,
    compress: str = DEFAULT_COMPRESS,
    predictor: int = DEFAULT_PREDICTOR,
    nodata: int = 0,
    bigtiff: str = "IF_SAFER",
) -> str:
    """Combina ``tiles`` (GeoTIFFs locais) em um COG ``output``.

    Le cada tile e escreve na janela correspondente da saida (streaming por
    tile), evitando carregar o mosaico inteiro em memoria.
    """
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    tiles = [t for t in tiles if t]
    if not tiles:
        raise ValueError("Nenhum tile para montar o COG.")

    with rasterio.open(tiles[0]) as src0:
        scale = src0.transform.a
        count = src0.count
        dtype = src0.dtypes[0]
        crs = src0.crs

    bounds = []
    for path in tiles:
        with rasterio.open(path) as src:
            b = src.bounds
            bounds.append((b.left, b.bottom, b.right, b.top))

    xmin, ymax, width, height = output_geometry(bounds, scale)
    transform = from_origin(xmin, ymax, scale, scale)

    tmp = f"{output}.tmp.tif"
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress,
        "predictor": predictor,
        "BIGTIFF": "YES",
        "nodata": nodata,
    }

    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            for path in tiles:
                with rasterio.open(path) as src:
                    row, col = dst.index(src.bounds.left, src.bounds.top)
                    dst.write(src.read(), window=Window(col, row, src.width, src.height))
        rio_copy(
            tmp,
            output,
            driver="COG",
            COMPRESS=compress,
            PREDICTOR=predictor,
            BIGTIFF=bigtiff,
        )
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return output
