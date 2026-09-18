"""Compara runners para uma unidade (M6).

Uso:

    python -m mapbiomas_cog.benchmarks.benchmark_runner \\
        --country ecuador --theme lulc --collection collection_03 \\
        --product coverage --unit 1985 [--sample 4] [--plan-only]

Requer autenticacao no EE. ``--plan-only`` nao faz chamadas de pixel: apenas
planeja o grid e estima o volume.
"""

from __future__ import annotations

import argparse
import time

from ..domain import config as cog_config
from ..domain.models import TaskSpec, Unit
from ..engine import planner
from ..engine.images import resolve_image


def _bytes_per_pixel(dtype: str) -> int:
    return {"byte": 1, "int16": 2, "float32": 4}.get((dtype or "byte").lower(), 1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark-runner", description=__doc__)
    parser.add_argument("--country", required=True)
    parser.add_argument("--theme", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample", type=int, default=4, help="tiles a medir no hv_local")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    obj = cog_config.load_obj(args.config)
    product = cog_config.make_product(
        obj, args.country, args.theme, args.collection, args.product
    )
    task = TaskSpec(product=product, unit=Unit(args.unit, product.kind))

    grid = planner.plan(task)
    raw_bytes = grid.n_tiles * grid.tile_size_px ** 2 * _bytes_per_pixel(product.dtype)
    print(f"grid: {grid.n_tiles} tiles | {grid.tile_size_px}px | scale={grid.scale} | crs={grid.crs}")
    print(f"volume bruto estimado: {raw_bytes / 1e6:.1f} MB (sem compressao)")
    if args.plan_only:
        return 0

    image = resolve_image(product.assetid, task.unit_key, product.kind)
    from ..engine.runners.hv_local import HvLocalRunner

    runner = HvLocalRunner(max_workers=args.workers)
    sample = grid.tiles[: max(1, args.sample)]

    start = time.time()
    total = 0
    for tile in sample:
        data = runner._compute_tile(image, tile, grid)
        total += len(data)
    elapsed = time.time() - start

    mb = total / 1e6
    print(f"hv_local: {len(sample)} tiles em {elapsed:.1f}s | {mb:.2f} MB | {mb / max(elapsed, 1e-6):.2f} MB/s")
    print(f"estimativa hv_local p/ unidade: {elapsed / len(sample) * grid.n_tiles:.0f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
