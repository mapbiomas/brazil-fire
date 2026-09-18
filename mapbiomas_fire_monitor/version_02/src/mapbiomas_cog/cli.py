"""CLI do mapbiomas_cog (M5).

    mapbiomas-cog --version
    mapbiomas-cog status [--country C --theme T --collection K --product P]
    mapbiomas-cog export  --country C --theme T --collection K --product P [--unit U] [--force]
    mapbiomas-cog sync    ...
    mapbiomas-cog assemble ...
    mapbiomas-cog retry   ...
    mapbiomas-cog publish ...
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="caminho do config.py (OBJ)")
    parser.add_argument("--bucket", default=None, help="bucket de processamento")
    parser.add_argument("--state-dir", default=None, help="diretorio do manifest local")
    parser.add_argument("--runner", default="ee_batch_tiled", help="runner do engine")


def _add_product_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--country", required=required)
    parser.add_argument("--theme", required=required)
    parser.add_argument("--collection", required=required)
    parser.add_argument("--product", required=required)
    parser.add_argument("--unit", action="append", default=None, help="unidade (pode repetir)")
    parser.add_argument("--force", action="store_true")


def _make_product(args):
    from .domain import config as cfg

    obj = cfg.load_obj(args.config)
    return cfg.make_product(obj, args.country, args.theme, args.collection, args.product)


def _make_task(args, product, unit_key):
    from .domain.models import TaskSpec, Unit

    return TaskSpec(
        product=product,
        unit=Unit(key=unit_key, kind=product.kind),
        runner=args.runner,
        force=getattr(args, "force", False),
    )


def _engine(args):
    from .engine import Engine

    return Engine(runner=args.runner, bucket=args.bucket, state_dir=args.state_dir)


def _units_from_ee(product, explicit):
    if explicit:
        return list(explicit)
    from .engine.images import list_units

    return list_units(product.assetid, product.kind)


def _units_from_manifest(product, explicit, state_dir):
    if explicit:
        return list(explicit)
    from .domain import paths
    from .state.manifest import Manifest

    data = Manifest(paths.local_state_path(product.root, state_dir)).data
    return sorted(data.get("units", {}))


# -- comandos ----------------------------------------------------------------
def _cmd_status(args) -> int:
    from .catalog import build_inventory, summary

    if args.country and args.product:
        product = _make_product(args)
        counts = summary(product.root, state_dir=args.state_dir)
        print(f"{product.root}: {counts or 'sem manifest'}")
        return 0

    inv = build_inventory(config_path=args.config)
    n_colls = sum(len(colls) for themes in inv.values() for colls in themes.values())
    print(f"catalogo: {len(inv)} pais(es), {n_colls} colecao(oes) com produtos")
    for country in inv:
        print(f" - {country}")
    return 0


def _cmd_export(args) -> int:
    product = _make_product(args)
    engine = _engine(args)
    for unit in _units_from_ee(product, args.unit):
        result = engine.run(_make_task(args, product, unit))
        print(f"[EXPORT] {unit}: submitted={result.submitted} ({result.detail})")
    return 0


def _cmd_sync(args) -> int:
    product = _make_product(args)
    engine = _engine(args)
    for unit in _units_from_manifest(product, args.unit, args.state_dir):
        result = engine.sync(_make_task(args, product, unit))
        print(f"[SYNC] {unit}: {result.detail}")
    return 0


def _cmd_retry(args) -> int:
    product = _make_product(args)
    engine = _engine(args)
    for unit in _units_from_manifest(product, args.unit, args.state_dir):
        result = engine.retry(_make_task(args, product, unit))
        print(f"[RETRY] {unit}: submitted={result.submitted} ({result.detail})")
    return 0


def _cmd_assemble(args) -> int:
    from .assemble import assemble_unit

    product = _make_product(args)
    for unit in _units_from_manifest(product, args.unit, args.state_dir):
        cog = assemble_unit(
            _make_task(args, product, unit),
            force=args.force,
            state_dir=args.state_dir,
            log=print,
        )
        print(f"[ASSEMBLE] {unit}: {cog}")
    return 0


def _cmd_publish(args) -> int:
    from .publish import publish_cog

    product = _make_product(args)
    for unit in _units_from_manifest(product, args.unit, args.state_dir):
        status = publish_cog(
            product.root,
            unit,
            bucket=args.bucket,
            force=args.force,
            log=print,
        )
        print(f"[PUBLISH] {unit}: {status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mapbiomas-cog", description=__doc__)
    parser.add_argument("--version", action="version", version=f"mapbiomas-cog {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_status = sub.add_parser("status", help="catalogo e status do manifest")
    _add_common(p_status)
    _add_product_args(p_status, required=False)
    p_status.set_defaults(func=_cmd_status)

    for name, func, help_text in (
        ("export", _cmd_export, "submete as tasks dos tiles pendentes"),
        ("sync", _cmd_sync, "atualiza o estado das tasks no EE"),
        ("retry", _cmd_retry, "reenvia apenas os tiles falhos"),
        ("assemble", _cmd_assemble, "monta o COG da unidade"),
        ("publish", _cmd_publish, "espelha o COG no bucket publico"),
    ):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        _add_product_args(p)
        p.set_defaults(func=func)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
