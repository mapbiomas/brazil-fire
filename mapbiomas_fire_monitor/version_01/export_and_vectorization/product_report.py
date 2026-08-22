"""Relatorio de metadados dos produtos de fogo do MapBiomas (multipais).

Imprime por produto (pais -> tema -> colecao): type declarado, dtype/bandas/max
observados no GEE, kind, salvamento sugerido do mosaico e flag vectorize.

Uso (onde houver GEE/GCS, ex.: Colab):
    python -m export_and_vectorization.product_report --country brasil --country indonesia
    python -m export_and_vectorization.product_report --country brasil --refresh
    python -m export_and_vectorization.product_report --all
"""

import argparse

from . import config
from .catalog import build_inventory, load_cache


def _fmt(v, width):
    s = str(v) if v is not None else "-"
    return s[:width].ljust(width)


def report(countries, refresh=False):
    inv = build_inventory(countries, refresh=refresh) if refresh else load_cache()
    if not inv or not any(inv.get(c) for c in countries):
        inv = build_inventory(countries, refresh=True)

    for country in countries:
        print("=" * 140)
        print(f"{country.upper()} — fire products")
        print("=" * 140)
        print(f"{'THEME':<8}{'COLLECTION':<14}{'PRODUCT':<32}{'TYPE':<9}{'KIND':<8}"
              f"{'BANDS':<7}{'DTYPE':<9}{'MAX':<8}{'SAVE(ot/nodata/pred)':<26}{'VEC':<5}")
        print("-" * 140)
        for theme, collections in inv.get(country, {}).items():
            for coll, prods in collections.items():
                for p in prods:
                    save = p.get("save", {})
                    bands = len(p.get("bands") or [])
                    n_units = len(p.get("units") or [])
                    save_str = "{}/nodata={}/pred={}".format(
                        save.get("ot"), save.get("nodata"), save.get("predictor"))
                    print(f"{theme:<8}{coll:<14}{_fmt(p['name'], 32)}"
                          f"{_fmt(p.get('declared_type'), 9)}{_fmt(p['kind'], 8)}"
                          f"{_fmt(bands, 7)}{_fmt(p.get('dtype'), 9)}{_fmt(p.get('max'), 8)}"
                          f"{save_str:<26}{str(p.get('vectorize')).ljust(5)}"
                          f" | units={n_units}")
    return inv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", default=None,
                        help="codigo do pais (pode repetir). Default: todos")
    parser.add_argument("--all", action="store_true", help="todos os paises configurados")
    parser.add_argument("--refresh", action="store_true", help="re-escaneia o GEE (ignora cache)")
    args = parser.parse_args()

    if args.all or not args.country:
        countries = list(config.OBJ)
    else:
        countries = [c for c in args.country if c in config.OBJ]

    report(countries, refresh=args.refresh)


if __name__ == "__main__":
    main()
