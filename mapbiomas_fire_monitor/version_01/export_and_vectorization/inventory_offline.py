"""Inventario offline dos grupos de dados configurados no OBJ (sem GEE).

Imprime a arvore pais -> tema -> colecao -> produtos (type/vectorize/visible)
apenas a partir do config.OBJ. Roda em qualquer lugar.

Uso:
    python -m export_and_vectorization.inventory_offline [--country brasil] [--all]
"""

import argparse

from . import config


def inventory(countries=None):
    countries = countries or list(config.OBJ)
    for country in countries:
        print("=" * 100)
        print(f"{country.upper()}  (flag {config.flag(country)})")
        print("=" * 100)
        themes = config.OBJ.get(country, {})
        if not themes:
            print("  (sem grupos configurados)")
            continue
        for theme, collections in themes.items():
            print(f"  [tema] {theme}")
            if not collections:
                print("    (vazio)")
                continue
            for coll, prods in collections.items():
                print(f"    [colecao] {coll}")
                if not prods:
                    print("      (vazio)")
                    continue
                for p in prods:
                    vis = "" if p.get("visible", True) else "  [OCULTO]"
                    vec = " vec" if p.get("vectorize") else ""
                    print(f"      - {p['product']:<32} type={p.get('type','byte'):<8}"
                          f" vectorize={p.get('vectorize', False)}{vec}{vis}")
                    print(f"          assetid: {p['assetid']}")
    return countries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", action="append", default=None, help="codigo do pais (pode repetir)")
    parser.add_argument("--all", action="store_true", help="todos os paises")
    args = parser.parse_args()
    countries = None
    if not args.all:
        countries = args.country or list(config.OBJ)
    inventory(countries)


if __name__ == "__main__":
    main()
