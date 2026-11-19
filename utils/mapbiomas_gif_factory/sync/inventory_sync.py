"""Inventário dos assets públicos do MapBiomas no Earth Engine.

Percorre `projects/mapbiomas-public/assets` (todas as iniciativas, todos os temas)
e grava `inventory/gee_public.json`: id e tipo de cada asset, em ordem. É a lista
de onde o gerador do catálogo escolhe, por exemplo, a coleção e a versão mais
recentes de uso e cobertura de cada iniciativa. Precisa do Earth Engine.

Uso:  python sync/inventory_sync.py --project mapbiomas-brazil
"""
import argparse
import concurrent.futures
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "inventory" / "gee_public.json"
TREE = "projects/mapbiomas-public/assets"


def children(ee, parent):
    assets, token = [], None
    while True:
        params = {"parent": parent, "pageSize": 1000}
        if token:
            params["pageToken"] = token
        page = ee.data.listAssets(params)
        assets += page.get("assets", [])
        token = page.get("nextPageToken")
        if not token:
            return assets


def walk(ee, parent, depth=0):
    found = []
    for asset in children(ee, parent):
        if asset["type"] == "FOLDER":
            if depth < 6:
                found += walk(ee, asset["id"], depth + 1)
        else:
            found.append({"id": asset["id"], "type": asset["type"]})
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="mapbiomas-brazil")
    args = parser.parse_args()
    import ee
    ee.Initialize(project=args.project)
    roots = [a["id"] for a in children(ee, TREE) if a["type"] == "FOLDER"]
    assets = []
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for result in pool.map(lambda root: walk(ee, root), roots):
            assets += result
    assets.sort(key=lambda a: a["id"])
    data = {"root": TREE, "assets": assets}
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
    if previous and previous.get("assets") == assets:
        print(f"Sem mudança: {len(assets)} assets.")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**data, "fetched": date.today().isoformat()}, indent=1) + "\n", encoding="utf-8", newline="\n")
    before = {a["id"] for a in (previous or {}).get("assets", [])}
    after = {a["id"] for a in assets}
    print(f"{len(assets)} assets; novos: {len(after - before)}; removidos: {len(before - after)}")
    for asset_id in sorted(after - before)[:50]:
        print(f"  + {asset_id}")
    for asset_id in sorted(before - after)[:50]:
        print(f"  - {asset_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
