"""Lê as tabelas mensais exportadas por gee_run_exports.py e grava
../data/gee_adm{N}_monthly.csv (entrada de build_from_gwis.py):
    level, gid, iso3, name, parent, year, month, burned_ha, area_ha, hotspots

Uso: python fetch_assets_to_csv.py --level 0
"""
import argparse
import csv
import pathlib

import ee

from ee_profiles import init

PROJECT = 'mapbiomas-fire-485203'
FOLDER = None  # projects/<projeto>/assets/GLOBAL_ANALISES
COLS = ['level', 'gid', 'iso3', 'name', 'parent', 'year', 'month', 'burned_ha', 'area_ha', 'hotspots']
OUT = pathlib.Path(__file__).resolve().parent.parent / 'data'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0, choices=[0, 1])
    ap.add_argument('--account', help='perfil de ee_profiles.py (padrão: login padrão)')
    ap.add_argument('--project', default=PROJECT, help='projeto Cloud registrado no Earth Engine')
    a = ap.parse_args()
    global FOLDER
    init(a.account, a.project)
    FOLDER = f'projects/{a.project}/assets/GLOBAL_ANALISES'

    prefix = f'{FOLDER}/mcd64a1_firms_adm{a.level}_'
    assets = sorted(x['id'] for x in ee.data.listAssets({'parent': FOLDER}).get('assets', [])
                    if x['id'].startswith(prefix))
    rows = []
    for aid in assets:
        feats = ee.FeatureCollection(aid).getInfo()['features']
        for f in feats:
            p = f['properties']
            rows.append({c: p.get(c, '') for c in COLS})
        print(aid.rsplit('/', 1)[-1], len(feats))

    rows.sort(key=lambda r: (int(r['year']), int(r['month']), -float(r['burned_ha'] or 0)))
    for r in rows:
        r['burned_ha'] = round(float(r['burned_ha'] or 0), 1)
        r['area_ha'] = round(float(r['area_ha'] or 0), 1)
        r['hotspots'] = int(round(float(r['hotspots'] or 0)))
    OUT.mkdir(exist_ok=True)
    path = OUT / f'gee_adm{a.level}_monthly.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print('gravado', path, len(rows), 'linhas de', len(assets), 'meses')


if __name__ == '__main__':
    main()
