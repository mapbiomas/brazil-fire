"""Lê as tabelas de frequência exportadas por gee_run_frequency.py e grava
../data/freq_adm0.csv (uma linha por país e período 2001–Y):

    gid, iso3, name, year_start, year_end, area_ha, burned_once_ha, f1 … f25

f<k> = área (ha) que queimou exatamente k anos no período. As tabelas podem vir
em faixas de longitude (várias linhas por país): aqui elas são somadas.

Uso: python fetch_frequency_to_csv.py --account workspace3
"""
import argparse
import csv
import pathlib
from collections import defaultdict

import ee

from ee_profiles import init

PROJECT = 'mapbiomas-brazil'
OUT = pathlib.Path(__file__).resolve().parent.parent / 'data'
MAXF = 25
CLASSES = [(1, 1), (2, 2), (3, 5), (6, 10), (11, MAXF)]  # c1..c5 no arquivo de estados (mesmas classes do painel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0, choices=[0, 1])
    ap.add_argument('--account')
    ap.add_argument('--project', default=PROJECT)
    a = ap.parse_args()
    init(a.account, a.project)
    folder = f'projects/{a.project}/assets/GLOBAL_ANALISES'
    prefix = f'{folder}/mcd64a1_freq_adm{a.level}_'
    assets = sorted(x['id'] for x in ee.data.listAssets({'parent': folder}).get('assets', []) if x['id'].startswith(prefix))

    acc = defaultdict(lambda: defaultdict(float))
    meta = {}
    for aid in assets:
        feats = ee.FeatureCollection(aid).getInfo()['features']
        for f in feats:
            p = f['properties']
            k = (p['gid'], int(p['year_end']))
            meta[p['gid']] = (p['iso3'], p['name'])
            for c, v in p.items():
                if c in ('area_ha', 'burned_once_ha') or (c.startswith('f') and c[1:].isdigit()):
                    acc[k][c] += float(v or 0)
        print(aid.rsplit('/', 1)[-1], len(feats), 'linhas')

    cols = ['gid', 'iso3', 'name', 'year_start', 'year_end', 'area_ha', 'burned_once_ha'] + (
        [f'f{k}' for k in range(1, MAXF + 1)] if a.level == 0 else [f'c{i}' for i in range(1, len(CLASSES) + 1)])
    OUT.mkdir(exist_ok=True)
    path = OUT / f'freq_adm{a.level}.csv'
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (gid, end) in sorted(acc, key=lambda k: (k[1], -acc[k]['burned_once_ha'])):
            d = acc[(gid, end)]
            if not d['burned_once_ha']:
                continue
            n = end - 2001 + 1
            if a.level == 0:
                tail = [round(d[f'f{k}']) if k <= n else '' for k in range(1, MAXF + 1)]
            else:  # estados: só as classes, para o arquivo ficar leve
                tail = [round(sum(d[f'f{k}'] for k in range(lo, min(hi, n) + 1))) for lo, hi in CLASSES]
            w.writerow([gid, *meta[gid], 2001, end, round(d['area_ha']), round(d['burned_once_ha'])] + tail)
    print('gravado', path, 'períodos:', sorted({k[1] for k in acc}))


if __name__ == '__main__':
    main()
