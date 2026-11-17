"""Frequência de fogo e área queimada acumulada (sem repetição) com MCD64A1,
no estilo da frequência do MapBiomas Fogo: para cada período 2001–Y
(Y = 2001…2025), conta em quantos anos cada pixel queimou e soma, por
país (ou estado), a área em cada classe de frequência.

    área acumulada 2001–Y = área com frequência >= 1 (queimou ao menos uma vez)

Uma task por período, para assets do GEE. Depois:
    python fetch_frequency_to_csv.py   -> data/freq_adm0.csv

Uso:
    python gee_run_frequency.py --account workspace3 --start 2001 --end 2025
"""
import argparse

import ee

from ee_profiles import init
from gee_run_exports import BOUNDS, MODIS_SCALE, TASK_TAG, units_and_lookup

PROJECT = 'mapbiomas-brazil'  # padrão; alternativa: mapbiomas-fire-485203
FIRST = 2001
STRIPS = 1  # faixas de longitude; >1 perdeu a faixa 135°E–180° e bordas (2026-11-17): não usar sem revisar


def annual_burned(y):
    """1 onde o pixel queimou em algum mês do ano y, 0 caso contrário."""
    col = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(f'{y}-01-01', f'{y + 1}-01-01').select('BurnDate')
    return col.max().gt(0).unmask(0).rename('burned').toByte()  # mesmo nome em todos os anos (coleção homogênea)


def period_table(end, level, uid_img, lookup, world):
    n = end - FIRST + 1
    freq = ee.ImageCollection([annual_burned(y) for y in range(FIRST, end + 1)]).sum().rename('freq')
    proj = ee.ImageCollection('MODIS/061/MCD64A1').first().select('BurnDate').projection()
    px = ee.Image.pixelArea().divide(1e4)
    bands = [px.multiply(freq.eq(k)).rename(f'f{k}') for k in range(1, n + 1)] + [px.rename('area')]
    stack = ee.Image.cat(bands).addBands(uid_img)
    # o mundo inteiro numa redução só estoura o tempo nos períodos longos: reduz em faixas de
    # longitude e grava uma linha por (unidade, faixa); o fetch soma as faixas
    strips = [world] if world is not None and STRIPS == 1 else [
        ee.Geometry.Rectangle([lo, -60, lo + 360 / STRIPS, 85], None, False) for lo in range(-180, 180, 360 // STRIPS)]
    if world is not None and STRIPS > 1:
        strips = [s.intersection(world, 1) for s in strips]
    groups = ee.List([])
    for g in strips:
        groups = groups.cat(ee.List(stack.reduceRegion(
            reducer=ee.Reducer.sum().repeat(n + 1).group(groupField=n + 1, groupName='uid'),
            geometry=g, crs=proj, scale=MODIS_SCALE, maxPixels=1e13, tileScale=8).get('groups')))

    def row(d):
        d = ee.Dictionary(d)
        sums = ee.List(d.get('sum'))
        mt = ee.Feature(lookup.filter(ee.Filter.eq('uid', d.get('uid'))).first())
        props = ee.Dictionary({
            'level': level, 'gid': mt.get('gid'), 'iso3': mt.get('iso3'), 'name': mt.get('name'),
            'parent': mt.get('parent'), 'year_start': FIRST, 'year_end': end,
            'area_ha': sums.get(n),
            'burned_once_ha': ee.Array(sums.slice(0, n)).reduce(ee.Reducer.sum(), [0]).get([0]),
        }).combine(ee.Dictionary.fromLists([f'f{k}' for k in range(1, n + 1)], sums.slice(0, n)))
        # geometria fictícia: export para asset não aceita feature sem geometria
        return ee.Feature(ee.Geometry.Point([0, 0]), props)

    return ee.FeatureCollection(groups.map(row))


def main():
    global STRIPS
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0, choices=[0, 1])
    ap.add_argument('--start', type=int, default=FIRST, help='primeiro ano final de período')
    ap.add_argument('--end', type=int, default=2025, help='último ano final de período')
    ap.add_argument('--account')
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--strips', type=int, default=STRIPS, help='faixas de longitude (mais faixas = menos timeout)')
    a = ap.parse_args()
    STRIPS = a.strips

    init(a.account, a.project)
    folder = f'projects/{a.project}/assets/GLOBAL_ANALISES'
    try:
        ee.data.createAsset({'type': 'FOLDER'}, folder)
    except ee.EEException:
        pass
    existing = {x['id'] for x in ee.data.listAssets({'parent': folder}).get('assets', [])}
    world = ee.Geometry.Rectangle([-180, -60, 180, 85], None, False)
    uid_img, lookup = units_and_lookup(a.level)
    for end in range(a.end, a.start - 1, -1):  # do mais longo para o mais curto
        name = f'mcd64a1_freq_adm{a.level}_{FIRST}_{end}'
        asset = f'{folder}/{name}'
        if asset in existing:
            print('já existe', name)
            continue
        task = ee.batch.Export.table.toAsset(
            collection=period_table(end, a.level, uid_img, lookup, world),
            description=f'WS_IPAM_{TASK_TAG}-{name}', assetId=asset)
        task.start()
        print('task', f'WS_IPAM_{TASK_TAG}-{name}', task.id)


if __name__ == '__main__':
    main()
