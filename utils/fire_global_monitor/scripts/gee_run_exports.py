"""Dispara exports mensais (um mês = uma task) de área queimada MCD64A1 e
focos de calor FIRMS por país (ADM0) ou estado/província (ADM1), para assets
do GEE. Depois rode fetch_assets_to_csv.py para montar data/adm{N}_monthly.csv.

Uso:
    python gee_run_exports.py --level 0 --start 2024-01 --end 2026-09
    python gee_run_exports.py --level 1 --start 2026-09 --end 2026-09

Mesma lógica de gee_export_mcd64a1_monthly.js (Code Editor), mais os focos.
"""
import argparse

import ee

from ee_profiles import init

PROJECT = 'mapbiomas-brazil'  # padrão; alternativa: mapbiomas-fire-485203
FOLDER = None  # projects/<projeto>/assets/GLOBAL_ANALISES
BOUNDS = {0: 'WM/geoLab/geoBoundaries/600/ADM0', 1: 'WM/geoLab/geoBoundaries/600/ADM1'}
MODIS_SCALE = 463.312716528
TASK_TAG = 'GLOBAL_ANALISES'  # nome das tasks: WS_IPAM_{TASK_TAG}-{arquivo}
WORLD = None


def months(start, end):
    y, m = map(int, start.split('-'))
    ye, me = map(int, end.split('-'))
    out = []
    while (y, m) <= (ye, me):
        out.append(f'{y:04d}-{m:02d}')
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def units_and_lookup(level):
    units = ee.FeatureCollection(BOUNDS[level])
    ids = units.aggregate_array('system:index')
    units = units.map(lambda f: f.set('uid', ids.indexOf(f.get('system:index')).add(1)))
    uid_img = units.reduceToImage(['uid'], ee.Reducer.first()).rename('uid').toInt()
    adm0 = ee.FeatureCollection(BOUNDS[0])

    def meta(f):
        iso3 = f.get('shapeGroup')
        parent = '' if level == 0 else ee.Feature(
            adm0.filter(ee.Filter.eq('shapeGroup', iso3)).first()).get('shapeName')
        return ee.Feature(None, {
            'uid': f.get('uid'),
            'gid': iso3 if level == 0 else f.get('shapeID'),
            'iso3': iso3, 'name': f.get('shapeName'), 'parent': parent,
        })

    return uid_img, units.map(meta)


def grouped(img, uid_img, nbands, crs, scale):
    r = img.addBands(uid_img).reduceRegion(
        reducer=ee.Reducer.sum().repeat(nbands).group(groupField=nbands, groupName='uid'),
        geometry=WORLD, crs=crs, scale=scale, maxPixels=1e13, tileScale=4)
    return ee.List(r.get('groups'))


def month_table(ym, level, uid_img, lookup):
    start = ee.Date(ym + '-01')
    end = start.advance(1, 'month')
    y, m = int(ym[:4]), int(ym[5:])

    # área queimada (MCD64A1) + área da unidade, na grade MODIS
    ba = ee.ImageCollection('MODIS/061/MCD64A1').filterDate(start, end).first()
    px = ee.Image.pixelArea().divide(1e4)
    burned = px.multiply(ba.select('BurnDate').gt(0).unmask(0))
    g_ba = grouped(burned.addBands(px), uid_img, 2, ba.select('BurnDate').projection(), MODIS_SCALE)

    # focos (FIRMS, MODIS 1 km): soma de pixels-dia com fogo ativo no mês
    firms = ee.ImageCollection('FIRMS').filterDate(start, end).select('T21')
    hot = firms.map(lambda i: i.gt(0).unmask(0).toInt()).sum().rename('hotspots')
    g_hs = grouped(hot, uid_img, 1, 'EPSG:4326', 1000)
    hs = ee.Dictionary.fromLists(
        g_hs.map(lambda d: ee.Number(ee.Dictionary(d).get('uid')).format('%d')),
        g_hs.map(lambda d: ee.List(ee.Dictionary(d).get('sum')).get(0)))

    def row(d):
        d = ee.Dictionary(d)
        uid = d.get('uid')
        sums = ee.List(d.get('sum'))
        mt = ee.Feature(lookup.filter(ee.Filter.eq('uid', uid)).first())
        # geometria fictícia: export para asset não aceita feature sem geometria
        return ee.Feature(ee.Geometry.Point([0, 0]), {
            'level': level, 'gid': mt.get('gid'), 'iso3': mt.get('iso3'),
            'name': mt.get('name'), 'parent': mt.get('parent'),
            'year': y, 'month': m,
            'burned_ha': sums.get(0), 'area_ha': sums.get(1),
            'hotspots': hs.get(ee.Number(uid).format('%d'), 0),
        })

    return ee.FeatureCollection(g_ba.map(row))


def main():
    global WORLD
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=0, choices=[0, 1])
    ap.add_argument('--account', help='perfil de ee_profiles.py (padrão: login padrão)')
    ap.add_argument('--project', default=PROJECT, help='projeto Cloud registrado no Earth Engine')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--newest-first', action='store_true', default=True)
    a = ap.parse_args()

    global FOLDER
    init(a.account, a.project)
    FOLDER = f'projects/{a.project}/assets/GLOBAL_ANALISES'
    WORLD = ee.Geometry.Rectangle([-180, -60, 180, 85], None, False)
    try:
        ee.data.createAsset({'type': 'FOLDER'}, FOLDER)
    except ee.EEException:
        pass  # já existe

    existing = {x['id'] for x in ee.data.listAssets({'parent': FOLDER}).get('assets', [])}
    uid_img, lookup = units_and_lookup(a.level)
    todo = months(a.start, a.end)
    if a.newest_first:
        todo = todo[::-1]
    for ym in todo:
        name = f'mcd64a1_firms_adm{a.level}_{ym.replace("-", "_")}'
        asset = f'{FOLDER}/{name}'
        if asset in existing:
            print('já existe', name)
            continue
        task = ee.batch.Export.table.toAsset(
            collection=month_table(ym, a.level, uid_img, lookup),
            description=f'WS_IPAM_{TASK_TAG}-{name}', assetId=asset)
        task.start()
        print('task', f'WS_IPAM_{TASK_TAG}-{name}', task.id)


if __name__ == '__main__':
    main()
