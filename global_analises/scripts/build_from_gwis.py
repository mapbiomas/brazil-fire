"""Converte os ZIPs do GWIS country profile (raw_gwis/) + os exports do GEE
(data/gee_adm0_monthly.csv, opcional) nos arquivos que o painel lê:

  data/adm0_monthly.csv      série completa por país
  data/adm1/<ANO>.csv        estados/províncias, um arquivo por ano (carregado sob demanda)
  data/meta.json             períodos e fontes

Colunas (vazio = sem dado):
  year, month, gid, iso3, name, parent,
  ba          área queimada MCD64A1 (ha)          — GWIS; 2026+ do GEE
  ba_forest, ba_savanna, ba_grass, ba_crop, ba_other   (ha, só GWIS)
  ba_viirs    área queimada VIIRS VNP64 (ha)      — GWIS
  co2, ch4, pm25, co   emissões GFED (t)           — GWIS
  hotspots    focos FIRMS/MODIS (pixels-dia)       — GEE
  area        área da unidade (ha, só países)      — GEE

Uso: python build_from_gwis.py
"""
import csv
import io
import json
import pathlib
import zipfile
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / 'raw_gwis'
DATA = ROOT / 'data'

BA_COLS = {'forest': 'ba_forest', 'savannas': 'ba_savanna', 'shrublands_grasslands': 'ba_grass',
           'croplands': 'ba_crop', 'other': 'ba_other'}
EMI_COLS = {'CO2': 'co2', 'CH4': 'ch4', 'PM25': 'pm25', 'CO': 'co'}
OUT_COLS = ['year', 'month', 'gid', 'iso3', 'name', 'parent', 'ba', 'ba_forest', 'ba_savanna', 'ba_grass',
            'ba_crop', 'ba_other', 'ba_viirs', 'co2', 'ch4', 'pm25', 'co', 'hotspots', 'area']


def read_zip(prefix):
    z = next(RAW.glob(prefix + '*.zip'))
    with zipfile.ZipFile(z) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            yield from csv.DictReader(io.TextIOWrapper(fh, encoding='utf-8'))


def main():
    # adm1[(y, m, gid1)] e adm0[(y, m, iso3)] -> dict de valores
    adm1 = defaultdict(lambda: defaultdict(float))
    adm0 = defaultdict(lambda: defaultdict(float))
    names0, names1 = {}, {}

    def add(r, field, v):
        if not v:
            return
        y, m, iso = int(r['year']), int(r['month']), r['gid_0']
        names0[iso] = r['country']
        adm0[(y, m, iso)][field] += v
        if r['gid_1']:
            names1[r['gid_1']] = (r['region'], iso)
            adm1[(y, m, r['gid_1'])][field] += v

    for r in read_zip('MCD64A1_'):
        tot = 0.0
        for src, dst in BA_COLS.items():
            v = float(r[src] or 0)
            add(r, dst, v)
            tot += v
        add(r, 'ba', tot)
    print('MCD64A1 ok')
    for r in read_zip('MVP64_'):
        add(r, 'ba_viirs', sum(float(r[c] or 0) for c in BA_COLS))
    print('VIIRS ok')
    for r in read_zip('emission_gfed_'):
        for src, dst in EMI_COLS.items():
            add(r, dst, float(r[src] or 0))
    print('GFED ok')

    gwis_last = max((y, m) for (y, m, _) in adm0)
    marks = defaultdict(set)  # quais campos existem em cada linha (para não gravar 0 onde é "sem dado")
    for k in adm0:
        marks[k].update(['ba', 'ba_forest', 'ba_savanna', 'ba_grass', 'ba_crop', 'ba_other', 'co2', 'ch4', 'pm25', 'co'])

    # GEE: focos, área da unidade e extensão da área queimada após o fim do GWIS
    gee = DATA / 'gee_adm0_monthly.csv'
    areas = {}
    gee_range = None
    if gee.exists():
        for r in csv.DictReader(open(gee, encoding='utf-8')):
            y, m, iso = int(r['year']), int(r['month']), r['iso3']
            k = (y, m, iso)
            names0.setdefault(iso, r['name'])
            if float(r['area_ha'] or 0):
                areas[iso] = float(r['area_ha'])
            adm0[k]['hotspots'] = float(r['hotspots'] or 0)
            marks[k].add('hotspots')
            if (y, m) > gwis_last:
                adm0[k]['ba'] = float(r['burned_ha'] or 0)
                marks[k].add('ba')
            gee_range = (min(gee_range[0], (y, m)), max(gee_range[1], (y, m))) if gee_range else ((y, m), (y, m))
        print('GEE ok', gee_range)

    def fmt(v, is_ha):
        return '' if v is None else (str(round(v)) if is_ha or v >= 100 else f'{v:.2f}'.rstrip('0').rstrip('.'))

    DATA.mkdir(exist_ok=True)
    with open(DATA / 'adm0_monthly.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(OUT_COLS)
        for (y, m, iso) in sorted(adm0):
            d = adm0[(y, m, iso)]
            if not any(d.values()):
                continue
            mk = marks[(y, m, iso)] | ({'ba_viirs'} if 'ba_viirs' in d else set())
            w.writerow([y, m, iso, iso, names0.get(iso, iso), ''] +
                       [fmt(d.get(c, 0.0) if c in mk or c in d else None, c.startswith('ba'))
                        for c in OUT_COLS[6:-1]] + [fmt(areas.get(iso), True)])

    (DATA / 'adm1').mkdir(exist_ok=True)
    years = defaultdict(list)
    for k in sorted(adm1):
        if any(adm1[k].values()):
            years[k[0]].append(k)
    for y, keys in years.items():
        with open(DATA / 'adm1' / f'{y}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(OUT_COLS)
            for (_, m, gid) in keys:
                d = adm1[(y, m, gid)]
                nm, iso = names1[gid]
                w.writerow([y, m, gid, iso, nm, names0.get(iso, iso)] +
                           [fmt(d.get(c, 0.0), c.startswith('ba')) for c in OUT_COLS[6:-2]] + ['', ''])

    meta = {
        'gwis': {'from': '2002-01', 'to': f'{gwis_last[0]}-{gwis_last[1]:02d}', 'viirs_from': '2013-01'},
        'gee': None if not gee_range else {'from': '%d-%02d' % gee_range[0], 'to': '%d-%02d' % gee_range[1]},
        'adm1_years': sorted(years),
    }
    (DATA / 'meta.json').write_text(json.dumps(meta, indent=1))
    print('ok', meta)


if __name__ == '__main__':
    main()
