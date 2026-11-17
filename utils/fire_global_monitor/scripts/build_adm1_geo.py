"""Gera data/adm1_geo.json: contornos simplificados de estados/províncias
(geoBoundaries v6 ADM1, via GEE) casados com os códigos GADM (gid_1) do GWIS.

O casamento é por nome dentro do mesmo país: exato após normalizar
(acentos, caixa, palavras genéricas), depois aproximado (difflib >= 0.75).
Unidades sem par ficam sem gid (aparecem cinza no mapa).

Uso: python build_adm1_geo.py   (precisa de data/adm1/*.csv já gerados)
"""
import csv
import difflib
import json
import pathlib
import re
import unicodedata
from collections import defaultdict

import ee

PROJECT = 'mapbiomas-fire-485203'
DATA = pathlib.Path(__file__).resolve().parent.parent / 'app' / 'data'
TOL_M = 6000      # tolerância de simplificação
BATCH = 250
GENERIC = r'\b(province|provincia|state|estado|region|regione|departamento|department|departement|oblast|district|' \
          r'prefecture|governorate|county|municipality|territory|capital|autonomous|republic|of|de|do|da|del|la|le|the)\b'


def norm(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    s = re.sub(GENERIC, ' ', s)
    return re.sub(r'[^a-z0-9]', '', s)


def rnd(polys):
    out = []
    for poly in polys:
        rings = [[[round(x, 2), round(y, 2)] for x, y in ring] for ring in poly if len(ring) >= 4]
        if rings:
            out.append(rings)
    return out


def main():
    # unidades GADM do GWIS
    gadm = defaultdict(dict)  # iso3 -> norm(name) -> gid
    for f in sorted((DATA / 'adm1').glob('*.csv')):
        for r in csv.DictReader(open(f, encoding='utf-8')):
            gadm[r['iso3']][norm(r['name'])] = r['gid']

    ee.Initialize(project=PROJECT)
    fc = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM1') \
        .map(lambda f: ee.Feature(f.geometry().simplify(TOL_M), {'iso': f.get('shapeGroup'), 'name': f.get('shapeName')}))
    n = fc.size().getInfo()
    lst = fc.toList(n)
    feats = []
    for i in range(0, n, BATCH):
        feats += ee.FeatureCollection(lst.slice(i, min(i + BATCH, n))).getInfo()['features']
        print(f'{len(feats)}/{n}', flush=True)

    out, hit, fuzzy = [], 0, 0
    used = defaultdict(set)
    for f in feats:
        p, g = f['properties'], f['geometry']
        if not g or g['type'] not in ('Polygon', 'MultiPolygon'):
            if g and g['type'] == 'GeometryCollection':
                polys = [x['coordinates'] for x in g['geometries'] if x['type'] == 'Polygon'] + \
                        [c for x in g['geometries'] if x['type'] == 'MultiPolygon' for c in x['coordinates']]
            else:
                continue
        else:
            polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
        cand = gadm.get(p['iso'], {})
        k = norm(p['name'])
        gid = cand.get(k)
        if gid:
            hit += 1
        elif cand:
            m = difflib.get_close_matches(k, [c for c in cand if cand[c] not in used[p['iso']]], n=1, cutoff=0.75)
            if m:
                gid = cand[m[0]]
                fuzzy += 1
        if gid:
            used[p['iso']].add(gid)
        polys = rnd(polys)
        if not polys:
            continue
        out.append({'gid': gid or '', 'iso': p['iso'], 'name': p['name'], 'p': polys})
    (DATA / 'adm1_geo.json').write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    total_gadm = sum(len(v) for v in gadm.values())
    print(f'feições {len(out)} · casadas exato {hit} + aproximado {fuzzy} · GADM com dados {total_gadm}')


if __name__ == '__main__':
    main()
