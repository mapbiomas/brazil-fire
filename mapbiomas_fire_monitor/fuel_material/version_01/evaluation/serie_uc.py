"""
Série de combustível na escala da Unidade de Conservação (caso: PARNA das Emas)
================================================================================

Série mensal de NPV/GV/Solo na UC inteira, separada por cobertura (MapBiomas col. 11) e por
tempo desde o último fogo em MESES (cicatriz mensal do MapBiomas Fogo col. 5.1:
`burned_monthly_YYYY`, valor = mês). Junto, a área queimada por mês (ICMBio por tipo e MapBiomas).

Fontes de combustível:
  --produto   asset Material Combustível (HLS, monthly_minnbr, 2019+)
  --landsat   o mesmo algoritmo (máscara QA, refletância > 0, mosaico minNBR mensal, SMA com os
              endmembers de `scripts/ems_globais.py`) aplicado ao Landsat C2 L2 no GEE, em memória.
              Serve para estender a série antes de 2019 e para comparar com o produto.
  --comparar  produto × Landsat, pixel a pixel, em alguns meses de 2019–2025

Nada é gravado no GEE. Saídas em tabelas/serie_uc_<uc>_*.csv.

Uso:
  python serie_uc.py --produto
  python serie_uc.py --comparar
  python serie_uc.py --landsat --anos 1985 2025
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuel_prescribed_burns_stats as fb  # noqa: E402  (inicializa o EE)
import ee  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tabelas')
FIRE = 'projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_'
MONTHLY = ee.Image(FIRE + 'monthly_burned_v1')          # burned_monthly_1985 … _2025, valor = mês
LULC = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3')
UCS = {'emas': ('PARNA das Emas', ee.FeatureCollection('WCMC/WDPA/current/polygons')
                .filter(ee.Filter.eq('NAME', 'Parque Nacional Das Emas')).geometry())}
# cobertura agrupada (códigos MapBiomas col. 11)
COB = {3: 1, 4: 2, 12: 3, 11: 4}          # 1 floresta · 2 savana · 3 campo · 4 campo alagado · 9 outros
COB_NOME = {1: 'floresta', 2: 'savana', 3: 'campo', 4: 'campo_alagado', 9: 'outros'}
# tempo desde o último fogo, em meses → classe
TSF_CLASSES = [(0, 12, 1, '<1 ano'), (12, 36, 2, '1–3 anos'), (36, 108, 3, '3–9 anos'), (108, 10000, 4, '≥9 anos ou sem registro')]
SEM = 9999
EM = {  # scripts/ems_globais.py (refletância ×10000; ordem blue, green, red, nir, swir1, swir2)
    'gv': [119, 475, 169, 6250, 2399, 675],
    'npv': [1514, 1597, 1421, 3053, 7707, 1975],
    'soil': [1799, 2479, 3158, 5437, 7707, 6646],
    'cloud': [4031, 8714, 7900, 8989, 7002, 6607],
}


# ----------------------------------------------------------------------------- histórico mensal
def months_since_fire(year, month):
    """Meses desde a última cicatriz MapBiomas ANTES do mês (year, month); SEM se não houver desde 1985."""
    t = (year - 1985) * 12 + (month - 1)
    bands = []
    for y in range(1985, min(year, 2025) + 1):
        b = MONTHLY.select(f'burned_monthly_{y}')
        idx = b.add((y - 1985) * 12 - 1).toInt().rename('idx')   # índice do mês da cicatriz
        bands.append(idx.updateMask(idx.lt(t)))
    last = ee.ImageCollection(bands).max()
    return ee.Image(t).subtract(last).unmask(SEM).rename('msf')


def cover_class(year):
    lc = LULC.select(f'classification_{min(max(year - 1, 1985), 2025)}')
    out = ee.Image(9)
    for code, k in COB.items():
        out = out.where(lc.eq(code), k)
    return out.rename('cob')


# classes finas, em anos completos desde a última cicatriz (para comparar com experimentos de campo,
# p.ex. Zupo et al. 2022: parcelas com 1, 2 e 4 anos sem fogo)
TSF_FINAS = [(0, 12, 1, '<1'), (12, 24, 2, '1'), (24, 36, 3, '2'), (36, 48, 4, '3'), (48, 108, 5, '4–8'),
             (108, 10000, 6, '≥9 ou sem registro')]


def tsf_class(msf, classes=None):
    out = ee.Image(0)
    for lo, hi, k, _ in (classes or TSF_CLASSES):
        out = out.where(msf.gte(lo).And(msf.lt(hi)), k)
    return out.rename('tsfc')


def grouped_means(img3, key, geom, scale=30):
    """Médias de 3 bandas e contagem, agrupadas por `key` (cob*10 + tsfc)."""
    red = ee.Reducer.mean().repeat(3).combine(ee.Reducer.count(), sharedInputs=False) \
        .group(groupField=4, groupName='k')
    img = img3.addBands(img3.select(0).rename('c')).addBands(key)
    d = img.reduceRegion(red, geom, scale, maxPixels=1e10, tileScale=4)
    return d.get('groups')


# ----------------------------------------------------------------------------- Landsat (algoritmo em memória)
L_BANDS = {'L5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
           'L7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
           'L8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
           'L9': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']}
L_COLS = {'L5': 'LANDSAT/LT05/C02/T1_L2', 'L7': 'LANDSAT/LE07/C02/T1_L2',
          'L8': 'LANDSAT/LC08/C02/T1_L2', 'L9': 'LANDSAT/LC09/C02/T1_L2'}
SIX = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']


def _prep(img, sensor):
    qa = img.select('QA_PIXEL')
    # bits: 1 nuvem dilatada, 2 cirrus, 3 nuvem, 4 sombra de nuvem, 5 neve
    clear = qa.bitwiseAnd(0b111110).eq(0).And(img.select('QA_RADSAT').eq(0))
    sr = img.select(L_BANDS[sensor], SIX).multiply(0.0000275).add(-0.2)
    valid = sr.reduce(ee.Reducer.min()).gt(0)
    sr = sr.updateMask(clear.And(valid))
    nbr = sr.normalizedDifference(['nir', 'swir2'])
    return sr.addBands(nbr.multiply(-1).rename('min_nbr'))


def landsat_month(year, month, geom, sensors=None):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')
    sensors = sensors or list(L_COLS)
    cols = [ee.ImageCollection(L_COLS[s]).filterBounds(geom).filterDate(start, end).map(lambda i, s=s: _prep(i, s))
            for s in sensors]
    col = cols[0]
    for c in cols[1:]:
        col = col.merge(c)
    empty = ee.Image.constant([0] * 7).rename(SIX + ['min_nbr']).toFloat().updateMask(0)
    mos = ee.Image(ee.Algorithms.If(col.size().gt(0), col.qualityMosaic('min_nbr'), empty)) \
        .select(SIX).multiply(10000)
    ems = [EM[k] for k in ('gv', 'npv', 'soil', 'cloud')]
    fr = mos.unmix(ems, True, True).rename(['gv', 'npv', 'soil', 'cloud'])   # totalmente restrita
    fr_u = mos.unmix(ems, False, False).max(0).rename(['gv_u', 'npv_u', 'soil_u', 'cloud_u'])  # sem restrição, max(0)
    return fr.multiply(100).addBands(fr_u.multiply(100)), col.size()


# ----------------------------------------------------------------------------- tarefas
def _months(y0, y1):
    return [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)]


def _write(path, rows, cols):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('→', path, len(rows), 'linhas')


def _series(uc, months, fuel_fn, tag, workers=4, classes=None):
    nome, geom = UCS[uc]

    def one(ym):
        y, m = ym
        img = fuel_fn(y, m)
        if img is None:
            return []
        key = cover_class(y).multiply(10).add(tsf_class(months_since_fire(y, m), classes)).rename('k')
        groups = fb.retry(lambda: grouped_means(img.select(['npv', 'gv', 'soil']), key, geom).getInfo()) or []
        out = []
        for g in groups:
            k = int(g['k'])
            means = g['mean']
            out.append({'uc': nome, 'year': y, 'month': m, 'cob': COB_NOME.get(k // 10, '?'), 'tsfc': k % 10,
                        'npv': means[0], 'gv': means[1], 'soil': means[2], 'n': g['count']})
        return out
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for i, part in enumerate(ex.map(one, months)):
            rows += part
            if i % 12 == 11:
                print(f'  {tag}: {months[i][0]} ok', flush=True)
    _write(os.path.join(TAB, f'serie_uc_{uc}_{tag}.csv'), rows,
           ['uc', 'year', 'month', 'cob', 'tsfc', 'npv', 'gv', 'soil', 'n'])


def produto(uc, y0=2019, y1=2025):
    def fuel(y, m):
        return fb.fuel_image(ee.Date.fromYMD(y, m, 15))
    _series(uc, _months(y0, y1), fuel, 'produto')


def landsat(uc, y0, y1):
    geom = UCS[uc][1]

    def fuel(y, m):
        # versão SEM restrição + max(0): é a que reproduz o produto (ver --comparar)
        img, _ = landsat_month(y, m, geom)
        return img.select(['npv_u', 'gv_u', 'soil_u'], ['npv', 'gv', 'soil'])
    _series(uc, _months(y0, y1), fuel, f'landsat_{y0}_{y1}', workers=3)


def seca_classes_finas(uc):
    """Agosto e setembro, classes finas (anos completos desde a última cicatriz): produto 2019–2025 e
    Landsat 1995–2018. Saídas: serie_uc_<uc>_seca_finas_produto.csv / _landsat.csv."""
    geom = UCS[uc][1]
    meses = lambda y0, y1: [(y, m) for y in range(y0, y1 + 1) for m in (8, 9)]
    _series(uc, meses(2019, 2025), lambda y, m: fb.fuel_image(ee.Date.fromYMD(y, m, 15)),
            'seca_finas_produto', classes=TSF_FINAS)

    def ls(y, m):
        img, _ = landsat_month(y, m, geom)
        return img.select(['npv_u', 'gv_u', 'soil_u'], ['npv', 'gv', 'soil'])
    _series(uc, meses(1995, 2018), ls, 'seca_finas_landsat', workers=3, classes=TSF_FINAS)


def queimadas(uc):
    """Área queimada mensal: ICMBio por tipo (2019–2025) e MapBiomas (1985–2025)."""
    nome, geom = UCS[uc]
    fc = fb.load_icmbio(list(range(2019, 2026))).filter(ee.Filter.eq('nome_uc', nome))
    cols = ['year', 'month', 'fire_type_code', 'area_ha', 'date_source']
    lst = fc.reduceColumns(ee.Reducer.toList(len(cols)), cols).get('list').getInfo()
    agg = {}
    for y, m, t, a, src in lst:
        if y is None or m is None:
            continue
        k = (int(y), int(m), int(t or 0))
        agg[k] = agg.get(k, 0) + (a or 0)
    rows = [{'fonte': 'ICMBio', 'year': y, 'month': m, 'fire_type_code': t,
             'tipo': fb.FIRE_TYPE_NAMES.get(t, ''), 'area_ha': a} for (y, m, t), a in sorted(agg.items())]
    area = ee.Image.pixelArea().divide(1e4)
    for y in range(1985, 2026):
        d = area.addBands(MONTHLY.select(f'burned_monthly_{y}')).reduceRegion(
            ee.Reducer.sum().group(groupField=1, groupName='m'), geom, 30, maxPixels=1e10, tileScale=4).getInfo()
        for g in d.get('groups', []):
            rows.append({'fonte': 'MapBiomas', 'year': y, 'month': int(g['m']), 'fire_type_code': '',
                         'tipo': 'cicatriz', 'area_ha': g['sum']})
    _write(os.path.join(TAB, f'serie_uc_{uc}_queimadas.csv'), rows,
           ['fonte', 'year', 'month', 'fire_type_code', 'tipo', 'area_ha'])
    uc_ha = geom.area(10).divide(1e4).getInfo()
    print(f'área da UC: {uc_ha:.0f} ha')


def comparar(uc, meses=((2019, 8), (2020, 6), (2021, 9), (2022, 5), (2023, 7), (2024, 8), (2025, 6), (2024, 2))):
    """Produto × Landsat em memória (L8/L9, mesmos meses), por pixel numa amostra dentro da UC."""
    nome, geom = UCS[uc]
    rows = []
    for y, m in meses:
        prod = fb.fuel_image(ee.Date.fromYMD(y, m, 15)).rename(['p_npv', 'p_gv', 'p_soil'])
        ls, n = landsat_month(y, m, geom, sensors=['L8', 'L9'])
        img = prod.addBands(ls)
        smp = img.sample(region=geom, scale=30, numPixels=3000, seed=1, geometries=False)
        feats = fb.retry(lambda: smp.getInfo()['features'])
        ncenas = n.getInfo()
        for f in feats:
            p = f['properties']
            p.update(year=y, month=m, n_cenas_landsat=ncenas)
            rows.append(p)
        print(f'  {y}-{m:02d}: {len(feats)} pixels, {ncenas} cenas Landsat', flush=True)
    cols = ['year', 'month', 'n_cenas_landsat', 'p_npv', 'p_gv', 'p_soil', 'npv', 'gv', 'soil', 'cloud',
            'npv_u', 'gv_u', 'soil_u', 'cloud_u']
    _write(os.path.join(TAB, f'serie_uc_{uc}_comparacao_landsat.csv'), rows, cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uc', default='emas', choices=list(UCS))
    ap.add_argument('--produto', action='store_true')
    ap.add_argument('--queimadas', action='store_true')
    ap.add_argument('--comparar', action='store_true')
    ap.add_argument('--landsat', action='store_true')
    ap.add_argument('--anos', type=int, nargs=2, default=[1985, 2025])
    ap.add_argument('--seca-finas', action='store_true', help='ago–set por anos completos desde o último fogo')
    a = ap.parse_args()
    if a.seca_finas:
        seca_classes_finas(a.uc)
    if a.queimadas:
        queimadas(a.uc)
    if a.comparar:
        comparar(a.uc)
    if a.produto:
        produto(a.uc)
    if a.landsat:
        landsat(a.uc, *a.anos)


if __name__ == '__main__':
    main()
