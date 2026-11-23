"""
Histórico de fogo e cobertura nas queimas prescritas ICMBio (queima × entorno)
==============================================================================

Para cada queima prescrita/controlada ICMBio, calcula no polígono (QUEIMA) e no ENTORNO de 1 km
(o mesmo da análise de combustível: buffer menos o polígono, sem cicatrizes ICMBio do mesmo ano):

  - tempo desde o último fogo  ← MapBiomas Fogo col. 5.1 `time_after_fire_Y`
        (Y = ano de t0; a banda Y conta só fogos até Y−1, então a própria queima não entra;
         0 = sem fogo registrado desde 1985 → gravado como 99, "sem registro")
  - intervalo médio de retorno do fogo ← média das bandas `interval_since_fire_1986 … _Y−1`,
        sem os zeros (0 = primeiro fogo registrado, não é intervalo); pixel só tem valor com ≥ 2 fogos
  - frequência em 10 anos       ← soma de `burned_area_(Y−10) … _(Y−1)`
  - % da queima detectada pelo MapBiomas no ano Y ← média de `burned_area_Y` no polígono
  - cobertura                   ← MapBiomas col. 11 `classification_(Y−1)`
  - NPV no mês t0−1 por cobertura × tempo desde o último fogo (dentro da queima e do entorno)
  - sensibilidade: NPV t0−1 do entorno só nos pixels da mesma classe de cobertura dominante da queima

Nada é gravado no GEE (sem assets/exports). Saída bruta em JSON Lines (um registro por queima, com
os histogramas), processada por `historico_fogo_figuras.py`:
  tabelas/historico_fogo_exemplos_bruto.jsonl      31 exemplos da biblioteca
  tabelas/historico_fogo_populacao_<ano>_bruto.jsonl  todas as queimas 1+2 ≥ 5 ha (2019–2025)

Uso:
  python historico_fogo.py --exemplos
  python historico_fogo.py --populacao --years 2019 2020 2021 2022 2023 2024 2025
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuel_prescribed_burns_stats as fb  # noqa: E402  (inicializa o EE)
import ee  # noqa: E402

FIRE = 'projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_'
TAF = ee.Image(FIRE + 'time_after_fire_v1')
ISF = ee.Image(FIRE + 'interval_since_fire_v1')
ABR = ee.Image(FIRE + 'annual_burned_v1')
LULC = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3')
BIOMES = os.environ.get('BIOMES_ASSET', '')   # limites dos biomas do IBGE (FeatureCollection com a propriedade 'Bioma')
SEM_REGISTRO = 99
SCALE = fb.SCALE
HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tabelas')


def _names(prefix, y0, y1):
    """Lista server-side de nomes de banda prefix+ano, de y0 a y1 (inclusive)."""
    return ee.List.sequence(y0, y1).map(lambda y: ee.String(prefix).cat(ee.Number(y).format('%d')))


def history_stack(y):
    """Bandas do histórico de fogo para queimas cujo t0 cai no ano y (ee.Number)."""
    y = ee.Number(y)
    tsf = TAF.select(ee.String('time_after_fire_').cat(y.format('%d')))
    tsf = tsf.where(tsf.eq(0), SEM_REGISTRO).rename('tsf')
    iv = ISF.select(_names('interval_since_fire_', 1986, y.subtract(1)))
    iv = iv.updateMask(iv.gt(0))
    imr = iv.reduce(ee.Reducer.mean()).rename('imr')
    f10 = ABR.select(_names('burned_area_', y.subtract(10), y.subtract(1))).unmask(0).gt(0) \
        .reduce(ee.Reducer.sum()).rename('f10')
    det = ABR.select(ee.String('burned_area_').cat(y.format('%d'))).unmask(0).gt(0).rename('det')
    lulc = LULC.select(ee.String('classification_').cat(y.subtract(1).format('%d'))).rename('lulc')
    return tsf, imr, f10, det, lulc


def _rr(img, reducer, geom):
    return img.reduceRegion(reducer, geom, SCALE, maxPixels=1e10, tileScale=4)


def zone(prefix, geom, npv, tsf, imr, f10, lulc, mask=None):
    """Estatísticas de uma zona; `mask` (imagem 0/1) restringe os pixels (entorno)."""
    m = (lambda im: im.updateMask(mask)) if mask is not None else (lambda im: im)
    hist = ee.Reducer.frequencyHistogram()
    grouped = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True).group(groupField=1, groupName='k')
    return {
        prefix + 'tsf_hist': _rr(m(tsf), hist, geom).get('tsf'),
        prefix + 'f10_hist': _rr(m(f10), hist, geom).get('f10'),
        prefix + 'lulc_hist': _rr(m(lulc), hist, geom).get('lulc'),
        prefix + 'imr_med': _rr(m(imr), ee.Reducer.median(), geom).get('imr'),
        prefix + 'imr_mean': _rr(m(imr), ee.Reducer.mean(), geom).get('imr'),
        prefix + 'imr_n': _rr(m(imr), ee.Reducer.count(), geom).get('imr'),
        prefix + 'npv_t0m1': _rr(m(npv), ee.Reducer.mean(), geom).get('npv'),
        # chave = cobertura*100 + tempo desde o último fogo (ex.: 1203 = campo, 3 anos; 1299 = campo, sem registro)
        prefix + 'npv_por_cob_tsf': _rr(m(npv.addBands(lulc.multiply(100).add(tsf).rename('k'))),
                                        grouped, geom).get('groups'),
    }


def history_stats(fc, file_year, with_fuel=True):
    """Map server-side: uma linha por queima (sem geometria)."""
    mask_other = fb.burned_mask(file_year).unmask(0).Not()
    biomes = ee.FeatureCollection(BIOMES)

    def per(f):
        f = ee.Feature(f)
        t0 = ee.Date(f.get('burn_date'))
        y = t0.get('year', fb.TZ)
        geom = f.geometry()
        ent = geom.buffer(fb.RING_M, 10).difference(geom, 10)
        tsf, imr, f10, det, lulc = history_stack(y)
        npv = fb.fuel_window(t0, [-1]).select('npv') if with_fuel else ee.Image.constant(0).rename('npv').updateMask(0)
        props = {'t0_year': y}
        props.update(zone('queima_', geom, npv, tsf, imr, f10, lulc))
        props.update(zone('entorno_', ent, npv, tsf, imr, f10, lulc, mask_other))
        props['queima_det_pct'] = ee.Number(_rr(det, ee.Reducer.mean(), geom).get('det')).multiply(100)
        # cobertura dominante da queima e entorno restrito a ela (sensibilidade)
        # Reducer.mode devolve o centro de um bin (ex.: 12.0000000001, 11.17): arredondar para a classe
        moda = ee.Number(_rr(lulc, ee.Reducer.mode(), geom).get('lulc')).round()
        props['queima_lulc_moda'] = moda
        same = mask_other.And(lulc.eq(moda))
        props['entorno_mesma_cob_npv_t0m1'] = _rr(npv.updateMask(same), ee.Reducer.mean(), ent).get('npv')
        props['entorno_mesma_cob_n'] = _rr(npv.updateMask(same), ee.Reducer.count(), ent).get('npv')
        props['entorno_mesma_cob_pct'] = ee.Number(
            _rr(lulc.eq(moda).updateMask(mask_other), ee.Reducer.mean(), ent).get('lulc')).multiply(100)
        props['bioma'] = ee.Feature(biomes.filterBounds(geom.centroid(10)).first()).get('Bioma')
        return f.set(props).setGeometry(None)
    return fc.map(per)


KEEP = ['orig_id', 'year', 'burn_ymd', 'img_ymd', 'date_source', 'month', 'fire_type_code', 'nome_uc', 'area_ha']


def run(sel, file_year, n, out_path, with_fuel=True, chunk=10):
    rows = fb.chunked_getinfo(history_stats(sel, file_year, with_fuel), n, chunk=chunk, workers=3)
    with open(out_path, 'a', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    return rows


def exemplos():
    with open(os.path.join(TAB, 'biblioteca_exemplos.csv'), encoding='utf-8') as fh:
        lib = list(csv.DictReader(fh))
    out = os.path.join(TAB, 'historico_fogo_exemplos_bruto.jsonl')
    if os.path.exists(out):
        os.remove(out)
    for y in sorted({int(r['year']) for r in lib}):
        ids = [r['orig_id'] for r in lib if int(r['year']) == y]
        sel = fb.refine_t0(fb.load_icmbio([y]).filter(ee.Filter.inList('orig_id', ids)))
        print(f'[{y}] {len(ids)} exemplos…', flush=True)
        run(sel, y, len(ids), out, chunk=4)
    # confere t0 com a biblioteca
    got = {json.loads(l)['orig_id']: json.loads(l) for l in open(out, encoding='utf-8')}
    diff = [r['orig_id'] for r in lib if got.get(r['orig_id'], {}).get('burn_ymd') != r['burn_ymd']]
    print(f'{len(got)}/{len(lib)} exemplos; t0 diferente da biblioteca: {diff or "nenhum"}')


def populacao(years, min_area, with_fuel):
    for y in years:
        out = os.path.join(TAB, f'historico_fogo_populacao_{y}_bruto.jsonl')
        if os.path.exists(out):
            print(f'[{y}] já existe ({out}); apague para refazer', flush=True)
            continue
        sel = (fb.load_icmbio([y])
               .filter(ee.Filter.inList('fire_type_code', fb.SELECT_CODES))
               .filter(ee.Filter.notNull(['burn_date']))
               .filter(ee.Filter.gte('area_ha', min_area)))
        sel = fb.refine_t0(sel)
        n = sel.size().getInfo()
        print(f'[{y}] {n} queimas…', flush=True)
        tmp = out + '.parcial'
        if os.path.exists(tmp):
            os.remove(tmp)
        run(sel, y, n, tmp, with_fuel, chunk=10)
        os.replace(tmp, out)
        print(f'[{y}] ok → {out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exemplos', action='store_true', help='31 exemplos da biblioteca')
    ap.add_argument('--populacao', action='store_true', help='todas as queimas prescritas/controladas')
    ap.add_argument('--years', type=int, nargs='+', default=list(range(2019, 2026)))
    ap.add_argument('--min-area', type=float, default=fb.MIN_AREA_HA)
    ap.add_argument('--sem-combustivel', action='store_true', help='população: só histórico de fogo (mais rápido)')
    a = ap.parse_args()
    if a.exemplos:
        exemplos()
    if a.populacao:
        populacao(a.years, a.min_area, not a.sem_combustivel)


if __name__ == '__main__':
    main()
