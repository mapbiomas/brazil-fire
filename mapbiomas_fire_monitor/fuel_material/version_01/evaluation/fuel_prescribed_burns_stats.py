"""
Combustível acumulado antes das queimas prescritas — ICMBio × MapBiomas Fogo (Material Combustível)
=====================================================================================================

O que faz (tudo em memória, NÃO cria assets nem exports no GEE):
  1. Lê os vetores ICMBio `ICMBIO_FOGO/AAF_ICMBIO_<ano>` e harmoniza as colunas
     (data da queima, mês, tipo de fogo codificado, UC, área).
  2. Para cada queima selecionada (padrão: prescrita + controlada, área >= 5 ha) calcula a
     média de NPV/GV/Solo (0–100) no mosaico mensal do mês ANTERIOR e do mês POSTERIOR à
     data da queima, dentro do polígono e num anel de referência ao redor (sem outras
     cicatrizes ICMBio do mesmo ano).
  3. Grava localmente:
       tabelas/stats_<anos>.csv           estatísticas por queima
       figuras/boxplot_<anos>.png         NPV antes/depois, queima × anel
       figuras/series_<anos>.png          série mensal média de NPV (−12 … +3 meses)
       figuras/mapas_<ano>/*.png           um mapa por queima: RGB (NPV,GV,Solo) antes | depois, contorno,
                                      título, legenda, escala e norte. A imagem é renderizada pelo
                                      GEE (getThumbURL = equivalente Python do ui.Thumbnail); não baixa GeoTIFF.

Tabela de códigos de tipo de fogo (fire_type_code):
  1 queima prescrita · 2 queima controlada · 3 aceiro · 4 incêndio · 5 indígena
  6 fogo natural / raio · 7 gestão de ignição (natural/antrópica) · 0 outros / vazio

Uso:
  python fuel_prescribed_burns_stats.py --years 2024 2025
  python fuel_prescribed_burns_stats.py --years 2024 --uc "PARNA das Emas" --max-burns 30 --maps-n 10
  python fuel_prescribed_burns_stats.py --years 2024 --uc "PARNA das Emas" --timeline-n 3 --timeline-offsets=-6,-4,-2,-1,0,1,2

Saídas da linha do tempo:
  figuras/mensal_<ano>/*_mensal.png  cenas mês a mês (t0−4 … t0+3) + triângulo RGB NPV–GV–Solo com a
                                     trajetória mensal das frações médias da queima e do anel
  tabelas/mensal_<anos>.csv          médias mensais de NPV/GV/Solo (queima e anel) por queima

Também pode ser importado num notebook:
  import fuel_prescribed_burns_stats as fb
  sel = fb.load_icmbio([2024]).filter(...)
  fb.make_maps(sel, 2024, 5, '../figuras/mapas_2024', {})
"""
import argparse
import csv
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import ee

# ----------------------------------------------------------------------------- parâmetros
EE_PROJECT = os.environ.get('EE_PROJECT')               # seu projeto do Google Earth Engine
ICMBIO_BASE = os.environ.get('ICMBIO_BURNS_PREFIX', '')  # prefixo dos assets anuais de polígonos ICMBio (privado), ex.: .../AAF_ICMBIO_
FUEL_BASE = 'projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material/monthly_'
FUEL_METHOD = 'minnbr'        # 'minnbr' | 'median'
FUEL_VARIANT = 'default'      # 'default' (FUEL_MATERIAL_BYTE_AAAA_MM) | 'brasil' (prefere *_BRASIL_*, cai no default)
SELECT_CODES = [1, 2]         # tipos de fogo analisados
MIN_AREA_HA = 5
RING_M = 1000                 # largura do anel de referência
BEFORE_MONTHS = [1]           # meses antes de t0 usados no ANTES (ex.: [1, 2] = média de t0−1 e t0−2)
AFTER_MONTHS = [1]            # meses depois de t0 usados no DEPOIS
SCALE = 30
TZ = 'America/Sao_Paulo'
BANDS = ['npv', 'gv', 'soil']
VIS = {'bands': BANDS, 'min': [0, 5, 10], 'max': [17, 42, 90]}   # preset 'fuel' do Toolkit
TIMELINE_OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3]   # cenas mês a mês na linha do tempo (0 = mês da queima)
TIMELINE_COLS = 4

FIRE_TYPE_CODES = {
    'queima prescrita': 1,
    'queima controlada': 2,
    'aceiro': 3,
    'incendio': 4, 'incêndio': 4,
    'indigena': 5, 'indígena': 5, 'queima por indigenas isolados': 5,
    'fogo natural': 6, 'raio': 6,
    'gestao de ignicao natural': 7, 'gestao de ignicao antropica': 7,
}
FIRE_TYPE_NAMES = {0: 'outros', 1: 'queima prescrita', 2: 'queima controlada', 3: 'aceiro',
                   4: 'incendio', 5: 'indigena', 6: 'fogo natural', 7: 'gestao de ignicao'}


# ----------------------------------------------------------------------------- harmonização
def harmonize_year(year):
    """FeatureCollection ICMBio de um ano com colunas harmonizadas."""
    fc = ee.FeatureCollection(ICMBIO_BASE + str(year))
    codes = ee.Dictionary(FIRE_TYPE_CODES)

    def fix(f):
        f = ee.Feature(f)
        names = f.propertyNames()
        # data: 'data' (2024+) ou 'data_img' (ms até 2022, string 'AAAA-MM-DD' em 2023)
        raw = ee.Algorithms.If(names.contains('data'), f.get('data'), f.get('data_img'))
        jul = ee.Algorithms.If(names.contains('juliano'), f.get('juliano'), 0)
        jul = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(jul, None), 0, jul))
        fallback = ee.Algorithms.If(jul.gt(0),
                                    ee.Date.fromYMD(year, 1, 1).advance(jul.subtract(1), 'day').millis(),
                                    None)
        ms = ee.Algorithms.If(ee.Algorithms.IsEqual(raw, None), fallback,
                              ee.Algorithms.If(ee.Algorithms.IsEqual(raw, ''), fallback,
                                               ee.Date(raw).millis()))
        date_ok = ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(ms, None), 0, 1))
        d_img = ee.Date(ee.Algorithms.If(date_ok, ms, 0))
        # data real da queima: 'obs' costuma trazer "Queima do dia dd/mm" (a data da imagem pode vir
        # semanas depois). Usa a data do obs se for válida e cair até 120 dias antes da imagem.
        obs = ee.Algorithms.If(names.contains('obs'), f.get('obs'), '')
        obs = ee.String(ee.Algorithms.If(ee.Algorithms.IsEqual(obs, None), '', obs))
        found = obs.match('(\\d{1,2})/(\\d{1,2})')
        # o .cat() só evita erro de índice quando não há "dd/mm" no obs; nesse caso a data do obs
        # NÃO é válida (antes, o preenchimento '01/01' virava uma data falsa de 1º de janeiro)
        mt = found.cat(['0', '1', '1'])
        dd, mm = ee.Number.parse(mt.get(1)), ee.Number.parse(mt.get(2))
        valid_dm = found.size().gt(0).And(dd.gte(1)).And(dd.lte(31)).And(mm.gte(1)).And(mm.lte(12))
        yy = d_img.get('year', TZ)
        d_obs = ee.Date.fromYMD(yy, mm.max(1).min(12), dd.max(1).min(28), TZ).advance(dd.subtract(dd.max(1).min(28)), 'day')
        d_obs = ee.Date(ee.Algorithms.If(d_obs.millis().gt(d_img.millis()), d_obs.advance(-1, 'year'), d_obs))
        lag = d_img.difference(d_obs, 'day')
        use_obs = ee.Number(date_ok).And(valid_dm).And(lag.gte(0)).And(lag.lte(120))
        d = ee.Date(ee.Algorithms.If(use_obs, d_obs, d_img))
        src = ee.Algorithms.If(use_obs, 'obs', ee.Algorithms.If(
            ee.Algorithms.IsEqual(raw, None), 'juliano', 'data_img'))
        # tipo: 'acao' (2025) ou 'classe'
        txt = ee.String(ee.Algorithms.If(names.contains('acao'), f.get('acao'),
                                         ee.Algorithms.If(names.contains('classe'), f.get('classe'), '')))
        txt = ee.String(ee.Algorithms.If(ee.Algorithms.IsEqual(txt, None), '', txt)).toLowerCase().trim()
        code = ee.Number(codes.get(txt, 0))
        oid = ee.Algorithms.If(names.contains('pk_aaf'), ee.Number(f.get('pk_aaf')).format('%d'),
                               ee.Algorithms.If(names.contains('GlobalID'), f.get('GlobalID'), f.id()))
        geom = f.geometry()
        return ee.Feature(geom, {
            'source': 'ICMBio',
            'orig_id': ee.String(str(year)).cat('_').cat(ee.String(oid)).cat('_').cat(f.id()),
            'year': year,
            'burn_date': ee.Algorithms.If(date_ok, d.millis(), None),
            'burn_ymd': ee.Algorithms.If(date_ok, d.format('YYYY-MM-dd', TZ), None),
            'month': ee.Algorithms.If(date_ok, d.get('month', TZ), None),
            'doy': ee.Algorithms.If(date_ok, d.getRelative('day', 'year', TZ).add(1), None),
            'date_source': ee.Algorithms.If(date_ok, src, None),
            'img_ymd': ee.Algorithms.If(date_ok, d_img.format('YYYY-MM-dd', TZ), None),
            'fire_type': txt,
            'fire_type_code': code,
            'nome_uc': f.get('nome_uc'),
            'cnuc': f.get('cnuc'),
            'obs': f.get('obs'),
            'area_ha': geom.area(10).divide(1e4),
        })
    return fc.map(fix)


def short_id(orig_id):
    """Id curto para nomes de arquivo: id original (pk_aaf/GlobalID) ou, se vazio, o system:index."""
    parts = orig_id.split('_')
    return parts[1] if len(parts) > 1 and parts[1] else parts[-1]


def load_icmbio(years):
    return ee.FeatureCollection([harmonize_year(y) for y in years]).flatten()


# ----------------------------------------------------------------------------- t0 pela série
# Para queimas SEM data no 'obs' (date_source = data_img | juliano), a data da imagem pode estar
# meses depois do fogo. Estimamos o mês da queima pela própria série do produto: a soma média
# NPV+GV+Solo no polígono despenca na cicatriz (a sombra/carvão domina; ex.: ~43% → ~14%).
# Entre t_img−4 … t_img, o mês com a MAIOR queda em relação ao mês anterior vira t0, se a queda
# for ≥ T0_MIN_DROP pontos e ≥ T0_MIN_REL da soma anterior. Senão, t0 = data da imagem.
T0_OFFSETS = [-4, -3, -2, -1, 0]
T0_MIN_DROP = 5.0
T0_MIN_REL = 0.20


def _estimate_t0(f):
    f = ee.Feature(f)
    t_img = ee.Date(f.get('burn_date'))
    geom = f.geometry()
    stack = ee.Image.cat([fuel_image(t_img.advance(o, 'month')).reduce(ee.Reducer.sum()).rename(f's{i}')
                          for i, o in enumerate(T0_OFFSETS)])
    dct = stack.reduceRegion(ee.Reducer.mean(), geom, SCALE * 2, maxPixels=1e10, tileScale=4)
    vals = [ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(dct.get(f's{i}'), None), -1, dct.get(f's{i}')))
            for i in range(len(T0_OFFSETS))]
    drops = ee.List([ee.Algorithms.If(vals[i - 1].gte(0).And(vals[i].gte(0)), vals[i - 1].subtract(vals[i]), -999)
                     for i in range(1, len(T0_OFFSETS))])
    mx = ee.Number(drops.reduce(ee.Reducer.max()))
    k = drops.indexOf(mx)
    prev = ee.Number(ee.List(vals).get(k))
    ok = mx.gte(T0_MIN_DROP).And(mx.gte(prev.multiply(T0_MIN_REL)))
    off = ee.Number(ee.List(T0_OFFSETS[1:]).get(k))
    shift = ee.Number(ee.Algorithms.If(ok, off, 0))
    m = t_img.advance(shift, 'month')
    new = ee.Date(ee.Algorithms.If(shift.eq(0), t_img, ee.Date.fromYMD(m.get('year', TZ), m.get('month', TZ), 15, TZ)))
    return f.set({
        'burn_date': new.millis(),
        'burn_ymd': new.format('YYYY-MM-dd', TZ),
        'month': new.get('month', TZ),
        'doy': new.getRelative('day', 'year', TZ).add(1),
        'date_source': ee.Algorithms.If(shift.lt(0), 'serie', f.get('date_source')),
        't0_shift_months': shift,
        't0_max_drop': mx,
    })


def refine_t0(fc):
    """Aplica a estimativa de t0 só às queimas cuja data não veio do 'obs'."""
    return fc.map(lambda f: ee.Feature(ee.Algorithms.If(
        ee.Algorithms.IsEqual(f.get('date_source'), 'obs'), f.set('t0_shift_months', 0), _estimate_t0(f))))


# ----------------------------------------------------------------------------- combustível
ee.Initialize(project=EE_PROJECT)
FUEL = ee.ImageCollection(FUEL_BASE + FUEL_METHOD)
EMPTY = ee.Image.constant([0, 0, 0]).rename(BANDS).updateMask(0).toFloat()


def fuel_image(date):
    """Mosaico mensal (npv,gv,soil) do mês que contém `date`; imagem vazia se não houver."""
    date = ee.Date(date)
    col = (FUEL.filter(ee.Filter.eq('year', date.get('year', TZ)))
               .filter(ee.Filter.eq('month', date.get('month', TZ))))
    is_br = ee.Filter.stringContains('system:index', '_BRASIL_')
    dflt = col.filter(is_br.Not())
    if FUEL_VARIANT == 'brasil':
        br = col.filter(is_br)
        col = ee.ImageCollection(ee.Algorithms.If(br.size().gt(0), br, dflt))
    else:
        col = dflt
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.first().select(BANDS).toFloat(), EMPTY))


def fuel_window(t0, offsets):
    """Média dos mosaicos nos meses t0+offset (offset em meses)."""
    t0 = ee.Date(t0)
    return ee.ImageCollection([fuel_image(t0.advance(o, 'month')) for o in offsets]).mean().rename(BANDS)


# ----------------------------------------------------------------------------- estatísticas
def burned_mask(year):
    """1 onde há qualquer polígono ICMBio (qualquer classe) no ano — excluído do anel."""
    return ee.Image(0).paint(ee.FeatureCollection(ICMBIO_BASE + str(year)), 1).selfMask()


def _zone_stats(img, geom, prefix):
    red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
    d = img.reduceRegion(red, geom, SCALE, maxPixels=1e10, tileScale=4)
    out = {}
    for b in BANDS:
        out[prefix + b] = d.get(b + '_mean')
    out[prefix + 'n'] = d.get('npv_count')
    return out


def compute_stats(fc, year):
    """Map server-side: uma linha de estatísticas por queima de um ano."""
    mask_other = burned_mask(year).unmask(0).Not()

    def per(f):
        f = ee.Feature(f)
        t0 = ee.Date(f.get('burn_date'))
        geom = f.geometry()
        ring = geom.buffer(RING_M, 10).difference(geom, 10)
        before = fuel_window(t0, [-m for m in BEFORE_MONTHS])
        after = fuel_window(t0, AFTER_MONTHS)
        props = {}
        props.update(_zone_stats(before, geom, 'burn_before_'))
        props.update(_zone_stats(after, geom, 'burn_after_'))
        props.update(_zone_stats(before.updateMask(mask_other), ring, 'ring_before_'))
        props.update(_zone_stats(after.updateMask(mask_other), ring, 'ring_after_'))
        props['n_pix_total'] = ee.Image.pixelArea().reduceRegion(
            ee.Reducer.count(), geom, SCALE, maxPixels=1e10).get('area')
        return f.set(props).setGeometry(None)
    return fc.map(per)


def monthly_series(f, year, offsets=range(-12, 4)):
    """Série de NPV médio (queima, anel) para uma feição — usada para os maiores polígonos."""
    f = ee.Feature(f)
    t0 = ee.Date(f.get('burn_date'))
    geom = f.geometry()
    ring = geom.buffer(RING_M, 10).difference(geom, 10)
    mask_other = burned_mask(year).unmask(0).Not()

    def one(o):
        img = fuel_image(t0.advance(o, 'month')).select('npv')
        a = img.reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=1e10, tileScale=4).get('npv')
        b = img.updateMask(mask_other).reduceRegion(ee.Reducer.mean(), ring, SCALE, maxPixels=1e10, tileScale=4).get('npv')
        return ee.Feature(None, {'offset': o, 'burn': a, 'ring': b})
    return ee.FeatureCollection([one(o) for o in offsets])


# ----------------------------------------------------------------------------- helpers locais
def retry(fn, tries=6):
    import time
    for k in range(tries):
        try:
            return fn()
        except ee.EEException as e:
            if k == tries - 1 or not any(t in str(e) for t in ('Too many concurrent', 'Computation timed out', 'quota')):
                raise
            time.sleep(5 * 2 ** k)
        except Exception as e:  # falhas de rede ao baixar a miniatura (IncompleteRead, ConnectionError…)
            import requests
            if k == tries - 1 or not isinstance(e, (requests.exceptions.RequestException, ConnectionError,
                                                    OSError)) and 'Connection' not in str(e):
                raise
            time.sleep(5 * 2 ** k)


def chunked_getinfo(fc, n, chunk=20, workers=2):
    lst = fc.toList(n)

    def get(i):
        return retry(lambda: ee.FeatureCollection(lst.slice(i, min(i + chunk, n))).getInfo()['features'])
    rows = []
    with ThreadPoolExecutor(workers) as ex:
        for part in ex.map(get, range(0, n, chunk)):
            rows += [p['properties'] for p in part]
    return rows


def thumb(img, region, dims=360):
    import requests
    from PIL import Image
    url = img.getThumbURL({'region': region, 'dimensions': dims, 'format': 'png'})
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert('RGB')


def rgb_with_outline(img, geom, ring_geom):
    bg = ee.Image(0).visualize(palette=['202020'])
    vis = bg.blend(img.select(BANDS).visualize(**VIS))
    outline = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(geom)]), 1, 2)
    ring_line = ee.Image().byte().paint(ee.FeatureCollection([ee.Feature(ring_geom)]), 1, 1)
    return vis.blend(ring_line.visualize(palette=['ffffff'])).blend(outline.visualize(palette=['ffff00']))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=[2024, 2025])
    ap.add_argument('--uc', nargs='*', default=None, help='filtrar por nome_uc (exato)')
    ap.add_argument('--max-burns', type=int, default=5000)
    ap.add_argument('--maps-n', type=int, default=8, help='mapas antes|depois por ano, das maiores queimas (0 = sem mapas)')
    ap.add_argument('--series-n', type=int, default=15, help='queimas por ano na série mensal (0 = sem série)')
    ap.add_argument('--no-refine-t0', action='store_true',
                    help='não estimar t0 pela série nas queimas sem data no obs')
    ap.add_argument('--timeline-n', type=int, default=5,
                    help='figuras mês a mês (cenas t0−4…t0+3 + triângulo) por ano, das maiores queimas (0 = nenhuma)')
    ap.add_argument('--timeline-offsets', type=lambda s: [int(x) for x in s.split(',')], default=TIMELINE_OFFSETS,
                    help='meses relativos a t0, separados por vírgula (ex.: -6,-4,-2,-1,0,1,2)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
                    help='pasta da análise; grava em <out>/figuras e <out>/tabelas')
    args = ap.parse_args()

    fig_dir, tab_dir = os.path.join(args.out, 'figuras'), os.path.join(args.out, 'tabelas')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    tag = '_'.join(map(str, args.years))

    all_rows, series_rows, monthly_rows = [], [], []
    for y in args.years:
        fc = load_icmbio([y])
        print(f'[{y}] tipos (todas as feições):', fc.aggregate_histogram('fire_type_code').getInfo())
        sel = (fc.filter(ee.Filter.inList('fire_type_code', SELECT_CODES))
                 .filter(ee.Filter.notNull(['burn_date']))
                 .filter(ee.Filter.gte('area_ha', MIN_AREA_HA)))
        if args.uc:
            sel = sel.filter(ee.Filter.inList('nome_uc', args.uc))
        sel = sel.sort('area_ha', False).limit(args.max_burns)
        if not args.no_refine_t0:
            sel = refine_t0(sel)
        n = sel.size().getInfo()
        print(f'[{y}] queimas selecionadas: {n}')
        if n == 0:
            continue
        print(f'[{y}] calculando estatísticas…', flush=True)
        rows = chunked_getinfo(compute_stats(sel, y), n)
        all_rows += rows

        if args.series_n:
            print(f'[{y}] séries mensais…', flush=True)
            top = sel.limit(args.series_n).toList(args.series_n)
            k = min(args.series_n, n)
            def get_series(i):
                f = ee.Feature(top.get(i))
                oid = f.get('orig_id').getInfo()
                s = retry(lambda: monthly_series(f, y).getInfo()['features'])
                return [dict(p['properties'], orig_id=oid) for p in s]
            with ThreadPoolExecutor(2) as ex:
                for part in ex.map(get_series, range(k)):
                    series_rows += part

        if args.maps_n:
            print(f'[{y}] mapas…', flush=True)
            make_maps(sel, y, min(args.maps_n, n), os.path.join(fig_dir, f'mapas_{y}'),
                      {r['orig_id']: r for r in rows})

        if args.timeline_n:
            print(f'[{y}] linhas do tempo mês a mês…', flush=True)
            monthly_rows += make_timelines(sel, y, min(args.timeline_n, n), os.path.join(fig_dir, f'mensal_{y}'),
                                           args.timeline_offsets)

    if not all_rows:
        print('Nenhuma queima selecionada.')
        return
    # derivados
    for r in all_rows:
        def g(k):
            v = r.get(k)
            return None if v is None else float(v)
        bb, ba, rb = g('burn_before_npv'), g('burn_after_npv'), g('ring_before_npv')
        r['dNPV_burn'] = None if bb is None or ba is None else ba - bb
        r['NPV_excess_vs_ring'] = None if bb is None or rb is None else bb - rb
        tot = g('n_pix_total')
        r['pct_valid_before'] = None if not tot or r.get('burn_before_n') is None else 100 * r['burn_before_n'] / tot
        r['fire_type_name'] = FIRE_TYPE_NAMES.get(int(r.get('fire_type_code') or 0))
    cols = ['source', 'orig_id', 'year', 'burn_ymd', 'img_ymd', 'date_source', 't0_shift_months', 'month', 'doy', 'fire_type', 'fire_type_code', 'fire_type_name',
            'nome_uc', 'cnuc', 'area_ha', 'obs',
            'burn_before_npv', 'burn_before_gv', 'burn_before_soil', 'burn_before_n',
            'burn_after_npv', 'burn_after_gv', 'burn_after_soil', 'burn_after_n',
            'ring_before_npv', 'ring_before_gv', 'ring_before_soil', 'ring_before_n',
            'ring_after_npv', 'ring_after_gv', 'ring_after_soil', 'ring_after_n',
            'n_pix_total', 'pct_valid_before', 'dNPV_burn', 'NPV_excess_vs_ring']
    csv_path = os.path.join(tab_dir, f'stats_{tag}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print('CSV:', csv_path)
    summarize(all_rows)
    make_boxplot(all_rows, os.path.join(fig_dir, f'boxplot_{tag}.png'))
    if series_rows:
        make_series(series_rows, os.path.join(fig_dir, f'series_{tag}.png'))
    if monthly_rows:
        mcols = ['orig_id', 'nome_uc', 'burn_ymd', 'offset'] + [f'{z}_{b}' for z in ('burn', 'ring') for b in BANDS]
        mpath = os.path.join(tab_dir, f'mensal_{tag}.csv')
        with open(mpath, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, mcols, extrasaction='ignore')
            w.writeheader()
            w.writerows(monthly_rows)
        print('CSV mensal:', mpath)


def summarize(rows):
    import statistics as st
    def col(k):
        return [float(r[k]) for r in rows if r.get(k) is not None]
    for k in ['burn_before_npv', 'burn_after_npv', 'ring_before_npv', 'ring_after_npv', 'dNPV_burn', 'NPV_excess_vs_ring']:
        v = col(k)
        if v:
            print(f'  {k:22s} n={len(v):5d}  média={st.mean(v):6.1f}  mediana={st.median(v):6.1f}')
    v = col('dNPV_burn')
    if v:
        print(f'  queimas com queda de NPV após o fogo: {100 * sum(x < 0 for x in v) / len(v):.0f}%')


def make_boxplot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [('burn_before_npv', 'Queima\nantes'), ('burn_after_npv', 'Queima\ndepois'),
            ('ring_before_npv', 'Entorno\nantes'), ('ring_after_npv', 'Entorno\ndepois')]
    data = [[float(r[k]) for r in rows if r.get(k) is not None] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bp = ax.boxplot(data, tick_labels=[l for _, l in keys], showfliers=False, patch_artist=True)
    for p, c in zip(bp['boxes'], ['#c0392b', '#e8a09a', '#7f8c8d', '#c7cdd1']):
        p.set_facecolor(c)
    ax.set_ylabel('NPV médio (%)')
    ax.set_title(f'Material combustível (NPV) – queimas ICMBio (n={len(rows)})\n'
                 f'mosaico {FUEL_METHOD}, antes = t0−{BEFORE_MONTHS} mês, depois = t0+{AFTER_MONTHS} mês')
    ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print('Boxplot:', path)


def make_series(series_rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import defaultdict
    acc = defaultdict(lambda: {'burn': [], 'ring': []})
    for r in series_rows:
        for z in ('burn', 'ring'):
            if r.get(z) is not None:
                acc[r['offset']][z].append(float(r[z]))
    offs = sorted(acc)
    mean = lambda v: sum(v) / len(v) if v else float('nan')
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(offs, [mean(acc[o]['burn']) for o in offs], '-o', color='#c0392b', label='Queima')
    ax.plot(offs, [mean(acc[o]['ring']) for o in offs], '-o', color='#7f8c8d', label=f'Entorno {RING_M} m')
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Meses em relação à data da queima (t0)')
    ax.set_ylabel('NPV médio (%)')
    ax.set_title(f'Série mensal média de NPV ({len({r["orig_id"] for r in series_rows})} maiores queimas)')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print('Série:', path)


# ----------------------------------------------------------------------------- cartografia (PIL)
# Título, legenda RGB, escala e norte seguem as fórmulas de
# (lógica adaptada de utils/mapbiomas_gif_factory)
# (_find_font, _render_rgb_legend, add_scale_bar, draw_north), adaptadas para a figura antes|depois.
MONTHS_PT = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']


def _font(size, bold=False):
    from PIL import ImageFont
    cands = ([r'C:\Windows\Fonts\arialbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else []) + [
        r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\segoeui.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def map_bounds(geojson_geom, pad_m):
    """[lon_min, lat_min, lon_max, lat_max] da geometria + pad_m metros (calculado no cliente)."""
    import math
    pts = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            pts.append(c)
        else:
            for x in c:
                walk(x)
    walk(geojson_geom['coordinates'])
    lons, lats = [q[0] for q in pts], [q[1] for q in pts]
    lat_c = (min(lats) + max(lats)) / 2
    dlat = pad_m / 111320.0
    dlon = pad_m / (111320.0 * math.cos(math.radians(lat_c)))
    return [min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat]


def _month_label(date_ms, offsets):
    import datetime as dt
    d = dt.datetime.fromtimestamp(date_ms / 1000, dt.timezone(dt.timedelta(hours=-3)))
    out = []
    for o in offsets:
        k = d.month - 1 + o
        out.append(f'{MONTHS_PT[k % 12]}/{d.year + k // 12}')
    return ' + '.join(out)


def _scale_bar(draw, x_right, y, bounds, map_w_px, font, target_px=180):
    """Barra de escala — mesma fórmula da fábrica de GIFs (km/px pela largura em graus × cos(lat))."""
    import math
    lon_min, lat_min, lon_max, lat_max = bounds
    km_per_px = (lon_max - lon_min) * 111.32 * math.cos(math.radians((lat_min + lat_max) / 2)) / map_w_px
    nice = [0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 15, 20, 25, 50, 100, 200, 500]
    km = min(nice, key=lambda v: abs(v - km_per_px * target_px))
    px = int(round(km / km_per_px))
    x0 = x_right - px
    for i in range(4):  # barra alternada preto/branco
        xa, xb = x0 + i * px // 4, x0 + (i + 1) * px // 4
        draw.rectangle([xa, y, xb, y + 8], fill=(0, 0, 0) if i % 2 == 0 else (255, 255, 255), outline=(0, 0, 0))
    lbl = f'{km:g} km'
    draw.text((x0 - 4, y + 12), '0', font=font, fill=(0, 0, 0))
    w = draw.textbbox((0, 0), lbl, font=font)[2]
    draw.text((x_right - w // 2, y + 12), lbl, font=font, fill=(0, 0, 0))


def _north(draw, cx, cy, size, font):
    draw.polygon([(cx, cy - size), (cx - size * 0.5, cy + size * 0.4), (cx, cy + size * 0.1)], fill=(0, 0, 0))
    draw.polygon([(cx, cy - size), (cx + size * 0.5, cy + size * 0.4), (cx, cy + size * 0.1)],
                 fill=(255, 255, 255), outline=(0, 0, 0))
    w = draw.textbbox((0, 0), 'N', font=font)[2]
    draw.text((cx - w / 2, cy - size - font.size - 4), 'N', font=font, fill=(0, 0, 0))


TERN_COLORS = {'burn': (255, 255, 0), 'ring': (255, 255, 255)}


def _frac_to_rgb(n, g, s):
    """Mesma cor do mapa: cada fração esticada pelo preset VIS (min/max por banda)."""
    out = []
    for v, lo, hi in zip((n, g, s), VIS['min'], VIS['max']):
        out.append(int(max(0, min(1, (v - lo) / (hi - lo))) * 255))
    return tuple(out)


TERN_NPV_MAX = 40   # recorte do triângulo (NPV 0–npv_max %): amplia a região onde ficam os dados
TERN_VSTRETCH = 2.5  # exagero vertical do eixo NPV (só visual; 1 = triângulo equilátero)


def ternary_image(size=330, tracks=None, font_size=14, npv_max=TERN_NPV_MAX, vstretch=TERN_VSTRETCH):
    """Triângulo NPV–GV–Solo com fundo nas cores do mapa (RGB = NPV, GV, Solo) e trajetórias.

    tracks: lista de dicts {'kind': 'burn'|'ring', 'points': [(tag, npv, gv, soil), ...]};
    as frações são renormalizadas para somar 100 antes de plotar.
    npv_max < 100 mostra só o trapézio NPV 0–npv_max %: com a mesma altura cabe um triângulo
    ~100/npv_max vezes maior, então os pontos ficam bem mais separados.
    Sobre o fundo: a cor de cada ponto do triângulo é a cor que essa mistura teria no mapa.
    """
    import math
    from PIL import Image, ImageDraw
    pad_x, pad_top, pad_bot = 52, 30, 50
    # As frações do produto não somam 100 (a sombra/nuvem do SMA não é exportada). Os pontos são
    # normalizados (NPV+GV+Solo = 100%); o fundo é pintado com a mistura reescalada para a soma
    # típica dos pontos da figura (mediana), para reproduzir as cores do mapa.
    sums = sorted(n + g + s_ for tr in (tracks or []) for _, n, g, s_ in tr['points']
                  if None not in (n, g, s_) and (n + g + s_) > 0)
    color_sum = sums[len(sums) // 2] if sums else 100.0
    side = size - 2 * pad_x
    S = side                                 # base GV–Solo ocupa a largura toda
    Hs = S * math.sqrt(3) / 2 * vstretch    # altura do triângulo (esticada verticalmente)
    base_y = pad_top + Hs * npv_max / 100.0  # y da base GV–Solo
    cx = size / 2
    left = (cx - S / 2, base_y)
    right = (cx + S / 2, base_y)
    top = (cx, base_y - Hs)
    vis_h = Hs * npv_max / 100.0             # altura visível (NPV até npv_max)
    img_h = int(pad_top + vis_h + pad_bot) if npv_max < 100 else int(pad_top + Hs + pad_bot)
    if npv_max >= 100:
        base_y = pad_top + Hs
        left, right, top = (cx - S / 2, base_y), (cx + S / 2, base_y), (cx, pad_top)
    img = Image.new('RGB', (size, img_h), (255, 255, 255))
    px = img.load()

    def to_xy(n, g, s):
        t = n + g + s
        n, g, s = n / t, g / t, s / t
        return (n * top[0] + g * left[0] + s * right[0], n * top[1] + g * left[1] + s * right[1])

    det = (left[1] - right[1]) * (top[0] - right[0]) + (right[0] - left[0]) * (top[1] - right[1])
    y0 = int(max(0, base_y - (vis_h if npv_max < 100 else Hs)))
    for y in range(y0, int(base_y) + 1):
        for x in range(0, size):
            a = ((left[1] - right[1]) * (x - right[0]) + (right[0] - left[0]) * (y - right[1])) / det
            b = ((right[1] - top[1]) * (x - right[0]) + (top[0] - right[0]) * (y - right[1])) / det
            c = 1 - a - b
            if a >= 0 and b >= 0 and c >= 0 and a * 100 <= npv_max + 1e-9:
                px[x, y] = _frac_to_rgb(color_sum * a, color_sum * b, color_sum * c)
    d = ImageDraw.Draw(img)
    f, fb = _font(font_size), _font(font_size + 1, True)

    step = 10 if npv_max <= 50 else 20
    for k in range(step, int(npv_max) + 1, step):
        if k >= 100:
            break
        p1, p2 = to_xy(k, 100 - k, 1e-9), to_xy(k, 1e-9, 100 - k)
        d.line([p1, p2], fill=(255, 255, 255), width=1)
        d.text((p2[0] + 4, p2[1] - 8), f'{k}%', font=f, fill=(90, 90, 90))
    # contorno (triângulo ou trapézio)
    if npv_max < 100:
        tl, tr_ = to_xy(npv_max, 100 - npv_max, 1e-9), to_xy(npv_max, 1e-9, 100 - npv_max)
        d.polygon([left, right, tr_, tl], outline=(0, 0, 0))
        d.text((tl[0] - 6, tl[1] - font_size - 10), f'NPV ↑ (até {npv_max:g}%)', font=fb, fill=(200, 0, 0))
    else:
        d.polygon([top, left, right], outline=(0, 0, 0))
        d.text((top[0] - 14, top[1] - font_size - 14), 'NPV', font=fb, fill=(200, 0, 0))
    d.text((left[0] - 30, left[1] - font_size), 'GV', font=fb, fill=(0, 140, 0))
    d.text((right[0] + 6, right[1] - font_size), 'Solo', font=fb, fill=(0, 60, 200))

    def halo_text(xy, t):
        x, y = xy
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            d.text((x + dx, y + dy), t, font=f, fill=(255, 255, 255))
        d.text((x, y), t, font=f, fill=(0, 0, 0))

    for tr in tracks or []:
        pts = [(tag, to_xy(n, g, s)) for tag, n, g, s in tr['points']
               if None not in (n, g, s) and (n + g + s) > 0]
        if not pts:
            continue
        col = TERN_COLORS.get(tr['kind'], (255, 255, 0))
        if len(pts) > 1:
            d.line([q for _, q in pts], fill=(0, 0, 0), width=4 if tr['kind'] == 'burn' else 3)
            d.line([q for _, q in pts], fill=col, width=2 if tr['kind'] == 'burn' else 1)
        for i, (tag, (x, y)) in enumerate(pts):
            r = 5 if tr['kind'] == 'burn' else 4
            d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(0, 0, 0))
            if tr['kind'] == 'burn' and tag is not None:
                lbl = str(tag) if not isinstance(tag, int) else ('t0' if tag == 0 else f'{tag:+d}')
                tw = d.textbbox((0, 0), lbl, font=f)[2]
                lx = x + 7 if x + 7 + tw < size - pad_x + 10 else x - 7 - tw   # perto da borda: à esquerda
                halo_text((lx, y - 18 if i % 2 == 0 else y + 4), lbl)
    fn = _font(font_size - 2)
    d.text((pad_x - 30, img.height - 2 * (font_size + 2) - 4),
           f'Frações sem sombra (NPV+GV+Solo = 100%) · fundo na cor do mapa p/ soma ≈ {color_sum:.0f}%',
           font=fn, fill=(70, 70, 70))
    if vstretch != 1:
        d.text((pad_x - 30, img.height - font_size - 4),
               f'Eixo NPV esticado ×{vstretch:g} na vertical (só para leitura; distâncias não são as do triângulo)',
               font=fn, fill=(70, 70, 70))
    return img


def _grow_to(fig, height):
    """Aumenta a altura da figura (fundo branco) se o conteúdo à direita não couber."""
    from PIL import Image, ImageDraw
    if height > fig.height:
        new = Image.new('RGB', (fig.width, height), (255, 255, 255))
        new.paste(fig, (0, 0))
        fig = new
    return fig, ImageDraw.Draw(fig)


def compose_burn_map(p, before, after, bounds, stats=None):
    """Figura: título | ANTES · DEPOIS | legenda RGB + contornos + médias de NPV + escala + norte."""
    from PIL import Image, ImageDraw
    W, H = before.size
    gap, margin = 16, 24
    top_h, bot_h = 96, 132
    tern_w = 520
    fig = Image.new('RGB', (2 * W + 2 * gap + tern_w + 2 * margin, top_h + H + bot_h), (255, 255, 255))
    d = ImageDraw.Draw(fig)
    f_t, f_s, f_l, f_b = _font(26, True), _font(17), _font(16), _font(18, True)

    tipo = FIRE_TYPE_NAMES.get(int(p.get('fire_type_code') or 0), p.get('fire_type'))
    d.text((margin, 14), f"{p.get('nome_uc') or '?'} — {tipo} em {p['burn_ymd']} ({p['area_ha']:.0f} ha)",
           font=f_t, fill=(0, 0, 0))
    d.text((margin, 54), f'MapBiomas Fogo – Material Combustível (HLS, mosaico mensal {FUEL_METHOD}) · '
                         f'polígono ICMBio AAF {p["year"]}', font=f_s, fill=(70, 70, 70))

    for i, (im, lab, offs) in enumerate([(before, 'ANTES', [-m for m in BEFORE_MONTHS]),
                                         (after, 'DEPOIS', AFTER_MONTHS)]):
        x = margin + i * (W + gap)
        fig.paste(im, (x, top_h))
        d.rectangle([x, top_h, x + W - 1, top_h + H - 1], outline=(0, 0, 0), width=1)
        tag = f'{lab} · {_month_label(p["burn_date"], offs)}'
        tw = d.textbbox((0, 0), tag, font=f_b)[2]
        d.rectangle([x + 8, top_h + 8, x + 20 + tw, top_h + 36], fill=(255, 255, 255))
        d.text((x + 14, top_h + 10), tag, font=f_b, fill=(0, 0, 0))

    # legenda RGB (como _render_rgb_legend), contornos e médias
    y = top_h + H + 14
    d.text((margin, y), 'Composição RGB:', font=f_b, fill=(0, 0, 0))
    bx = margin + d.textbbox((0, 0), 'Composição RGB: ', font=f_b)[2]
    for (lbl, col), lo, hi in zip([('NPV (palha seca)', (220, 40, 40)), ('GV (verde)', (40, 170, 60)),
                                   ('Solo', (50, 90, 220))], VIS['min'], VIS['max']):
        d.rectangle([bx, y + 2, bx + 18, y + 20], fill=col, outline=(90, 90, 90))
        t = f'{lbl} {lo}–{hi}%'
        d.text((bx + 24, y), t, font=f_l, fill=(0, 0, 0))
        bx += 24 + d.textbbox((0, 0), t, font=f_l)[2] + 22
    y += 30
    d.rectangle([margin, y + 2, margin + 30, y + 20], fill=(32, 32, 32))
    d.line([margin + 3, y + 11, margin + 27, y + 11], fill=(255, 255, 0), width=3)
    t1 = 'Queima (ICMBio)'
    d.text((margin + 38, y), t1, font=f_l, fill=(0, 0, 0))
    x2 = margin + 38 + d.textbbox((0, 0), t1, font=f_l)[2] + 26
    d.rectangle([x2, y + 2, x2 + 30, y + 20], fill=(32, 32, 32))
    d.line([x2 + 3, y + 11, x2 + 27, y + 11], fill=(255, 255, 255), width=2)
    d.text((x2 + 38, y), f'Entorno ({RING_M / 1000:g} km)', font=f_l, fill=(0, 0, 0))
    if stats:
        y += 30

        def v(k):
            x = stats.get(k)
            return '—' if x is None else f'{float(x):.1f}'
        d.text((margin, y), f"NPV médio — queima: {v('burn_before_npv')} → {v('burn_after_npv')}   ·   "
                            f"entorno: {v('ring_before_npv')} → {v('ring_after_npv')}   (antes → depois, %)",
               font=f_l, fill=(0, 0, 0))

    # triângulo das frações (médias antes → depois)
    if stats:
        g = lambda k: None if stats.get(k) is None else float(stats[k])
        tracks = [
            {'kind': 'ring', 'points': [(None, g('ring_before_npv'), g('ring_before_gv'), g('ring_before_soil')),
                                        (None, g('ring_after_npv'), g('ring_after_gv'), g('ring_after_soil'))]},
            {'kind': 'burn', 'points': [('antes', g('burn_before_npv'), g('burn_before_gv'), g('burn_before_soil')),
                                        ('depois', g('burn_after_npv'), g('burn_after_gv'), g('burn_after_soil'))]},
        ]
    else:
        tracks = []
    tx = margin + 2 * (W + gap)
    d.text((tx, top_h), 'Frações médias (queima ● amarelo, entorno ● branco)', font=_font(14), fill=(0, 0, 0))
    tern = ternary_image(tern_w, tracks)
    fig, d = _grow_to(fig, top_h + 22 + tern.height + 12)
    fig.paste(tern, (tx, top_h + 22))

    map_right = margin + 2 * W + gap
    _scale_bar(d, map_right - 70, top_h + H + 72, bounds, W, f_l)
    _north(d, map_right - 18, top_h + H + 70, 22, f_b)
    return fig


def compose_timeline(p, scenes, bounds, tern):
    """Linha do tempo: cenas mês a mês (t0−k … t0+k) + triângulo com a trajetória mensal."""
    from PIL import Image, ImageDraw
    W, H = scenes[0][1].size
    cols = TIMELINE_COLS
    rows = (len(scenes) + cols - 1) // cols
    gap, margin, top_h, bot_h, lab_h = 12, 24, 96, 70, 46
    tern_w = tern.size[0]
    grid_w = cols * W + (cols - 1) * gap
    fig = Image.new('RGB', (2 * margin + grid_w + 2 * gap + tern_w,
                            top_h + rows * (H + lab_h) + (rows - 1) * gap + bot_h), (255, 255, 255))
    d = ImageDraw.Draw(fig)
    f_t, f_s, f_l, f_b = _font(26, True), _font(17), _font(15), _font(17, True)
    tipo = FIRE_TYPE_NAMES.get(int(p.get('fire_type_code') or 0), p.get('fire_type'))
    d.text((margin, 14), f"{p.get('nome_uc') or '?'} — {tipo} em {p['burn_ymd']} ({p['area_ha']:.0f} ha): mês a mês",
           font=f_t, fill=(0, 0, 0))
    d.text((margin, 54), f'MapBiomas Fogo – Material Combustível (HLS, mosaico mensal {FUEL_METHOD}) · '
                         f'RGB = NPV, GV, Solo · contorno amarelo = queima, branco = entorno {RING_M / 1000:g} km',
           font=f_s, fill=(70, 70, 70))
    for i, (o, im, st) in enumerate(scenes):
        r, c = divmod(i, cols)
        x = margin + c * (W + gap)
        y = top_h + r * (H + lab_h + gap)
        tag = 't0' if o == 0 else f't{o:+d}'
        label = f'{tag} · {_month_label(p["burn_date"], [o])}' + ('  (mês da queima)' if o == 0 else '')
        d.text((x, y), label, font=f_b, fill=(190, 0, 0) if o == 0 else (0, 0, 0))
        fig.paste(im, (x, y + 22))
        d.rectangle([x, y + 22, x + W - 1, y + 22 + H - 1], outline=(190, 0, 0) if o == 0 else (0, 0, 0),
                    width=3 if o == 0 else 1)
        def v(k):
            val = st.get(k)
            return '—' if val is None else f'{float(val):.1f}'
        d.text((x, y + 24 + H), f"NPV queima {v('burn_npv')} · entorno {v('ring_npv')}", font=f_l, fill=(0, 0, 0))
    tx = margin + grid_w + 2 * gap
    d.text((tx, top_h), 'Trajetória mensal das frações médias', font=f_b, fill=(0, 0, 0))
    d.text((tx, top_h + 22), '● amarelo = queima (rótulo = mês rel. a t0) · ● branco = entorno', font=_font(13), fill=(60, 60, 60))
    fig, d = _grow_to(fig, top_h + 44 + tern.height + 12)
    fig.paste(tern, (tx, top_h + 44))
    yb = top_h + rows * (H + lab_h) + (rows - 1) * gap + 18   # logo abaixo da grade
    _scale_bar(d, margin + grid_w - 70, yb, bounds, W, f_l)
    _north(d, margin + grid_w - 18, yb, 20, f_b)
    d.text((margin, yb), f'Composição RGB (preset fuel do Toolkit): NPV {VIS["min"][0]}–{VIS["max"][0]}% · '
                         f'GV {VIS["min"][1]}–{VIS["max"][1]}% · Solo {VIS["min"][2]}–{VIS["max"][2]}%',
           font=f_l, fill=(0, 0, 0))
    return fig


def scene_stats(f, year, offsets):
    """Médias de NPV/GV/Solo na queima e no anel para cada mês t0+offset (uma chamada)."""
    f = ee.Feature(f)
    t0 = ee.Date(f.get('burn_date'))
    geom = f.geometry()
    ring = geom.buffer(RING_M, 10).difference(geom, 10)
    clean = burned_mask(year).unmask(0).Not()
    feats = []
    for o in offsets:
        img = fuel_image(t0.advance(o, 'month'))
        a = img.reduceRegion(ee.Reducer.mean(), geom, SCALE, maxPixels=1e10, tileScale=4)
        b = img.updateMask(clean).reduceRegion(ee.Reducer.mean(), ring, SCALE, maxPixels=1e10, tileScale=4)
        props = {'offset': o}
        for band in BANDS:
            props['burn_' + band] = a.get(band)
            props['ring_' + band] = b.get(band)
        feats.append(ee.Feature(None, props))
    return ee.FeatureCollection(feats)


def make_timelines(sel, year, n, out_dir, offsets=None, dims=360):
    """Uma figura por queima (as n maiores) com as cenas mensais e o triângulo das frações."""
    offsets = offsets or TIMELINE_OFFSETS
    feats = sel.limit(n).getInfo()['features']
    os.makedirs(out_dir, exist_ok=True)

    def render(fi):
        p = fi['properties']
        geom = ee.Geometry(fi['geometry'])
        ring = geom.buffer(RING_M, 10).difference(geom, 10)
        bounds = map_bounds(fi['geometry'], RING_M * 1.5)
        region = ee.Geometry.Rectangle(bounds, None, False)
        t0 = ee.Date(p['burn_date'])
        st = {r['properties']['offset']: r['properties']
              for r in retry(lambda: scene_stats(ee.Feature(fi), year, offsets).getInfo()['features'])}
        scenes = []
        for o in offsets:
            im = retry(lambda: thumb(rgb_with_outline(fuel_image(t0.advance(o, 'month')), geom, ring), region, dims))
            scenes.append((o, im, st.get(o, {})))
        g = lambda o, k: None if st.get(o, {}).get(k) is None else float(st[o][k])
        tracks = [{'kind': 'ring', 'points': [(o, g(o, 'ring_npv'), g(o, 'ring_gv'), g(o, 'ring_soil')) for o in offsets]},
                  {'kind': 'burn', 'points': [(o, g(o, 'burn_npv'), g(o, 'burn_gv'), g(o, 'burn_soil')) for o in offsets]}]
        fig = compose_timeline(p, scenes, bounds, ternary_image(520, tracks))
        uc = ''.join(c if c.isalnum() else '_' for c in (p.get('nome_uc') or 'UC'))
        path = os.path.join(out_dir, f"{year}_{uc}_{p['burn_ymd']}_{short_id(p['orig_id'])}_mensal.png")
        fig.save(path)
        rows = [dict(orig_id=p['orig_id'], nome_uc=p.get('nome_uc'), burn_ymd=p['burn_ymd'], **st[o]) for o in offsets if o in st]
        return path, rows
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(render, feats))
    print(f'Linhas do tempo ({len(res)}):', out_dir)
    return [r for _, rows in res for r in rows]


def make_maps(sel, year, n, out_dir, stats_by_id, dims=640):
    """Um PNG por queima (as n maiores). Renderização no GEE via getThumbURL (equivalente ao ui.Thumbnail)."""
    feats = sel.limit(n).getInfo()['features']
    os.makedirs(out_dir, exist_ok=True)

    def render(fi):
        p = fi['properties']
        geom = ee.Geometry(fi['geometry'])
        ring = geom.buffer(RING_M, 10).difference(geom, 10)
        bounds = map_bounds(fi['geometry'], RING_M * 1.5)
        region = ee.Geometry.Rectangle(bounds, None, False)
        t0 = ee.Date(p['burn_date'])
        b = retry(lambda: thumb(rgb_with_outline(fuel_window(t0, [-m for m in BEFORE_MONTHS]), geom, ring), region, dims))
        a = retry(lambda: thumb(rgb_with_outline(fuel_window(t0, AFTER_MONTHS), geom, ring), region, dims))
        fig = compose_burn_map(p, b, a, bounds, stats_by_id.get(p['orig_id']))
        uc = ''.join(c if c.isalnum() else '_' for c in (p.get('nome_uc') or 'UC'))
        path = os.path.join(out_dir, f"{year}_{uc}_{p['burn_ymd']}_{short_id(p['orig_id'])}.png")
        fig.save(path)
        return path
    with ThreadPoolExecutor(2) as ex:
        paths = list(ex.map(render, feats))
    print(f'Mapas ({len(paths)}):', out_dir)
    return paths


if __name__ == '__main__':
    main()
