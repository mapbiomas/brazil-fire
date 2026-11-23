"""
Biblioteca de exemplos — queimas prescritas/controladas ICMBio × Material Combustível
====================================================================================

Seleciona exemplos estratificados por **bioma** (IBGE, pelo centroide do polígono) e **época**
do ano da queima, e gera para cada um as duas figuras de `fuel_prescribed_burns_stats.py`:
  - mapa ANTES | DEPOIS + triângulo NPV–GV–Solo
  - linha do tempo mês a mês (t0−4 … t0+3) + trajetória no triângulo

Épocas (mês de t0):  chuva = nov–mar · seca_inicial = abr–jun · seca_tardia = jul–out
Seleção por estrato: as maiores queimas (≥ MIN_AREA_HA), no máximo uma por ano e uma por UC,
até N_POR_ESTRATO exemplos — para variar anos e lugares.

Saídas (incrementais; dá para interromper e rodar de novo, estratos prontos são pulados):
  figuras/biblioteca/<bioma>_<época>/*.png
  figuras/biblioteca/README.md           índice com miniaturas e números
  tabelas/biblioteca_exemplos.csv        estatísticas por exemplo (+ bioma, época)
  tabelas/biblioteca_mensal.csv          médias mensais por exemplo

Uso:
  python biblioteca_exemplos.py                 # 2019–2025, até 3 por estrato
  python biblioteca_exemplos.py --n 2 --years 2023 2024 2025
"""
import argparse
import csv
import glob
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fuel_prescribed_burns_stats as fb  # noqa: E402  (inicializa o EE)
import ee  # noqa: E402

BIOMES = os.environ.get('BIOMES_ASSET', '')   # limites dos biomas do IBGE (FeatureCollection com a propriedade 'Bioma')
EPOCAS = {'chuva': (11, 12, 1, 2, 3), 'seca_inicial': (4, 5, 6), 'seca_tardia': (7, 8, 9, 10)}
EPOCA_LABEL = {'chuva': 'chuva (nov–mar)', 'seca_inicial': 'seca inicial (abr–jun)', 'seca_tardia': 'seca tardia (jul–out)'}


def slug(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return ''.join(c if c.isalnum() else '_' for c in s).strip('_')


def epoca(month):
    for k, v in EPOCAS.items():
        if month in v:
            return k
    return '?'


def candidates(years, min_area):
    fc = (fb.load_icmbio(years)
          .filter(ee.Filter.inList('fire_type_code', fb.SELECT_CODES))
          .filter(ee.Filter.notNull(['burn_date']))
          .filter(ee.Filter.gte('area_ha', min_area)))
    pts = fc.map(lambda f: ee.Feature(f.geometry().centroid(10), {
        'orig_id': f.get('orig_id'), 'year': f.get('year'), 'month': f.get('month'),
        'area_ha': f.get('area_ha'), 'nome_uc': f.get('nome_uc'), 'burn_ymd': f.get('burn_ymd'),
        'fire_type_code': f.get('fire_type_code'), 'date_source': f.get('date_source'),
        'img_ymd': f.get('img_ymd')}))
    j = ee.FeatureCollection(ee.Join.saveFirst('b').apply(
        pts, ee.FeatureCollection(BIOMES), ee.Filter.intersects(leftField='.geo', rightField='.geo')))
    j = j.map(lambda f: ee.Feature(None, f.toDictionary(
        ['orig_id', 'year', 'month', 'area_ha', 'nome_uc', 'burn_ymd', 'fire_type_code', 'date_source', 'img_ymd']))
        .set('bioma', ee.Feature(f.get('b')).get('Bioma')))
    cols = ['orig_id', 'year', 'month', 'area_ha', 'nome_uc', 'burn_ymd', 'fire_type_code', 'date_source', 'img_ymd', 'bioma']
    lst = j.reduceColumns(ee.Reducer.toList(len(cols)), cols).get('list').getInfo()
    return [dict(zip(cols, r)) for r in lst]


def pick(rows, n):
    strata = {}
    for r in rows:
        if r.get('bioma') is None or r.get('month') is None:
            continue
        strata.setdefault((r['bioma'], epoca(int(r['month']))), []).append(r)
    chosen = {}
    for k, lst in strata.items():
        lst.sort(key=lambda r: -r['area_ha'])
        sel, years, ucs = [], set(), set()
        for r in lst:                       # 1ª passada: anos e UCs distintos
            if r['year'] in years or r['nome_uc'] in ucs:
                continue
            sel.append(r); years.add(r['year']); ucs.add(r['nome_uc'])
            if len(sel) == n:
                break
        for r in lst:                       # completa se faltou (estratos pequenos)
            if len(sel) == n:
                break
            if r not in sel:
                sel.append(r)
        chosen[k] = sel
    return strata, chosen


def run_stratum(key, items, fig_root):
    bioma, ep = key
    out_dir = os.path.join(fig_root, f'{slug(bioma)}_{ep}')
    ids = [r['orig_id'] for r in items]
    stats_rows, monthly_rows = [], []
    for y in sorted({r['year'] for r in items}):
        y_ids = [r['orig_id'] for r in items if r['year'] == y]
        sel = fb.refine_t0(fb.load_icmbio([y]).filter(ee.Filter.inList('orig_id', y_ids)).sort('area_ha', False))
        n = len(y_ids)
        rows = fb.chunked_getinfo(fb.compute_stats(sel, y), n)
        fb.make_maps(sel, y, n, out_dir, {r['orig_id']: r for r in rows})
        monthly_rows += fb.make_timelines(sel, y, n, out_dir)
        stats_rows += rows
    meta = {r['orig_id']: r for r in items}
    for r in stats_rows + monthly_rows:
        r['bioma'], r['epoca'] = bioma, ep
    for r in stats_rows:
        bb, ba, rb = (None if r.get(k) is None else float(r[k]) for k in ('burn_before_npv', 'burn_after_npv', 'ring_before_npv'))
        r['dNPV_burn'] = None if bb is None or ba is None else ba - bb
        r['NPV_excess_vs_ring'] = None if bb is None or rb is None else bb - rb
        r['fig_dir'] = os.path.basename(out_dir)
        tot, nb = r.get('n_pix_total'), r.get('burn_before_n')
        r['pct_valid_before'] = None if not tot or nb is None else min(100.0, 100.0 * float(nb) / float(tot))
    return stats_rows, monthly_rows, ids


def append_csv(path, rows, cols):
    """Acrescenta linhas; se o arquivo existente tiver outro cabeçalho, reescreve com as colunas atuais."""
    old = read_csv(path)
    if old and list(old[0].keys()) != cols:
        rows = old + rows
        os.remove(path)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerows(rows)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))


def write_index(fig_root, tab_dir, strata, chosen):
    stats = read_csv(os.path.join(tab_dir, 'biblioteca_exemplos.csv'))
    by_id = {r['orig_id']: r for r in stats}
    f = lambda v: '—' if v in (None, '') else f'{float(v):.1f}'
    L = ['# Biblioteca de exemplos — queimas prescritas/controladas ICMBio',
         '',
         'Gerada por `scripts/biblioteca_exemplos.py`. Para cada exemplo: mapa **antes | depois** e '
         '**linha do tempo mês a mês** (t0−4 … t0+3), ambos com o triângulo NPV–GV–Solo.',
         'RGB = NPV, GV, Solo (preset `fuel` do Toolkit). NPV em % (média na queima / no entorno de 1 km).',
         '',
         '**t0 (data da queima)** vem do campo `obs` ("Queima do dia dd/mm") quando existe e cai até 120 dias '
         'antes da data da imagem. Sem data no obs, t0 é **estimado pela série**: entre t_img−4 e t_img, o mês '
         'com a maior queda da soma NPV+GV+Solo no polígono (≥ 5 pontos e ≥ 20%); senão, fica a data da imagem. '
         'A v1 usava sempre a data da '
         'imagem, e em vários casos o "mês antes" já tinha a cicatriz (ver `../biblioteca_v1_data_imagem/`).',
         '',
         '## Estratos (queimas prescritas/controladas ≥ 20 ha, 2019–2025)',
         '',
         '| Bioma | Época | Queimas disponíveis | Exemplos | Status |',
         '|---|---|---|---|---|']
    for k in sorted(chosen):
        done = all(i['orig_id'] in by_id for i in chosen[k])
        L.append(f'| {k[0]} | {EPOCA_LABEL[k[1]]} | {len(strata[k])} | {len(chosen[k])} | {"pronto" if done else "pendente"} |')
    L += ['', '## Resumo por estrato (exemplos prontos)', '',
          '| Bioma | Época | n | NPV antes (queima) | NPV antes (entorno) | NPV depois (queima) | ΔNPV |', '|---|---|---|---|---|---|---|']
    for k in sorted(chosen):
        rs = [by_id[i['orig_id']] for i in chosen[k] if i['orig_id'] in by_id]
        if not rs:
            continue
        def mean(col):
            v = [float(r[col]) for r in rs if r.get(col) not in (None, '')]
            return sum(v) / len(v) if v else None
        L.append(f'| {k[0]} | {EPOCA_LABEL[k[1]]} | {len(rs)} | {f(mean("burn_before_npv"))} | '
                 f'{f(mean("ring_before_npv"))} | {f(mean("burn_after_npv"))} | {f(mean("dNPV_burn"))} |')
    for k in sorted(chosen):
        sub = f'{slug(k[0])}_{k[1]}'
        L += ['', f'## {k[0]} — {EPOCA_LABEL[k[1]]}', '']
        for it in chosen[k]:
            r = by_id.get(it['orig_id'])
            stem_id = fb.short_id(it['orig_id'])
            pngs = sorted(glob.glob(os.path.join(fig_root, sub, f"*_{stem_id}.png")) +
                          glob.glob(os.path.join(fig_root, sub, f"*_{stem_id}_mensal.png")))
            src = (r or it).get('date_source')
            t0 = (r or it).get('burn_ymd')
            fonte = {'obs': f"data do campo obs (imagem {it.get('img_ymd')})", 'data_img': 'data da imagem',
                     'juliano': 'data pelo dia juliano',
                     'serie': f"estimada pela série (maior queda de NPV+GV+Solo; imagem {it.get('img_ymd')})"
                     }.get(src, '?')
            L.append(f"### {it['nome_uc']} · {t0} · {it['area_ha']:.0f} ha · "
                     f"{fb.FIRE_TYPE_NAMES.get(int(it['fire_type_code']), '')}")
            L.append(f'_t0 = {fonte}_')
            L.append('')
            if r:
                L.append(f"NPV queima {f(r.get('burn_before_npv'))} → {f(r.get('burn_after_npv'))} · "
                         f"entorno {f(r.get('ring_before_npv'))} → {f(r.get('ring_after_npv'))} · "
                         f"excesso vs entorno {f(r.get('NPV_excess_vs_ring'))}")
                pv = r.get('pct_valid_before')
                if pv not in (None, '') and float(pv) < 50:
                    L.append('')
                    L.append(f'⚠️ só {float(pv):.0f}% dos pixels da queima têm dado no mês anterior (nuvem): interpretar com cuidado.')
            else:
                L.append('_pendente_')
            L.append('')
            for p in pngs:
                rel = os.path.relpath(p, fig_root).replace(os.sep, '/')
                L.append(f'![{os.path.basename(p)}]({rel})')
                L.append('')
    with open(os.path.join(fig_root, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='+', default=list(range(2019, 2026)))
    ap.add_argument('--n', type=int, default=3, help='exemplos por estrato bioma × época')
    ap.add_argument('--min-area', type=float, default=20)
    ap.add_argument('--redo', action='store_true', help='refaz todos os exemplos (apaga as tabelas da biblioteca)')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    args = ap.parse_args()
    fig_root = os.path.join(args.out, 'figuras', 'biblioteca')
    tab_dir = os.path.join(args.out, 'tabelas')
    os.makedirs(fig_root, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)

    print('Buscando candidatos…', flush=True)
    rows = candidates(args.years, args.min_area)
    strata, chosen = pick(rows, args.n)
    total = sum(len(v) for v in chosen.values())
    print(f'{len(rows)} candidatas, {len(chosen)} estratos, {total} exemplos', flush=True)
    for k in sorted(chosen):
        print(f'  {k[0]:15s} {k[1]:13s} disponíveis={len(strata[k]):5d} exemplos={len(chosen[k])}', flush=True)

    stats_csv = os.path.join(tab_dir, 'biblioteca_exemplos.csv')
    monthly_csv = os.path.join(tab_dir, 'biblioteca_mensal.csv')
    if args.redo:
        for pth in (stats_csv, monthly_csv):
            if os.path.exists(pth):
                os.remove(pth)
    done_ids = {r['orig_id'] for r in read_csv(stats_csv)}
    write_index(fig_root, tab_dir, strata, chosen)
    stat_cols = ['bioma', 'epoca', 'fig_dir', 'orig_id', 'year', 'burn_ymd', 'img_ymd', 'date_source',
                 't0_shift_months', 't0_max_drop', 'month', 'fire_type', 'fire_type_code',
                 'nome_uc', 'cnuc', 'area_ha', 'obs',
                 'burn_before_npv', 'burn_before_gv', 'burn_before_soil', 'burn_before_n',
                 'burn_after_npv', 'burn_after_gv', 'burn_after_soil', 'burn_after_n',
                 'ring_before_npv', 'ring_before_gv', 'ring_before_soil',
                 'ring_after_npv', 'ring_after_gv', 'ring_after_soil',
                 'n_pix_total', 'pct_valid_before', 'dNPV_burn', 'NPV_excess_vs_ring']
    month_cols = ['bioma', 'epoca', 'orig_id', 'nome_uc', 'burn_ymd', 'offset'] + \
                 [f'{z}_{b}' for z in ('burn', 'ring') for b in fb.BANDS]
    # estratos menores primeiro: aparecem resultados de todos os biomas cedo
    for i, k in enumerate(sorted(chosen, key=lambda k: len(strata[k]))):
        todo = [it for it in chosen[k] if it['orig_id'] not in done_ids]
        if not todo:
            print(f'[{i + 1}/{len(chosen)}] {k} já pronto', flush=True)
            continue
        print(f'[{i + 1}/{len(chosen)}] {k[0]} · {k[1]}: {len(todo)} exemplos…', flush=True)
        try:
            srows, mrows, _ = run_stratum(k, todo, fig_root)
        except Exception as e:  # segue para o próximo estrato
            print(f'   ERRO em {k}: {e}', flush=True)
            continue
        append_csv(stats_csv, srows, stat_cols)
        append_csv(monthly_csv, mrows, month_cols)
        done_ids |= {r['orig_id'] for r in srows}
        write_index(fig_root, tab_dir, strata, chosen)
        print(f'   ok ({len(srows)} exemplos) — índice atualizado', flush=True)
    print('FIM. Índice:', os.path.join(fig_root, 'README.md'), flush=True)


if __name__ == '__main__':
    main()
