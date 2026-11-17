"""Junta i18n/traducoes.csv, i18n/extra.csv e i18n/paises.csv em data/i18n.json,
que o painel carrega para trocar de idioma.

Uso: python build_i18n.py
"""
import csv
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
I18N = ROOT / 'i18n'
LANGS = ['en', 'es', 'fr', 'ind', 'zh']
PH = '{…}'


def rows(name):
    return list(csv.DictReader(open(I18N / name, encoding='utf-8-sig')))


def main():
    t = {}

    def put(pt, tr, override=False):
        if not pt or not any(tr):
            return
        if pt not in t or override:
            t[pt] = tr
        else:  # completa idiomas que faltam
            t[pt] = [a or b for a, b in zip(t[pt], tr)]

    for r in rows('traducoes.csv'):
        tr = [r[l].replace('[REVISAR]', '').rstrip() if r[l].endswith('[REVISAR]') else r[l] for l in LANGS]
        pt = r['pt']
        if PH in pt:
            # trecho de template literal: guarda os pedaços entre as interpolações
            parts = pt.split(PH)
            trs = [x.split(PH) if x else [] for x in tr]
            for i, p in enumerate(parts):
                if re.search(r'[A-Za-zÀ-ÿ]{2,}', p):
                    put(p, [x[i] if len(x) == len(parts) else '' for x in trs])
        else:
            put(pt, tr)
    for r in rows('extra.csv'):
        put(r['pt'], [r[l] for l in LANGS], override=False)

    countries = {r['iso3']: [r['pt']] + [r[l] for l in LANGS] for r in rows('paises.csv')}
    out = {'langs': ['pt'] + LANGS, 't': t, 'countries': countries}
    (ROOT / 'data' / 'i18n.json').write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print('data/i18n.json:', len(t), 'textos ·', len(countries), 'países')


if __name__ == '__main__':
    main()
