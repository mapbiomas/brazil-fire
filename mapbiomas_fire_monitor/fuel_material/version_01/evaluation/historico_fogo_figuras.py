"""
Tabelas e figuras do histórico de fogo (lê a saída bruta de `historico_fogo.py`; não usa o GEE)
================================================================================================

Entradas:  tabelas/historico_fogo_exemplos_bruto.jsonl, tabelas/biblioteca_exemplos.csv
Saídas:
  tabelas/historico_fogo_exemplos.csv       uma linha por exemplo (queima × entorno)
  tabelas/historico_fogo_npv_cob_tsf.csv    NPV t0−1 por exemplo × cobertura × classe de tempo desde o último fogo
  figuras/historico_fogo/npv_por_tempo_desde_fogo.png   curva dentro da mesma paisagem, por cobertura
  figuras/historico_fogo/excesso_entorno_cobertura.png  excesso de NPV: entorno inteiro × mesma cobertura
  figuras/historico_fogo/deteccao_mapbiomas.png         % da queima detectada pelo MapBiomas no ano

Uso:  python historico_fogo_figuras.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tabelas')
FIG = os.path.join(HERE, '..', 'figuras', 'historico_fogo')
SEM = 99

# paleta de referência (dataviz): categóricas em ordem fixa; ordinal em azul (passos 250→700)
C_QUEIMA, C_ENTORNO = '#eb6834', '#2a78d6'
C_COB = {'campo': '#2a78d6', 'savana': '#eb6834', 'campo alagado': '#1baf7a', 'floresta': '#eda100'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
COB_MB = {12: 'campo', 4: 'savana', 11: 'campo alagado', 3: 'floresta', 15: 'pastagem', 9: 'silvicultura',
          33: 'água', 6: 'floresta alagável', 29: 'afloramento rochoso'}
CLASSES = ['1', '2–3', '4–9', '≥10 ou sem registro']
MIN_PIX = 50


def tsf_class(t):
    t = int(t)
    return CLASSES[0] if t == 1 else CLASSES[1] if t <= 3 else CLASSES[2] if t <= 9 else CLASSES[3]


def hist_median(h):
    if not h:
        return np.nan
    k = sorted((float(a), w) for a, w in h.items())
    c = np.cumsum([w for _, w in k])
    return k[int(np.searchsorted(c, c[-1] / 2))][0]


def hist_pct(h, f):
    if not h:
        return np.nan
    tot = sum(h.values())
    return 100 * sum(w for a, w in h.items() if f(float(a))) / tot


def hist_mean(h):
    if not h:
        return np.nan
    tot = sum(h.values())
    return sum(float(a) * w for a, w in h.items()) / tot


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load():
    raw = [json.loads(l) for l in open(os.path.join(TAB, 'historico_fogo_exemplos_bruto.jsonl'), encoding='utf-8')]
    lib = pd.read_csv(os.path.join(TAB, 'biblioteca_exemplos.csv'), encoding='utf-8').set_index('orig_id')
    return raw, lib


def table_examples(raw, lib):
    rows = []
    for r in raw:
        L = lib.loc[r['orig_id']]
        row = {'bioma': L['bioma'], 'epoca': L['epoca'], 'nome_uc': r['nome_uc'], 'burn_ymd': r['burn_ymd'],
               'area_ha': r['area_ha'], 'orig_id': r['orig_id'],
               'npv_queima_t0m1': L['burn_before_npv'], 'npv_entorno_t0m1': L['ring_before_npv'],
               'excesso_entorno': L['NPV_excess_vs_ring'],
               'cob_dominante_queima': COB_MB.get(int(r['queima_lulc_moda']), str(r['queima_lulc_moda'])),
               'entorno_pct_mesma_cob': r.get('entorno_mesma_cob_pct'),
               'npv_entorno_mesma_cob_t0m1': r.get('entorno_mesma_cob_npv_t0m1'),
               'queima_det_mapbiomas_pct': r.get('queima_det_pct')}
        row['excesso_entorno_mesma_cob'] = (L['burn_before_npv'] - row['npv_entorno_mesma_cob_t0m1']
                                            if row['npv_entorno_mesma_cob_t0m1'] is not None else np.nan)
        for z, lab in (('queima_', 'queima'), ('entorno_', 'entorno')):
            h = r.get(z + 'tsf_hist') or {}
            row[f'{lab}_tsf_mediana'] = hist_median(h)
            row[f'{lab}_pct_tsf_1'] = hist_pct(h, lambda a: a == 1)
            row[f'{lab}_pct_tsf_2_3'] = hist_pct(h, lambda a: 2 <= a <= 3)
            row[f'{lab}_pct_tsf_4_9'] = hist_pct(h, lambda a: 4 <= a <= 9)
            row[f'{lab}_pct_tsf_10mais'] = hist_pct(h, lambda a: 10 <= a < SEM)
            row[f'{lab}_pct_sem_registro'] = hist_pct(h, lambda a: a == SEM)
            row[f'{lab}_intervalo_retorno_mediana'] = r.get(z + 'imr_med')
            row[f'{lab}_freq_10anos_media'] = hist_mean(r.get(z + 'f10_hist'))
        rows.append(row)
    d = pd.DataFrame(rows)
    d['tsf_mediana_dif'] = d['queima_tsf_mediana'] - d['entorno_tsf_mediana']
    d.to_csv(os.path.join(TAB, 'historico_fogo_exemplos.csv'), index=False, encoding='utf-8', float_format='%.2f')
    return d


def table_curve(raw):
    acc = []
    for r in raw:
        for z in ('queima_', 'entorno_'):
            for g in (r.get(z + 'npv_por_cob_tsf') or []):
                if 'mean' not in g:
                    continue
                k = int(g['k'])
                acc.append({'orig_id': r['orig_id'], 'nome_uc': r['nome_uc'], 'zona': z[:-1],
                            'cob': COB_MB.get(k // 100, str(k // 100)), 'tsf': k % 100,
                            'classe': tsf_class(k % 100), 'npv': g['mean'], 'n': g['count']})
    a = pd.DataFrame(acc)
    a.to_csv(os.path.join(TAB, 'historico_fogo_npv_cob_tsf.csv'), index=False, encoding='utf-8', float_format='%.3f')
    return a


def curve_by_site(a, cob):
    b = a[a.cob == cob].copy()
    b['w'] = b.npv * b.n
    s = b.groupby(['orig_id', 'classe'])[['w', 'n']].sum()
    s['npv'] = s.w / s.n
    s = s.reset_index()
    s = s[s.n >= MIN_PIX]
    s = s[s.groupby('orig_id').classe.transform('nunique') >= 2]      # só paisagens com ≥ 2 classes
    s['anom'] = s.npv - s.groupby('orig_id').npv.transform('mean')
    return s


def fig_curve(a, path):
    cobs = ['campo', 'savana', 'campo alagado', 'floresta']
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    x = np.arange(len(CLASSES))
    summary = []
    for ax, cob in zip(axes, cobs):
        s = curve_by_site(a, cob)
        for _, g in s.groupby('orig_id'):
            g = g.set_index('classe').reindex(CLASSES)
            ax.plot(x, g.anom.values, color=MUTED, lw=0.8, alpha=0.55, marker='o', ms=2.5, zorder=1)
        med = s.groupby('classe').anom.median().reindex(CLASSES)
        nn = s.groupby('classe').orig_id.nunique().reindex(CLASSES).fillna(0).astype(int)
        ax.plot(x, med.values, color=C_COB[cob], lw=2.4, marker='o', ms=8, mec='white', mew=2, zorder=3)
        for i, m in enumerate(med.values):
            if not np.isnan(m):
                ax.annotate(('0.0' if abs(m) < 0.05 else f'{m:+.1f}').replace('-', '−'), (i, m), xytext=(0, 10), textcoords='offset points',
                            ha='center', fontsize=9, color=INK, fontweight='bold')
        ax.axhline(0, color=MUTED, lw=0.8)
        ticks = ['1', '2–3', '4–9', '≥10 ou\nsem reg.']
        ax.set_xticks(x, [f'{t}\n(n={n})' for t, n in zip(ticks, nn.values)], fontsize=8.5)
        ax.set_title(f'{cob.capitalize()}  ({s.orig_id.nunique()} paisagens)', fontsize=10.5, color=INK, loc='left')
        style(ax)
        summary.append((cob, med.to_dict(), nn.to_dict()))
    axes[0].set_ylabel('NPV em t0−1 menos a média\nda paisagem (pontos percentuais)', fontsize=9, color=INK2)
    fig.supxlabel('Anos desde o último fogo (MapBiomas Fogo col. 5.1)', fontsize=9.5, color=INK2, y=0.0)
    fig.suptitle('NPV antes da queima por tempo desde o último fogo, dentro da mesma paisagem e cobertura',
                 fontsize=11.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.875, 'Linhas finas: cada exemplo (queima + entorno de 1 km). Linha cheia: mediana. '
             f'Classes com ≥ {MIN_PIX} pixels. Outras coberturas têm 6 paisagens ou menos e não são mostradas.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86), h_pad=0)
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    return summary


def fig_excess(d, path):
    d = d.sort_values('excesso_entorno').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    y = np.arange(len(d))
    for i, r in d.iterrows():
        ax.plot([r.excesso_entorno, r.excesso_entorno_mesma_cob], [i, i], color=GRID, lw=2.2, zorder=1)
    ax.scatter(d.excesso_entorno, y, s=46, color=C_ENTORNO, edgecolor='white', lw=1.5, zorder=3,
               label=f'Entorno inteiro (mediana {d.excesso_entorno.median():+.1f} pp)')
    ax.scatter(d.excesso_entorno_mesma_cob, y, s=46, color=C_QUEIMA, edgecolor='white', lw=1.5, zorder=3,
               marker='D', label=f'Entorno da mesma cobertura (mediana {d.excesso_entorno_mesma_cob.median():+.1f} pp)')
    ax.axvline(0, color=MUTED, lw=1)
    labels = [f"{r.nome_uc} · {str(r.burn_ymd)[:7]}" for _, r in d.iterrows()]
    ax.set_yticks(y, labels, fontsize=7.8)
    ax.set_xlabel('Excesso de NPV da queima sobre o entorno, mês t0−1 (pontos percentuais)', fontsize=9, color=INK2)
    fig.suptitle('Excesso de NPV antes da queima: entorno inteiro × entorno da mesma cobertura',
                 fontsize=11, color=INK, x=0.01, ha='left')
    style(ax)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.35, 1.0), ncol=1, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)


def fig_detection(d, path):
    d = d.sort_values('queima_det_mapbiomas_pct').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    y = np.arange(len(d))
    ax.barh(y, d.queima_det_mapbiomas_pct, color=C_ENTORNO, height=0.62)
    for i, v in enumerate(d.queima_det_mapbiomas_pct):
        ax.text(v + 1, i, f'{v:.0f}%', va='center', fontsize=7.5, color=INK2)
    ax.set_yticks(y, [f"{r.nome_uc} · {str(r.burn_ymd)[:7]}" for _, r in d.iterrows()], fontsize=7.8)
    ax.set_xlim(0, 108)
    ax.set_xlabel('% da queima ICMBio marcada como queimada pelo MapBiomas Fogo no ano', fontsize=9, color=INK2)
    fig.suptitle(f'Detecção das queimas prescritas pelo MapBiomas Fogo (mediana {d.queima_det_mapbiomas_pct.median():.0f}%)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    style(ax)
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    raw, lib = load()
    d = table_examples(raw, lib)
    a = table_curve(raw)
    summary = fig_curve(a, os.path.join(FIG, 'npv_por_tempo_desde_fogo.png'))
    fig_excess(d, os.path.join(FIG, 'excesso_entorno_cobertura.png'))
    fig_detection(d, os.path.join(FIG, 'deteccao_mapbiomas.png'))
    print('exemplos:', len(d))
    print(f"excesso: entorno inteiro mediana {d.excesso_entorno.median():+.2f} ({(d.excesso_entorno > 0).sum()}/{len(d)} > 0)"
          f" · mesma cobertura {d.excesso_entorno_mesma_cob.median():+.2f} ({(d.excesso_entorno_mesma_cob > 0).sum()}/{len(d)} > 0)")
    print(f"tempo desde o último fogo (mediana): queima=entorno {(d.tsf_mediana_dif == 0).sum()}, "
          f"queima>entorno {(d.tsf_mediana_dif > 0).sum()}, queima<entorno {(d.tsf_mediana_dif < 0).sum()}")
    print(f"detecção MapBiomas: mediana {d.queima_det_mapbiomas_pct.median():.0f}%, zero em {(d.queima_det_mapbiomas_pct < 1).sum()}")
    for cob, med, nn in summary:
        print(cob, {k: (round(v, 2), nn[k]) for k, v in med.items()})


if __name__ == '__main__':
    main()
