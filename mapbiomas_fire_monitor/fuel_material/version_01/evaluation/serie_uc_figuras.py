"""
Figuras da série de combustível na escala da UC (lê as tabelas de `serie_uc.py`; não usa o GEE)
================================================================================================

Saídas em figuras/serie_uc/:
  <uc>_serie_produto.png       NPV mensal 2019–2025 (campo + savana) por tempo desde o último fogo
                               + área queimada mensal (ICMBio por tipo)
  <uc>_climatologia.png        NPV por mês do ano, por tempo desde o último fogo (mediana 2019–2025)
  <uc>_landsat_vs_produto.png  comparação pixel a pixel: algoritmo aplicado ao Landsat × produto (HLS)
  <uc>_serie_landsat.png       NPV do pico da seca (ago–set), 1985–2025, Landsat, + área queimada anual

Uso:  python serie_uc_figuras.py [--uc emas]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, '..', 'tabelas')
FIG = os.path.join(HERE, '..', 'figuras', 'serie_uc')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
# ordinal (azul, passos 250 → 700): mais escuro = mais tempo sem fogo
TSF_COL = {1: '#86b6ef', 2: '#3987e5', 3: '#1c5cab', 4: '#0d366b'}
TSF_NOME = {1: '< 1 ano', 2: '1–3 anos', 3: '3–9 anos', 4: '≥ 9 anos ou sem registro'}
# área queimada: categóricas em ordem fixa
TIPO_COL = {'Queima prescrita/controlada': '#2a78d6', 'Gestão de ignição': '#1baf7a',
            'Incêndio': '#eb6834', 'Outros / sem classe': '#b4b2ab'}
COBS = ['campo', 'savana']
MESES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']


def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def weighted(d, by):
    d = d.copy()
    d['w'] = d.npv * d.n
    g = d.groupby(by)[['w', 'n']].sum()
    g['npv'] = g.w / g.n
    return g


def tipo(code):
    code = int(code) if str(code) not in ('', 'nan') else 0
    return {1: 'Queima prescrita/controlada', 2: 'Queima prescrita/controlada', 7: 'Gestão de ignição',
            4: 'Incêndio'}.get(code, 'Outros / sem classe')


def fig_series(uc, path):
    d = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_produto.csv'))
    d = d[d.cob.isin(COBS)]
    g = weighted(d, ['year', 'month', 'tsfc']).reset_index()
    g['date'] = pd.to_datetime(dict(year=g.year, month=g.month, day=15))
    q = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_queimadas.csv'))
    q = q[(q.fonte == 'ICMBio') & (q.year >= 2019)].copy()
    q['tipo'] = q.fire_type_code.map(tipo)
    qb = q.groupby(['year', 'month', 'tipo']).area_ha.sum().unstack(fill_value=0).reindex(columns=list(TIPO_COL), fill_value=0)
    qb.index = pd.to_datetime([f'{y}-{m:02d}-15' for y, m in qb.index])

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for k in (1, 2, 3, 4):
        s = g[(g.tsfc == k) & (g.n >= 200)].set_index('date').npv.sort_index()
        s = s.reindex(pd.date_range('2019-01-15', '2025-12-15', freq='MS') + pd.Timedelta(days=14))
        a1.plot(s.index, s.values, color=TSF_COL[k], lw=2, label=TSF_NOME[k])
    a1.set_ylabel('NPV médio (%)', fontsize=9, color=INK2)
    h, l = a1.get_legend_handles_labels()
    a1.legend(h, [f'{x}' for x in l], fontsize=8.5, frameon=False, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0),
              title='Tempo desde o último fogo:', title_fontsize=8.5, alignment='left')
    style(a1)
    # área queimada ANUAL por fonte, barras agrupadas centradas no meio de cada ano
    anos = list(range(2019, 2026))
    fic = os.path.join(TAB, f'serie_uc_{uc}_icmbio_anual.csv')
    if os.path.exists(fic):   # união dos polígonos (sem contar sobreposição), comparável ao satélite
        icm = pd.read_csv(fic).set_index('year').icmbio_uniao_ha
    else:
        icm = q.groupby('year').area_ha.sum()
    qa = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_queimadas.csv'))
    mbf = qa[qa.fonte == 'MapBiomas'].groupby('year').area_ha.sum()
    mon = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_monitor_anual.csv')).set_index('year').monitor_ha
    fontes = [('ICMBio (polígonos de todos os tipos, sem sobreposição)', icm, '#eb6834'),
              ('MapBiomas Fogo, Coleção 5.1', mbf, '#1baf7a'),
              ('Monitor do Fogo (MapBiomas)', mon, '#eda100')]
    larg = 95   # dias
    for i, (lab, s, c) in enumerate(fontes):
        xs = [pd.Timestamp(f'{y}-07-02') + pd.Timedelta(days=(i - 1) * (larg + 8)) for y in anos]
        vs = [s.get(y, np.nan) / 1000 for y in anos]
        a2.bar(xs, vs, width=larg, color=c, label=lab, edgecolor='white', lw=1.0)
        for x, v in zip(xs, vs):
            if v == v:
                a2.text(x, v + 0.8, f'{v:.0f}', ha='center', va='bottom', fontsize=7, color=INK2)
    a2.set_ylim(0, max(np.nanmax([s.get(y, np.nan) for y in anos]) for _, s, _ in fontes) / 1000 * 1.22)
    a2.set_ylabel('Área queimada\nno ano (mil ha)', fontsize=9, color=INK2)
    a2.legend(fontsize=8, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.25))
    a2.xaxis.set_major_locator(mdates.YearLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    style(a2)
    fig.suptitle('PARNA das Emas: NPV mensal em campo e savana por tempo desde o último fogo (produto, 2019–2025)',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2)
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    clim = g[g.n >= 200].groupby(['month', 'tsfc']).npv.median().unstack()
    return clim


def fig_clim(clim, uc, path):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(1, 13)
    for k in (1, 2, 3, 4):
        if k in clim:
            ax.plot(x, clim[k].reindex(x).values, color=TSF_COL[k], lw=2.2, marker='o', ms=5, mec='white', mew=1.5,
                    label=TSF_NOME[k])
            v = clim[k].reindex(x).values
            ax.text(12.25, v[-1], TSF_NOME[k], va='center', fontsize=8, color=INK2)
    ax.set_xticks(x, MESES)
    ax.set_xlim(0.6, 14.6)
    ax.set_ylabel('NPV médio (%)', fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc='upper left')
    style(ax)
    fig.suptitle('PARNA das Emas: ciclo anual do NPV (campo e savana), mediana 2019–2025',
                 fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)


def fig_compare(uc, path):
    d = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_comparacao_landsat.csv')).dropna(subset=['p_npv', 'npv_u'])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    stats = {}
    for ax, b, nome in zip(axes, ['npv', 'gv', 'soil'], ['NPV', 'GV', 'Solo']):
        x, y = d['p_' + b], d[b + '_u']
        lim = np.nanpercentile(np.r_[x, y], 99.5)
        ax.hexbin(x, y, gridsize=45, extent=(0, lim, 0, lim), cmap='Blues', mincnt=1, bins='log', linewidths=0)
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls='--')
        r = np.corrcoef(x, y)[0, 1]
        bias, mae = np.mean(y - x), np.mean(np.abs(y - x))
        stats[b] = (r, bias, mae)
        ax.text(0.04, 0.96, f'r = {r:.2f}\nviés = {bias:+.1f} pp\nEAM = {mae:.1f} pp'.replace('-', '−'),
                transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
        ax.set_title(nome, fontsize=10, color=INK, loc='left')
        ax.set_xlabel('Produto (HLS), %', fontsize=8.5, color=INK2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        style(ax, grid='both')
    axes[0].set_ylabel('Algoritmo aplicado ao Landsat 8/9, %', fontsize=8.5, color=INK2)
    meses = ', '.join(f'{m:02d}/{y}' for y, m in sorted(set(zip(d.year, d.month))))
    fig.suptitle('PARNA das Emas: produto (HLS) × o mesmo algoritmo aplicado ao Landsat 8/9, pixel a pixel',
                 fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.885, f'{len(d):,} pixels de 30 m; meses {meses}'.replace(',', '.', 1), fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    return stats


def fig_landsat(uc, path):
    f = os.path.join(TAB, f'serie_uc_{uc}_landsat_1985_2025.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[d.cob.isin(COBS) & d.month.isin([8, 9])]
    g = weighted(d, ['year', 'tsfc']).reset_index()
    tot = weighted(d, ['year']).npv
    p = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_produto.csv'))
    p = p[p.cob.isin(COBS) & p.month.isin([8, 9])]
    ptot = weighted(p, ['year']).npv
    q = pd.read_csv(os.path.join(TAB, f'serie_uc_{uc}_queimadas.csv'))
    mb = q[q.fonte == 'MapBiomas'].groupby('year').area_ha.sum()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]})
    for k in (1, 2, 3, 4):
        s = g[(g.tsfc == k) & (g.n >= 500)].set_index('year').npv.reindex(range(1985, 2026))
        a1.plot(s.index, s.values, color=TSF_COL[k], lw=1.6, marker='o', ms=3.5, label=TSF_NOME[k])
    a1.plot(tot.index, tot.values, color=MUTED, lw=1.2, ls=':', label='Todos (Landsat)')
    a1.plot(ptot.index, ptot.values, color=INK, lw=0, marker='D', ms=5, label='Todos (produto HLS)')
    a1.set_ylabel('NPV médio ago–set (%)', fontsize=9, color=INK2)
    a1.legend(fontsize=8, frameon=False, ncol=3, loc='upper left', bbox_to_anchor=(0, 1.2))
    style(a1)
    a2.bar(mb.index, mb.values / 1000, color='#eb6834', width=0.75)
    a2.set_ylabel('Área queimada\nMapBiomas (mil ha)', fontsize=9, color=INK2)
    style(a2)
    fig.suptitle('PARNA das Emas: NPV no pico da seca, 1985–2025 (algoritmo aplicado ao Landsat), campo e savana',
                 fontsize=10.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2)
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    return tot


# Zupo et al. (2022, Austral Ecology 47:1101–1112), campo sujo, parcelas com 1, 2 e 4 anos sem fogo (kg m-2)
ZUPO_MORTO = {'1': 0.31, '2': 0.69, '4–8': 0.86}
ZUPO_TOTAL = {'1': 0.54, '2': 0.86, '4–8': 1.06}
FINAS = ['<1', '1', '2', '3', '4–8', '≥9 ou sem registro']


def curva_finas(uc, fonte):
    """Razão NPV(classe)/NPV(2 anos) em campo+savana, ago–set; mediana e IQR entre os meses."""
    f = os.path.join(TAB, f'serie_uc_{uc}_seca_finas_{fonte}.csv')
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[d.cob.isin(COBS)]
    g = weighted(d, ['year', 'month', 'tsfc']).reset_index()
    g = g[g.n >= 200]
    t = g.pivot_table(index=['year', 'month'], columns='tsfc', values='npv')
    r = t.div(t[3], axis=0)                  # tsfc 3 = 2 anos completos
    r.columns = [FINAS[int(c) - 1] for c in r.columns]
    return pd.DataFrame({'med': r.median(), 'q1': r.quantile(.25), 'q3': r.quantile(.75), 'n': r.count(),
                         'npv_med': t.median().values})


def fig_literatura(uc, path):
    prod, ls = curva_finas(uc, 'produto'), curva_finas(uc, 'landsat')
    if prod is None:
        return None
    x = {c: i for i, c in enumerate(FINAS)}
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    series = [('Produto (HLS), PARNA das Emas, ago–set 2019–2025', prod, '#2a78d6', 'o', -0.12),
              ('Mesmo algoritmo no Landsat, PARNA das Emas, ago–set 1995–2018', ls, '#1baf7a', 's', 0.12)]
    for lab, s, c, mk, dx in series:
        if s is None:
            continue
        s = s.reindex(FINAS).dropna(subset=['med'])
        xs = np.array([x[k] for k in s.index]) + dx
        ax.vlines(xs, s.q1, s.q3, color=c, lw=2, alpha=0.45)
        ax.plot(xs, s.med, color=c, lw=1.8, marker=mk, ms=8, mec='white', mew=1.8, label=lab)
    zx = [x[k] for k in ZUPO_MORTO]
    zy = [v / ZUPO_MORTO['2'] for v in ZUPO_MORTO.values()]
    ax.plot(zx, zy, color='#eb6834', lw=1.8, ls='--', marker='D', ms=8, mec='white', mew=1.8,
            label='Combustível morto medido em campo (Zupo et al., 2022), campo sujo')
    # Andrade (1998) e Neto et al. (1998), citados por Oliveras et al. (2013): biomassa herbácea de campo sujo
    # recuperada em 80% após 1 ano e totalmente após 2 anos
    ax.plot([x['1'], x['2']], [0.8, 1.0], color=INK2, lw=0, marker='^', ms=9, mfc='white', mew=1.8,
            label='Biomassa herbácea recuperada em campo sujo (Andrade, 1998; Neto et al., 1998)')
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(FINAS)), ['< 1', '1', '2', '3', '4 a 8', '≥ 9 ou\nsem registro'])
    ax.set_xlabel('Anos completos desde o último fogo', fontsize=9, color=INK2)
    ax.set_ylabel('Valor relativo ao de 2 anos sem fogo', fontsize=9, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    style(ax)
    fig.suptitle('Recuperação do combustível fino após o fogo: produto e medida de campo',
                 fontsize=11, color=INK, x=0.01, ha='left')
    fig.text(0.01, 0.865, 'NPV de campos e savanas (barras: intervalo interquartil entre os meses) e medidas de campo,\n'
             'todos divididos pelo valor com 2 anos sem fogo.', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(path, dpi=200, facecolor='white')
    plt.close(fig)
    out = pd.concat({'produto': prod, 'landsat': ls if ls is not None else pd.DataFrame()}, axis=0)
    out.round(3).to_csv(os.path.join(TAB, f'serie_uc_{uc}_curva_vs_literatura.csv'), encoding='utf-8')
    return prod, ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uc', default='emas')
    uc = ap.parse_args().uc
    os.makedirs(FIG, exist_ok=True)
    clim = fig_series(uc, os.path.join(FIG, f'{uc}_serie_produto.png'))
    fig_clim(clim, uc, os.path.join(FIG, f'{uc}_climatologia.png'))
    clim.round(2).to_csv(os.path.join(TAB, f'serie_uc_{uc}_climatologia.csv'), encoding='utf-8')
    print('climatologia (mediana NPV por mês × classe):\n', clim.round(1))
    print('comparação Landsat × produto (r, viés, EAM):', {k: tuple(round(x, 2) for x in v)
                                                         for k, v in fig_compare(uc, os.path.join(FIG, f'{uc}_landsat_vs_produto.png')).items()})
    tot = fig_landsat(uc, os.path.join(FIG, f'{uc}_serie_landsat.png'))
    print('série Landsat:', 'pendente' if tot is None else tot.round(1).to_dict())
    lit = fig_literatura(uc, os.path.join(FIG, f'{uc}_curva_vs_literatura.png'))
    if lit:
        print('curva vs literatura (produto):\n', lit[0].round(2), '\n(landsat):\n', None if lit[1] is None else lit[1].round(2))


if __name__ == '__main__':
    main()
