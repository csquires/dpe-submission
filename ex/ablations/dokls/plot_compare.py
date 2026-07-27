"""
dokls two-leg vs model_selection direct comparison plots.

color = method family; source + loss-variant = linestyle + marker:
  MS direct        solid  + circle 'o'
  dokls base       dashed + square 's'
  dokls _DV        dotted + star   '*'
  dokls _NWJ       dotted + X      'X'

regret is computed AT PLOT TIME, per (panel, x-point): normalized 0=best..1=worst
over exactly the series present at that point (dokls + MS together), from their
eldr_err. so the pool is dynamic across the sweep.

2x7 grid: p* rows {q0,q1} x KL cols. metrics: eldr_err (abs), pointwise MAE,
regret. all 10 dokls methods + MS on the 6 shared.

PROTOTYPE mode (--x N): dokls over the diagonal N, MS at the single N=8192 cell,
to preview styling before the decoupled Nstar cells land. FINAL mode (--x nstar,
after backfill): dokls/MS over Nstar at fixed N=8192.
"""
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from ex.utils.plot_style import apply as apply_style, style_for
from ex.ablations.dokls.variants import resolve
from ex.ablations.dokls.step4_plot_results import dokls_style
from ex.ablations.dokls.plot_vs_N import load_dokls, load_ms, MS_COMMON, MS_TESTIDX

PSTARS = [0, 1]
PNAME = {0: 'q_0', 1: 'q_1'}
NVALS = [1024, 2048, 4096, 8192]
MS_CELL_N = 8192


def _family_color(method):
    base = method
    for suf in ('_NWJ', '_DV'):
        if method.endswith(suf):
            base = method[: -len(suf)]
    if base == 'MHT':
        base = 'MultiHeadTDRE'
    return style_for(base)['color']


def comp_style(method, source):
    """color=family; MS direct = solid + circle; dokls uses the canonical
    dokls_style (base dashed+square, DV dotted+star, NWJ dotted+X)."""
    if source == 'ms':
        return dict(color=_family_color(method), linestyle='-', marker='o')
    return dict(dokls_style(method))


def regret_over_contenders(err_by_series):
    """err_by_series: {(source,method): value}. -> {(source,method): regret}."""
    vals = np.array([v for v in err_by_series.values() if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {k: np.nan for k in err_by_series}
    lo, hi = vals.min(), vals.max()
    span = hi - lo
    out = {}
    for k, v in err_by_series.items():
        if not np.isfinite(v):
            out[k] = np.nan
        elif span <= 0:
            out[k] = 0.0
        else:
            out[k] = (v - lo) / span
    return out


def plot_compare(kl, dokls_err, ms_err, dokls_val, ms_val, xvals, *,
                 metric, ylabel, out_dir, prefix, yscale):
    """one 2x7 figure. metric in {eldr_err, mae, regret}. for regret, dokls_val/
    ms_val are ignored and regret is derived from dokls_err/ms_err per point.
    dokls_err/dokls_val: {(pstar, xkey): {method: (arr(n_kl,), se)}}.
    ms_err/ms_val: {pstar: {method: (n_kl,)}} at the MS cell (single xkey MS_CELL_N).
    """
    apply_style()
    nkl = len(kl)
    x = np.array(xvals, dtype=float)
    dokls_methods = sorted({m for md in dokls_err.values() for m in md})
    fig, axes = plt.subplots(2, nkl, figsize=(2.7 * nkl, 6.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for j in range(nkl):
            ax = axes[i][j]
            # assemble per-(method,source) traces over x
            for m in dokls_methods:
                ys = []
                for xk in xvals:
                    md = dokls_err.get((pstar, xk), {}) if metric == 'regret' \
                        else dokls_val.get((pstar, xk), {})
                    if metric == 'regret':
                        # regret computed below; collect err here for the pool
                        ys.append(dokls_err.get((pstar, xk), {}).get(m, (np.full(nkl, np.nan),))[0][j]
                                  if m in dokls_err.get((pstar, xk), {}) else np.nan)
                    else:
                        ys.append(md[m][0][j] if m in md else np.nan)
                ys = np.array(ys)
                if metric == 'regret':
                    ys = _regret_series(dokls_err, ms_err, pstar, j, xvals, ('dokls', m), nkl)
                if np.isfinite(ys).any():
                    ax.plot(x, ys, **comp_style(m, 'dokls'), linewidth=0.85,
                            markersize=4, alpha=0.8)
            # MS at its single cell
            for m in MS_COMMON:
                if m not in ms_err.get(pstar, {}):
                    continue
                if metric == 'regret':
                    yv = _regret_point(dokls_err, ms_err, pstar, j, MS_CELL_N, ('ms', m))
                else:
                    yv = ms_val[pstar][m][j]
                ax.plot([MS_CELL_N], [yv], **comp_style(m, 'ms'), linewidth=0,
                        markersize=7, markerfacecolor='none', markeredgewidth=1.5)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xticks(xvals)
            ax.set_xticklabels([f'{v // 1024}k' for v in xvals], fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'KL$={kl[j]:g}$', fontsize=10)
            if i == 1:
                ax.set_xlabel(r'$N$ (=$N_*$ on diagonal)')
            if j == 0:
                ax.set_ylabel(rf'$p_*={PNAME[pstar]}$' + '\n' + ylabel)
    _legend(fig, dokls_methods)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')


def _pool_at(dokls_err, ms_err, pstar, j, xk):
    """{(source,method): err} present at (pstar, KL=j, x=xk)."""
    pool = {}
    for m, (arr, _se) in dokls_err.get((pstar, xk), {}).items():
        pool[('dokls', m)] = arr[j]
    if xk == MS_CELL_N:
        for m, arr in ms_err.get(pstar, {}).items():
            pool[('ms', m)] = arr[j]
    return pool


def _regret_point(dokls_err, ms_err, pstar, j, xk, key):
    return regret_over_contenders(_pool_at(dokls_err, ms_err, pstar, j, xk)).get(key, np.nan)


def _regret_series(dokls_err, ms_err, pstar, j, xvals, key, nkl):
    return np.array([_regret_point(dokls_err, ms_err, pstar, j, xk, key) for xk in xvals])


def _legend(fig, dokls_methods):
    handles = []
    labels = []
    for m in dokls_methods:
        handles.append(Line2D([0], [0], **comp_style(m, 'dokls'), markersize=5))
        labels.append(f'{m} (2-leg)')
    src = [
        Line2D([0], [0], color='0.35', linestyle='-', marker='o',
               markerfacecolor='none', label='MS direct'),
    ]
    leg1 = fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
                      framealpha=0.9, bbox_to_anchor=(0.5, 0.045))
    fig.add_artist(leg1)
    fig.legend(handles=src, loc='lower center', ncol=1, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))


def main(variant=None):
    _tag, _route, config = resolve(variant)
    processed_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    kl = np.array(config['kl_distances'], dtype=float)
    ms_h5 = os.path.join(os.path.dirname(__file__), 'ms_ref', 'tr8192_te8192.h5')

    eldr = load_dokls(processed_dir, 'eldr_err_mean', 'eldr_err_se')
    mae = load_dokls(processed_dir, 'mae', None)
    ms_eldr = load_ms(ms_h5, 'eldr_err')
    ms_mae = load_ms(ms_h5, 'mae')

    plot_compare(kl, eldr, ms_eldr, eldr, ms_eldr, NVALS, metric='eldr_err',
                 ylabel='ELDR error (abs)', out_dir=figures_dir,
                 prefix='dokls_cmp_eldr_err', yscale='log')
    plot_compare(kl, eldr, ms_eldr, mae, ms_mae, NVALS, metric='mae',
                 ylabel='Pointwise LDR MAE', out_dir=figures_dir,
                 prefix='dokls_cmp_pointwise_mae', yscale='log')
    plot_compare(kl, eldr, ms_eldr, eldr, ms_eldr, NVALS, metric='regret',
                 ylabel='Rel. regret (per-point pool)', out_dir=figures_dir,
                 prefix='dokls_cmp_regret', yscale='linear')
    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='dokls-vs-MS comparison (styling prototype).')
    p.add_argument('--variant', type=str, default=None)
    args = p.parse_args()
    main(args.variant)
