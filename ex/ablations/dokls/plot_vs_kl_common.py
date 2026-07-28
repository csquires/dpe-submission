"""
dokls-vs-MS comparison on the COMMON cells, KL-stratified (x = KL).

Companion/transpose of plot_vs_nstar.py (which is x = N_*). Here x = the 7 KL
levels; a 2x3 grid: rows = p* {q0, q1}, cols = N_* {2048, 4096, 8192} at fixed
N=8192 (the cells shared by dokls two-leg and MS direct). MS + all dokls methods
overlaid. 3 figures: eldr_err, pointwise MAE, regret (regret computed at plot
time per (panel, KL) over the contenders present). Reuses the vs-Nstar loaders
and the shared comp_style, so styling matches: MS = solid+circle; dokls base =
dashed+square, _DV = dotted+star, _NWJ = dotted+X; color = method family.

usage: python -m ex.ablations.dokls.plot_vs_kl_common
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ex.utils.plot_style import apply as apply_style
from ex.ablations.dokls.variants import resolve
from ex.ablations.dokls.step4_plot_results import PSTARS, PNAME
from ex.ablations.dokls.plot_compare import comp_style
from ex.ablations.dokls.plot_vs_N import MS_COMMON
from ex.ablations.dokls.plot_vs_nstar import (
    load_dokls_nstar, load_ms_nstar, _regret_point, _dokls_methods, _legend,
    NSTARS)


def plot_metric(kl, dokls_err, ms_err, dokls_val, ms_val, *, metric, ylabel,
                out_dir, prefix, yscale):
    """one 2x3 (p* x N_*) figure over KL. x = kl. metric in {eldr_err,mae,regret}."""
    apply_style()
    nkl = len(kl)
    x = np.array(kl, dtype=float)
    dokls_methods = _dokls_methods(dokls_err)
    ncol = len(NSTARS)
    fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for jc, ns in enumerate(NSTARS):
            ax = axes[i][jc]
            for m in dokls_methods:                       # dokls traces over KL
                if metric == 'regret':
                    ys = np.array([_regret_point(dokls_err, ms_err, pstar, j, ns, ('dokls', m))
                                   for j in range(nkl)])
                else:
                    md = dokls_val.get((pstar, ns), {})
                    ys = md[m][0] if m in md else np.full(nkl, np.nan)
                if np.isfinite(ys).any():
                    ax.plot(x, ys, **comp_style(m, 'dokls'), linewidth=0.85,
                            markersize=4, alpha=0.85)
            for m in MS_COMMON:                           # MS direct traces over KL
                if metric == 'regret':
                    ys = np.array([_regret_point(dokls_err, ms_err, pstar, j, ns, ('ms', m))
                                   for j in range(nkl)])
                else:
                    ys = ms_val.get((pstar, ns), {}).get(m, np.full(nkl, np.nan))
                if np.isfinite(ys).any():
                    c = comp_style(m, 'ms')['color']
                    ax.plot(x, ys, color=c, linestyle='-', marker='o',
                            markerfacecolor='none', markeredgewidth=1.5,
                            linewidth=0.9, markersize=6, alpha=0.9, zorder=5)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'$N_*={ns // 1024}$k ($N=8192$)', fontsize=10)
            if i == 1:
                ax.set_xlabel('KL')
            if jc == 0:
                ax.set_ylabel(rf'$p_*={PNAME[pstar]}$' + '\n' + ylabel)
    _legend(fig, dokls_methods)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')


def main(variant=None):
    _tag, _route, config = resolve(variant)
    processed_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    kl = np.array(config['kl_distances'], dtype=float)
    ms_ref_dir = os.path.join(os.path.dirname(__file__), 'ms_ref')

    dokls_eldr = load_dokls_nstar(processed_dir, 'eldr_err_mean', 'eldr_err_se')
    dokls_mae = load_dokls_nstar(processed_dir, 'mae', None)
    ms_eldr = load_ms_nstar(ms_ref_dir, 'eldr_err')
    ms_mae = load_ms_nstar(ms_ref_dir, 'mae')

    plot_metric(kl, dokls_eldr, ms_eldr, dokls_eldr, ms_eldr, metric='eldr_err',
                ylabel='ELDR error (abs)', out_dir=figures_dir,
                prefix='dokls_cmp_vsKL_eldr_err', yscale='log')
    plot_metric(kl, dokls_eldr, ms_eldr, dokls_mae, ms_mae, metric='mae',
                ylabel='Pointwise LDR MAE', out_dir=figures_dir,
                prefix='dokls_cmp_vsKL_pointwise_mae', yscale='log')
    plot_metric(kl, dokls_eldr, ms_eldr, dokls_eldr, ms_eldr, metric='regret',
                ylabel='Rel. regret (per-point pool)', out_dir=figures_dir,
                prefix='dokls_cmp_vsKL_regret', yscale='linear')
    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='dokls-vs-MS comparison over KL on common cells.')
    p.add_argument('--variant', type=str, default=None)
    args = p.parse_args()
    main(args.variant)
