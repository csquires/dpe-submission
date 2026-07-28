"""
dokls-ONLY plots over ALL dokls cells (no MS overlay). Two sweeps:
  DIAGONAL  : N_* = N, N in {1024, 2048, 4096, 8192}
  DECOUPLED : N = 8192 fixed, N_* in {2048, 4096, 8192}
each shown both KL-stratified and N-stratified. The DIAGONAL KL-stratified
banner is the existing dokls_*_grid; this module adds the other three:

  dokls_diag_vsN_{metric}     x=N,   panels=KL   (diagonal, N-stratified)
  dokls_dec_vsNstar_{metric}  x=N_*, panels=KL   (decoupled, N-stratified)
  dokls_dec_vsKL_{metric}     x=KL,  panels=N_*  (decoupled, KL-stratified)

metric in {eldr_err, pointwise_mae, regret}. regret = the precomputed per-cell
cross-method dokls regret (regret_mean) from step3. styling = dokls_style
(family color + loss-variant marker/linestyle); no MS here (that lives in the
dokls_cmp_vs{Nstar,KL} comparison plots).

usage: python -m ex.ablations.dokls.plot_dokls_only
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from ex.utils.plot_style import apply as apply_style
from ex.utils.faceted_lines import order_methods, MARKER_SIZE
from ex.ablations.dokls.variants import resolve
from ex.ablations.dokls.step4_plot_results import dokls_style, PSTARS, PNAME
from ex.ablations.dokls.plot_vs_N import load_dokls, NVALS
from ex.ablations.dokls.plot_vs_nstar import load_dokls_nstar, NSTARS


def _methods(data):
    seen = set()
    for md in data.values():
        seen |= set(md)
    return [m for m in order_methods(seen)
            if any(m in md and np.isfinite(md[m][0]).any() for md in data.values())]


def _legend(fig, methods):
    handles = [Line2D([0], [0], **dokls_style(m), markersize=5) for m in methods]
    fig.legend(handles, methods, loc='lower center', ncol=5, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.01))


def plot_vs_budget(kl, data, budgets, xlabel, ylabel, out_dir, prefix, yscale):
    """x = budget (N or N_*), 2x7 grid (p* rows x KL cols). data={(pstar,budget):{m:(mean(7,),se)}}."""
    apply_style()
    nkl = len(kl)
    x = np.array(budgets, dtype=float)
    methods = _methods(data)
    fig, axes = plt.subplots(2, nkl, figsize=(2.6 * nkl, 6.4),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for j in range(nkl):
            ax = axes[i][j]
            for m in methods:
                ys = np.array([data.get((pstar, b), {}).get(m, (np.full(nkl, np.nan),))[0][j]
                               for b in budgets])
                if np.isfinite(ys).any():
                    ax.plot(x, ys, markersize=MARKER_SIZE, linewidth=0.9, alpha=0.85,
                            **dokls_style(m))
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xticks(budgets)
            ax.set_xticklabels([f'{b // 1024}k' for b in budgets], fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'KL$={kl[j]:g}$', fontsize=10)
            if i == 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(rf'$p_*={PNAME[pstar]}$' + '\n' + ylabel)
    _legend(fig, methods)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}')


def plot_vs_kl(kl, data, budgets, budget_label, ylabel, out_dir, prefix, yscale):
    """x = KL, 2 x len(budgets) grid (p* rows x budget cols). dokls-only."""
    apply_style()
    nkl = len(kl)
    x = np.array(kl, dtype=float)
    methods = _methods(data)
    nc = len(budgets)
    fig, axes = plt.subplots(2, nc, figsize=(3.1 * nc, 6.4),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for jc, b in enumerate(budgets):
            ax = axes[i][jc]
            for m in methods:
                md = data.get((pstar, b), {})
                ys = md[m][0] if m in md else np.full(nkl, np.nan)
                if np.isfinite(ys).any():
                    ax.plot(x, ys, markersize=MARKER_SIZE, linewidth=0.9, alpha=0.85,
                            **dokls_style(m))
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'${budget_label}={b // 1024}$k', fontsize=10)
            if i == 1:
                ax.set_xlabel('KL')
            if jc == 0:
                ax.set_ylabel(rf'$p_*={PNAME[pstar]}$' + '\n' + ylabel)
    _legend(fig, methods)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}')


def main(variant=None):
    _tag, _route, config = resolve(variant)
    pd = config['processed_results_dir']
    fd = config['figures_dir']
    kl = np.array(config['kl_distances'], dtype=float)
    jobs = [
        ('eldr_err_mean', 'eldr_err_se', 'ELDR error (abs)', 'log', 'eldr_err'),
        ('mae', None, 'Pointwise LDR MAE', 'log', 'pointwise_mae'),
        ('regret_mean', 'regret_se', 'Rel. ELDR regret', 'linear', 'regret'),
    ]
    for mean_key, se_key, ylabel, yscale, mn in jobs:
        diag = load_dokls(pd, mean_key, se_key)          # {(pstar, N): ...}
        dec = load_dokls_nstar(pd, mean_key, se_key)     # {(pstar, N_*): ...}
        plot_vs_budget(kl, diag, NVALS, r'$N$ ($N_*=N$ diagonal)', ylabel, fd,
                       f'dokls_diag_vsN_{mn}', yscale)
        plot_vs_budget(kl, dec, NSTARS, r'$N_*$ ($N=8192$)', ylabel, fd,
                       f'dokls_dec_vsNstar_{mn}', yscale)
        plot_vs_kl(kl, dec, NSTARS, 'N_*', ylabel, fd, f'dokls_dec_vsKL_{mn}', yscale)
    print(f'\nDone. Figures in: {fd}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='dokls-only plots over all dokls cells.')
    p.add_argument('--variant', type=str, default=None)
    args = p.parse_args()
    main(args.variant)
