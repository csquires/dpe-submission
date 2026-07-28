"""FINAL dokls-vs-model_selection comparison over N_* at fixed N=8192.

x = N_* (the p* sample budget) in {2048, 4096, 8192}; N (the p0/p1 fit budget) is
pinned to 8192. this is the decoupled counterpart to plot_compare's diagonal (N_*=N)
prototype: here both dokls and MS sweep N_* with the fit budget held fixed, matching
model_selection's fixed-train / swept-test design.

3 figures, each a 2x7 grid (rows = p* in {q0, q1}, cols = the 7 KL levels):
  dokls_cmp_vsNstar_eldr_err       -- absolute ELDR error
  dokls_cmp_vsNstar_pointwise_mae  -- pointwise LDR MAE
  dokls_cmp_vsNstar_regret         -- relative regret, computed AT PLOT TIME

styling (canonical dokls scheme; color = method family):
  MS direct   solid  + open circle 'o'   (6 shared methods)
  dokls base  dashed + square 's'
  dokls _DV   dotted + star   '*'
  dokls _NWJ  dotted + X       'X'
regret at each (panel, N_*): normalize 0=best..1=worst over the series present at
that point (10 dokls + up to 6 MS together), from their eldr_err.

CRUX handled by load_dokls_nstar: N_* lives only in the FILE NAME. all dokls
processed files carry N=8192 in their metric keys (suffix _{pstar}_8192) -- the
diagonal file two_leg_q{p}_N8192.h5 (N_*=8192) and the decoupled files
two_leg_q{p}_N8192_ns{2048,4096}.h5 (N_*=2048/4096) share that suffix. we select
N_* by opening the right file, not by parsing the key.

usage: python -m ex.ablations.dokls.plot_vs_nstar
"""
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator
import numpy as np

from ex.utils.plot_style import apply as apply_style
from ex.utils.faceted_lines import order_methods
from ex.ablations.dokls.variants import resolve
from ex.ablations.dokls.plot_compare import comp_style, regret_over_contenders
from ex.ablations.dokls.plot_vs_N import load_ms, MS_COMMON

PSTARS = [0, 1]
PNAME = {0: 'q_0', 1: 'q_1'}
NSTARS = [2048, 4096, 8192]        # x-axis (p* budget)
N_FIXED = 8192                     # p0/p1 budget, pinned; also the dokls key suffix


def _dokls_fname(pstar, nstar):
    """diagonal file for N_*=8192, decoupled _ns file otherwise (all N=8192)."""
    if nstar == N_FIXED:
        return f'two_leg_q{pstar}_N{N_FIXED}.h5'
    return f'two_leg_q{pstar}_N{N_FIXED}_ns{nstar}.h5'


def load_dokls_nstar(processed_dir, mean_key, se_key):
    """{(pstar, nstar): {method: (mean(7,), se(7,) or None)}} over N_*.

    keys are '{mean_key}_two_leg_{method}_{pstar}_{N_FIXED}' in every file (N, not
    N_*); N_* is chosen by _dokls_fname, not by the suffix.
    """
    data = {}
    for pstar in PSTARS:
        for nstar in NSTARS:
            path = os.path.join(processed_dir, _dokls_fname(pstar, nstar))
            md = {}
            if os.path.exists(path):
                prefix, suffix = f'{mean_key}_two_leg_', f'_{pstar}_{N_FIXED}'
                with h5py.File(path, 'r') as f:
                    for k in f.keys():
                        if not (k.startswith(prefix) and k.endswith(suffix)):
                            continue
                        m = k[len(prefix): len(k) - len(suffix)]
                        se = None
                        if se_key:
                            sk = f'{se_key}_two_leg_{m}_{pstar}_{N_FIXED}'
                            if sk in f:
                                se = f[sk][:]
                        md[m] = (f[k][:], se)
            data[(pstar, nstar)] = md
    return data


def load_ms_nstar(ms_ref_dir, kind):
    """{(pstar, nstar): {method: (7,)}} from ms_ref/tr8192_te{nstar}.h5 (q0/q1 cols)."""
    out = {}
    for nstar in NSTARS:
        per_ps = load_ms(os.path.join(ms_ref_dir, f'tr8192_te{nstar}.h5'), kind)
        for pstar in PSTARS:
            out[(pstar, nstar)] = per_ps.get(pstar, {})
    return out


def _pool(dokls_err, ms_err, pstar, j, nstar):
    """{(source, method): eldr_err} present at (pstar, KL=j, N_*=nstar)."""
    pool = {('dokls', m): arr[j] for m, (arr, _se) in dokls_err[(pstar, nstar)].items()}
    for m, arr in ms_err.get((pstar, nstar), {}).items():
        pool[('ms', m)] = arr[j]
    return pool


def _regret_point(dokls_err, ms_err, pstar, j, nstar, key):
    return regret_over_contenders(_pool(dokls_err, ms_err, pstar, j, nstar)).get(key, np.nan)


def _dokls_methods(dokls_err):
    """canonical-ordered dokls methods with finite data in >=1 panel."""
    seen = set()
    for md in dokls_err.values():
        seen |= set(md)
    return [m for m in order_methods(seen)
            if any(m in md and np.isfinite(md[m][0]).any() for md in dokls_err.values())]


def plot_metric(kl, dokls_err, ms_err, dokls_val, ms_val, *, metric, ylabel,
                out_dir, prefix, yscale):
    """one 2x7 (p* x KL) figure over N_*. metric in {eldr_err, mae, regret}.

    dokls_val/ms_val carry the plotted value for eldr_err/mae; for regret both are
    ignored and each point is derived from dokls_err/ms_err via _regret_point.
    """
    apply_style()
    nkl = len(kl)
    x = np.array(NSTARS, dtype=float)
    dokls_methods = _dokls_methods(dokls_err)
    fig, axes = plt.subplots(2, nkl, figsize=(2.7 * nkl, 6.6),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for j in range(nkl):
            ax = axes[i][j]
            for m in dokls_methods:                       # dokls traces over N_*
                if metric == 'regret':
                    ys = np.array([_regret_point(dokls_err, ms_err, pstar, j, ns, ('dokls', m))
                                   for ns in NSTARS])
                else:
                    ys = np.array([dokls_val[(pstar, ns)][m][0][j]
                                   if m in dokls_val[(pstar, ns)] else np.nan
                                   for ns in NSTARS])
                if np.isfinite(ys).any():
                    ax.plot(x, ys, **comp_style(m, 'dokls'), linewidth=0.85,
                            markersize=4, alpha=0.85)
            for m in MS_COMMON:                           # MS direct traces over N_*
                if metric == 'regret':
                    ys = np.array([_regret_point(dokls_err, ms_err, pstar, j, ns, ('ms', m))
                                   for ns in NSTARS])
                else:
                    ys = np.array([ms_val[(pstar, ns)][m][j]
                                   if m in ms_val.get((pstar, ns), {}) else np.nan
                                   for ns in NSTARS])
                if np.isfinite(ys).any():
                    c = comp_style(m, 'ms')['color']
                    ax.plot(x, ys, color=c, linestyle='-', marker='o',
                            markerfacecolor='none', markeredgewidth=1.5,
                            linewidth=0.9, markersize=6, alpha=0.9, zorder=5)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xticks(NSTARS)
            ax.set_xticklabels([f'{ns // 1024}k' for ns in NSTARS], fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'KL$={kl[j]:g}$', fontsize=10)
            if i == 1:
                ax.set_xlabel(r'$N_*$ ($N=8192$)')
            if j == 0:
                ax.set_ylabel(rf'$p_*={PNAME[pstar]}$' + '\n' + ylabel)
    _legend(fig, dokls_methods)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')


def _legend(fig, dokls_methods):
    handles = [Line2D([0], [0], **comp_style(m, 'dokls'), markersize=5)
               for m in dokls_methods]
    labels = [f'{m} (2-leg)' for m in dokls_methods]
    src = [Line2D([0], [0], color='0.35', linestyle='-', marker='o',
                  markerfacecolor='none', label='MS direct')]
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
    ms_ref_dir = os.path.join(os.path.dirname(__file__), 'ms_ref')

    dokls_eldr = load_dokls_nstar(processed_dir, 'eldr_err_mean', 'eldr_err_se')
    dokls_mae = load_dokls_nstar(processed_dir, 'mae', None)
    ms_eldr = load_ms_nstar(ms_ref_dir, 'eldr_err')
    ms_mae = load_ms_nstar(ms_ref_dir, 'mae')

    plot_metric(kl, dokls_eldr, ms_eldr, dokls_eldr, ms_eldr, metric='eldr_err',
                ylabel='ELDR error (abs)', out_dir=figures_dir,
                prefix='dokls_cmp_vsNstar_eldr_err', yscale='log')
    plot_metric(kl, dokls_eldr, ms_eldr, dokls_mae, ms_mae, metric='mae',
                ylabel='Pointwise LDR MAE', out_dir=figures_dir,
                prefix='dokls_cmp_vsNstar_pointwise_mae', yscale='log')
    plot_metric(kl, dokls_eldr, ms_eldr, dokls_eldr, ms_eldr, metric='regret',
                ylabel='Rel. regret (per-point pool)', out_dir=figures_dir,
                prefix='dokls_cmp_vsNstar_regret', yscale='linear')
    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='dokls-vs-MS comparison over N_* (fixed N=8192).')
    p.add_argument('--variant', type=str, default=None)
    args = p.parse_args()
    main(args.variant)
