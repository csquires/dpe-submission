"""
dokls metric-vs-N plots at each KL level, with the model_selection direct-DRE
comparator overlaid on the one matched cell (Nstar = N = 8192, p* in {q0, q1}).

3 figures (eldr_err, pointwise_mae, regret). each a 2x7 grid: rows = p* {q0, q1},
cols = the 7 KL levels. x = N (log; dokls diagonal Nstar = N). dokls traces use
the family-color + loss-linestyle scheme with SQUARE markers; the MS direct
comparator is an open CIRCLE at N=8192 for the 6 methods common to both
(BDRE, MultiHeadTDRE, TSM, CTSM, VFM, FMDRE) -- dokls's NWJ/DV variants and MS's
triangular/S2 variants have no counterpart.

usage: python -m ex.ablations.dokls.plot_vs_N
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
from ex.utils.faceted_lines import order_methods, MARKER_SIZE
from ex.ablations.dokls.variants import resolve
from ex.ablations.dokls.step4_plot_results import dokls_style, PSTARS, PNAME, NVALS

MS_COMMON = ['BDRE', 'MultiHeadTDRE', 'TSM', 'CTSM', 'VFM', 'FMDRE']
MS_TESTIDX = {0: 2, 1: 3}          # dokls pstar_idx -> MS test-set idx (q0=2, q1=3)
MS_CELL_N = 8192                   # the single matched N


def load_dokls(processed_dir, mean_key, se_key):
    """{(pstar, N): {method: (mean(n_kl,), se or None)}} from the per-variant files."""
    data = {}
    for pstar in PSTARS:
        for N in NVALS:
            path = os.path.join(processed_dir, f'two_leg_q{pstar}_N{N}.h5')
            if not os.path.exists(path):
                continue
            prefix, suffix = f'{mean_key}_two_leg_', f'_{pstar}_{N}'
            md = {}
            with h5py.File(path, 'r') as f:
                for k in f.keys():
                    if k.startswith(prefix) and k.endswith(suffix):
                        m = k[len(prefix): len(k) - len(suffix)]
                        se = None
                        if se_key and f'{se_key}_two_leg_{m}_{pstar}_{N}' in f:
                            se = f[f'{se_key}_two_leg_{m}_{pstar}_{N}'][:]
                        md[m] = (f[k][:], se)
            data[(pstar, N)] = md
    return data


def load_ms(ms_h5, kind):
    """{pstar: {method: (n_kl,)}} at the matched cell. kind in {eldr_err, regret, mae}."""
    out = {0: {}, 1: {}}
    if not os.path.exists(ms_h5):
        return out
    with h5py.File(ms_h5, 'r') as f:
        for m in MS_COMMON:
            if kind == 'mae':
                key = f'maes_by_kl_{m}'
                if key not in f:
                    continue
                val = np.nanmean(f[key][:], axis=1)          # (n_kl, ntest)
            else:
                key = f'{kind}_{m}_mean'
                if key not in f:
                    continue
                val = f[key][:]                              # (n_kl, ntest)
            for ps, ti in MS_TESTIDX.items():
                out[ps][m] = val[:, ti]
    return out


def _methods(dokls_data):
    seen = set()
    for md in dokls_data.values():
        seen |= set(md)
    return [m for m in order_methods(seen)
            if any(m in md and np.isfinite(md[m][0]).any() for md in dokls_data.values())]


def plot_metric(kl, dokls_data, ms_data, *, ylabel, out_dir, prefix, yscale):
    apply_style()
    methods = _methods(dokls_data)
    nkl = len(kl)
    N = np.array(NVALS, dtype=float)
    fig, axes = plt.subplots(2, nkl, figsize=(2.55 * nkl, 6.4),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for j in range(nkl):
            ax = axes[i][j]
            for m in methods:                                # dokls traces vs N
                ys = np.array([dokls_data.get((pstar, Nv), {}).get(m, (np.full(nkl, np.nan),))[0][j]
                               for Nv in NVALS])
                if not np.isfinite(ys).any():
                    continue
                kw = dokls_style(m)                          # family color + variant marker/linestyle
                ax.plot(N, ys, markersize=MARKER_SIZE, linewidth=0.9, alpha=0.8, **kw)
            for m in MS_COMMON:                              # MS comparator @ N=8192
                if m in ms_data.get(pstar, {}):
                    yv = ms_data[pstar][m][j]
                    c = style_for(m)['color']
                    ax.plot([MS_CELL_N], [yv], marker='o', markersize=MARKER_SIZE + 3,
                            markerfacecolor='none', markeredgecolor=c,
                            markeredgewidth=1.6, linestyle='none', zorder=5)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xticks(NVALS)
            ax.set_xticklabels([f'{n // 1024}k' for n in NVALS], fontsize=8)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'KL$={kl[j]:g}$', fontsize=10)
            if i == 1:
                ax.set_xlabel(r'$N$')
            if j == 0:
                ax.set_ylabel(rf'$p_* = {PNAME[pstar]}$' + '\n' + ylabel)

    method_handles = [Line2D([0], [0], markersize=MARKER_SIZE, **dokls_style(m))
                      for m in methods]
    src_handles = [
        Line2D([0], [0], marker='s', color='0.35', linestyle='-', label='dokls (two-leg)'),
        Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor='0.35',
               linestyle='none', markersize=MARKER_SIZE + 3, label='MS direct @ N=8192'),
    ]
    leg1 = fig.legend(method_handles, methods, loc='lower center', ncol=5,
                      fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.045))
    fig.add_artist(leg1)
    fig.legend(handles=src_handles, loc='lower center', ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
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
    ms_h5 = os.path.join(os.path.dirname(__file__), 'ms_ref', 'tr8192_te8192.h5')

    jobs = [
        ('eldr_err_mean', 'eldr_err_se', 'eldr_err', 'ELDR error (abs)', 'log',
         'dokls_vsN_eldr_err'),
        ('mae', None, 'mae', 'Pointwise LDR MAE', 'log', 'dokls_vsN_pointwise_mae'),
        ('regret_mean', 'regret_se', 'regret', 'Rel. ELDR regret', 'linear',
         'dokls_vsN_regret'),
    ]
    for mean_key, se_key, ms_kind, ylabel, yscale, prefix in jobs:
        dokls_data = load_dokls(processed_dir, mean_key, se_key)
        ms_data = load_ms(ms_h5, ms_kind)
        plot_metric(kl, dokls_data, ms_data, ylabel=ylabel, out_dir=figures_dir,
                    prefix=prefix, yscale=yscale)
    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='dokls metric-vs-N plots + MS overlay.')
    p.add_argument('--variant', type=str, default=None)
    args = p.parse_args()
    main(args.variant)
