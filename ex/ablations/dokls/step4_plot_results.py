"""
Step 4: Plot Results for dokls ELDR Estimation (model_selection schema).

one figure per metric: a 2x4 grid of panels (rows = p* in {q0, q1}, cols =
N in {1024, 2048, 4096, 8192}), x = KL(p0||p1) on a log axis. every method's
trace on each panel (thin translucent line + light SE band), a single shared
legend below the grid. sibling {stem}.md/.tex tables carry the plotted values
(one section per (p*, N)). metrics:
  eldr_err       -- absolute ELDR error vs analytic truth (mean +/- SE)
  pointwise_mae  -- pointwise LDR MAE aggregated over cells (mean; no band)
  variance       -- estimator variance across instances (mean +/- SE)
  bias_signed    -- signed ELDR bias (mean +/- SE)

styling: color encodes the method FAMILY (base method), linestyle encodes the
loss variant -- base solid, NWJ dashed, DV dotted -- so the 4 classifier
ablation variants stay distinguishable (plain style_for maps them all to gray).
"""
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ex.utils.plot_style import apply as apply_style, style_for
from ex.utils.faceted_lines import order_methods, MARKER_SIZE
from ex.utils.tables import fmt_pm, write_tables
from ex.ablations.dokls.variants import resolve

PSTARS = [0, 1]
PNAME = {0: 'q_0', 1: 'q_1'}
NVALS = [1024, 2048, 4096, 8192]
# canonical dokls trace schema (single source of truth across all dokls plots):
# color = method family; loss variant -> (linestyle, marker).
#   base  dashed  + square 's'
#   _DV   dotted  + star   '*'
#   _NWJ  dotted  + X       'X'
# the model_selection direct comparator (in plot_compare) is solid + circle 'o'.
_VARIANT_STYLE = {'base': ('--', 's'), '_DV': (':', '*'), '_NWJ': (':', 'X')}


def dokls_style(method):
    """family color + per-loss-variant (linestyle, marker); see _VARIANT_STYLE.

    MHT_* share the MultiHeadTDRE color. returns a plot-kw dict with
    color/linestyle/marker.
    """
    base, key = method, 'base'
    for suf in ('_NWJ', '_DV'):
        if method.endswith(suf):
            base, key = method[: -len(suf)], suf
            break
    if base == 'MHT':
        base = 'MultiHeadTDRE'
    ls, marker = _VARIANT_STYLE[key]
    kw = dict(style_for(base))
    kw['linestyle'] = ls
    kw['marker'] = marker
    return kw


def load_metric(processed_dir, mean_key, se_key):
    """{(pstar, N): {method: (mean(n_kl,), se(n_kl,) or None)}}.

    one processed file per (pstar, N) variant; keys are
    '{mean_key}_two_leg_{method}_{pstar}_{N}'.
    """
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
                    if not (k.startswith(prefix) and k.endswith(suffix)):
                        continue
                    m = k[len(prefix): len(k) - len(suffix)]
                    mean = f[k][:]
                    se = None
                    if se_key:
                        sk = f'{se_key}_two_leg_{m}_{pstar}_{N}'
                        if sk in f:
                            se = f[sk][:]
                    md[m] = (mean, se)
            data[(pstar, N)] = md
    return data


def _finite_methods(data):
    """methods with finite data in at least one panel, in canonical order."""
    seen = set()
    for md in data.values():
        seen |= set(md)
    return [m for m in order_methods(seen)
            if any(m in md and np.isfinite(md[m][0]).any() for md in data.values())]


def plot_grid(kl, data, *, ylabel, out_dir, prefix, yscale='linear'):
    """2x4 (p* x N) grid, x=KL, all methods; shared legend below. returns methods."""
    apply_style()
    band_alpha, line_w, line_alpha = 0.12, 0.9, 0.8
    methods = _finite_methods(data)

    nr, nc = len(PSTARS), len(NVALS)
    ncol_leg = 5
    n_rows_leg = int(np.ceil(len(methods) / ncol_leg))
    panel_h = 3.4
    legend_h = 0.34 * n_rows_leg + 0.25
    fig_h = panel_h * nr + legend_h
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, fig_h),
                             sharex=True, sharey=True, squeeze=False)
    for i, pstar in enumerate(PSTARS):
        for j, N in enumerate(NVALS):
            ax = axes[i][j]
            md = data.get((pstar, N), {})
            for m in methods:
                if m not in md:
                    continue
                mean, se = md[m]
                if not np.isfinite(mean).any():
                    continue
                kw = dokls_style(m)
                ax.plot(kl, mean, label=m, linewidth=line_w, alpha=line_alpha,
                        markersize=MARKER_SIZE, **kw)
                if se is not None:
                    e = np.nan_to_num(se)
                    lo, hi = mean - e, mean + e
                    if yscale == 'log':
                        lo = np.maximum(lo, 1e-4)
                    ax.fill_between(kl, lo, hi, color=kw['color'],
                                    alpha=band_alpha, linewidth=0)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(rf'$N={N}$')
            if i == nr - 1:
                ax.set_xlabel(r'KL$(p_0 \| p_1)$')
            if j == 0:
                ax.set_ylabel(rf'$p_* = {PNAME[pstar]}$' + '\n' + ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=ncol_leg, fontsize=11,
               framealpha=0.9, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.8, rect=(0, legend_h / fig_h, 1, 1))

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')
    return methods


def emit_tables(kl, data, methods, *, title, stem):
    """one table file; a section per (p*, N), rows=methods, cols=KL values."""
    header = ['Method'] + [f'KL={k:g}' for k in kl]
    sections = []
    for pstar in PSTARS:
        for N in NVALS:
            md = data.get((pstar, N), {})
            rows = []
            for m in methods:
                if m not in md:
                    continue
                mean, se = md[m]
                se = se if se is not None else np.full_like(mean, np.nan)
                rows.append([m] + [fmt_pm(mean[ki], se[ki]) for ki in range(len(kl))])
            if rows:
                sections.append((f'{title} -- p*={PNAME[pstar]}, N={N}', header, rows))
    if sections:
        write_tables(stem, sections)


def main(variant=None):
    tag, _route, config = resolve(variant)
    processed_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    kl = np.array(config['kl_distances'], dtype=float)

    jobs = [
        ('eldr_err_mean', 'eldr_err_se', 'ELDR error (abs)', 'log',
         'dokls_eldr_err_grid', 'Absolute ELDR error, mean +/- SE'),
        ('mae', None, 'Pointwise LDR MAE', 'log',
         'dokls_pointwise_mae_grid', 'Pointwise LDR MAE'),
        ('regret_mean', 'regret_se', 'Rel. ELDR regret', 'linear',
         'dokls_regret_grid', 'Per-cell normalized ELDR regret, median +/- boot SE'),
        ('variance', 'variance_se', 'Variance', 'log',
         'dokls_variance_grid', 'Estimator variance, mean +/- SE'),
        ('bias_signed', 'bias_signed_se', 'Signed bias', 'linear',
         'dokls_bias_grid', 'Signed ELDR bias, mean +/- SE'),
    ]
    for mean_key, se_key, ylabel, yscale, prefix, title in jobs:
        data = load_metric(processed_dir, mean_key, se_key)
        if not any(data.values()):
            print(f'  skip {prefix}: no data')
            continue
        methods = plot_grid(kl, data, ylabel=ylabel, out_dir=figures_dir,
                            prefix=prefix, yscale=yscale)
        emit_tables(kl, data, methods, title=title,
                    stem=os.path.join(figures_dir, prefix.removesuffix('_grid') + '_table'))
    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Step 4: Plot dokls ELDR results (clean schema).')
    p.add_argument('--variant', type=str, default=None,
                   help='variant tag; only used to resolve config (all variants plotted).')
    args = p.parse_args()
    main(args.variant)
