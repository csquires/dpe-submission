"""
Step 4: Plot Results for model_selection ELDR Estimation

one figure per metric: a 2x2 grid of panels, one per held-out p* test set, with
every method's trace on each panel (thin lines + translucent band) and a single
shared legend below the grid. sibling {stem}.md/.tex tables carry the plotted
values (one section per p*). metrics:
  regret        -- per-cell normalized ELDR regret (median + bootstrap-SE band)
  eldr_err      -- absolute ELDR error vs analytic truth (mean +/- SE)
  pointwise_mae -- pointwise LDR MAE, aggregated over instances (mean +/- SE)
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
from ex.synth.model_selection.variants import resolve


TEST_SET_TITLES = [r'$p_* = p_0$', r'$p_* = p_1$', r'$p_* = q_0$', r'$p_* = q_1$']
ALIAS = {'MDRE_15': 'MDRE'}


def _style(method):
    return style_for(ALIAS.get(method, method))


def plot_pstar_grid(x, mean, se, *, ylabel, out_dir, prefix, yscale='linear',
                    band_clip=None, ylim=None):
    """one row of p* panels (left to right), all methods each, shared legend below.

    mean/se: dict method -> (n_kl, n_test). band_clip optionally bounds the
    +/- band (e.g. (0, 1) for regret); ylim pins the shared y-range. emits
    {out_dir}/{prefix}.{pdf,png}; returns the ordered methods drawn.
    """
    apply_style()
    band_alpha = 0.12   # 18 overlapping bands per panel; lighter than default
    line_w = 0.85       # thinner + translucent traces so overlaps stay visible
    line_alpha = 0.75
    methods = [m for m in order_methods(mean.keys()) if np.isfinite(mean[m]).any()]
    n_test = next(iter(mean.values())).shape[1]

    ncol_leg = 6
    n_rows_leg = int(np.ceil(len(methods) / ncol_leg))
    panel_h = 4.6
    legend_h = 0.34 * n_rows_leg + 0.2
    fig_h = panel_h + legend_h
    fig, axes = plt.subplots(1, n_test, figsize=(4.9 * n_test, fig_h),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for t, ax in enumerate(axes):
        for m in methods:
            y = mean[m][:, t]
            if not np.isfinite(y).any():
                continue
            e = np.nan_to_num(se[m][:, t])
            kw = _style(m)
            ax.plot(x, y, label=m, linewidth=line_w, markersize=MARKER_SIZE,
                    alpha=line_alpha, **kw)
            lo, hi = y - e, y + e
            if yscale == 'log':
                lo = np.maximum(lo, 1e-4)
            if band_clip is not None:
                lo = np.maximum(lo, band_clip[0])
                hi = np.minimum(hi, band_clip[1])
            ax.fill_between(x, lo, hi, color=kw['color'],
                            alpha=band_alpha, linewidth=0)
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(TEST_SET_TITLES[t] if t < len(TEST_SET_TITLES) else f'test {t}')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(r'KL$(p_0 \| p_1)$')
    axes[0].set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=ncol_leg, fontsize=12,
               framealpha=0.9, handlelength=1.6, columnspacing=1.2)
    fig.tight_layout(pad=0.5, w_pad=0.8, rect=(0, legend_h / fig_h, 1, 1))

    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{prefix}.{ext}'), dpi=150)
    plt.close(fig)
    print(f'  saved {prefix}.{{pdf,png}}')
    return methods


def emit_tables(x, mean, se, methods, *, title, stem, n_test):
    """one table file; a section per p* with rows = methods, cols = KL values."""
    header = ['Method'] + [f'KL={k:g}' for k in x]
    sections = []
    for t in range(n_test):
        rows = [[m] + [fmt_pm(mean[m][ki, t], se[m][ki, t]) for ki in range(len(x))]
                for m in methods]
        label = TEST_SET_TITLES[t].replace('$', '').replace(r'\_', '_') \
            if t < len(TEST_SET_TITLES) else f'test {t}'
        sections.append((f'{title} -- {label}', header, rows))
    write_tables(stem, sections)


def load_pair(f, mean_pat, se_pat):
    """(mean, se) method dicts for keys '{pat}' with {m} substituted."""
    prefix, suffix = mean_pat.split('{m}')
    methods = [k[len(prefix):-len(suffix)] for k in f.keys()
               if k.startswith(prefix) and k.endswith(suffix)]
    mean = {m: f[mean_pat.format(m=m)][:] for m in methods}
    se = {m: f[se_pat.format(m=m)][:] for m in methods}
    return mean, se


def main(variant: str | None = None):
    tag, config = resolve(variant)
    processed_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    kl_distances = np.array(config['kl_distances'], dtype=float)
    n_test = config['ntest_sets']

    summary = f'{processed_dir}/new_pstar.h5'
    if not os.path.exists(summary):
        raise FileNotFoundError(f'{summary} not found; run step3 first.')

    with h5py.File(summary, 'r') as f:
        reg_mean, reg_se = load_pair(f, 'regret_{m}_mean', 'regret_{m}_se')
        err_mean, err_se = load_pair(f, 'eldr_err_{m}_mean', 'eldr_err_{m}_se')
        # pointwise mae is stored raw as maes_by_kl_{m} (n_kl, n_inst, n_test);
        # aggregate over instances to mean +/- SE here.
        mae_mean, mae_se = {}, {}
        for k in f.keys():
            if not k.startswith('maes_by_kl_'):
                continue
            m = k[len('maes_by_kl_'):]
            arr = f[k][:]
            n_inst = arr.shape[1]
            mae_mean[m] = np.nanmean(arr, axis=1)
            mae_se[m] = np.nanstd(arr, axis=1, ddof=1) / np.sqrt(n_inst)

    jobs = [
        (reg_mean, reg_se, 'Rel. ELDR regret', 'model_selection_regret_grid',
         'linear', 'ELDR regret (median +/- bootstrap SE)',
         {'band_clip': (0.0, 1.0), 'ylim': (-0.02, 1.02)}),
        (err_mean, err_se, 'ELDR error (abs)', 'model_selection_eldr_err_grid',
         'log', 'Absolute ELDR error, mean +/- SE', {}),
        (mae_mean, mae_se, 'Pointwise LDR MAE', 'model_selection_pointwise_mae_grid',
         'log', 'Pointwise LDR MAE, mean +/- SE', {}),
    ]
    for mean, se, ylabel, prefix, yscale, title, extra in jobs:
        if not mean:
            print(f'  skip {prefix}: no data')
            continue
        methods = plot_pstar_grid(kl_distances, mean, se, ylabel=ylabel,
                                  out_dir=figures_dir, prefix=prefix, yscale=yscale,
                                  **extra)
        emit_tables(kl_distances, mean, se, methods, title=title,
                    stem=os.path.join(figures_dir, f'{prefix.removesuffix("_grid")}_table'),
                    n_test=n_test)

    print(f'\nDone. Figures in: {figures_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Step 4: Plot ELDR estimation results.')
    parser.add_argument('--variant', type=str, required=True,
                        help='Variant tag (precedence: --variant > $DPE_MS_VARIANT > DEFAULT_TAG)')
    args = parser.parse_args()
    main(args.variant)
