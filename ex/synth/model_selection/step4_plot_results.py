"""
Step 4: Plot Results for model_selection ELDR Estimation

eig-/occupancy-style line plots: one figure per test set, every method on the
same axes (thin lines + translucent +/- SE band), colors/markers from
ex.utils.plot_style. the shared legend is emitted as its own figure.

plots ABSOLUTE ELDR error (eldr_err_{m} from step3) vs KL(p0 || p1).
"""
import os

import h5py
import numpy as np

from src.utils.io import _load_config
from ex.utils.faceted_lines import plot_panels, plot_legend


TEST_SET_TITLES = [r'$p_* = p_0$', r'$p_* = p_1$', r'$p_* = q_0$', r'$p_* = q_1$']


def main():
    config = _load_config('ex/synth/model_selection/config.yaml')
    processed_dir = config['processed_results_dir']
    figures_dir = config['figures_dir']
    kl_distances = np.array(config['kl_distances'], dtype=float)
    n_test = config['ntest_sets']

    summary = f'{processed_dir}/new_pstar.h5'
    if not os.path.exists(summary):
        raise FileNotFoundError(f'{summary} not found; run step3 first.')

    # eldr_err_{m}_mean / _se are grids of shape (n_kl, n_test) = absolute ELDR error.
    with h5py.File(summary, 'r') as f:
        methods = [k[len('eldr_err_'):-len('_mean')]
                   for k in f.keys()
                   if k.startswith('eldr_err_') and k.endswith('_mean')]
        mean = {m: f[f'eldr_err_{m}_mean'][:] for m in methods}   # (n_kl, n_test)
        se   = {m: f[f'eldr_err_{m}_se'][:]   for m in methods}

    # facets = test sets; x = KL distance. grids are already [n_kl, n_test].
    facets = [(f'test{t}', TEST_SET_TITLES[t] if t < len(TEST_SET_TITLES) else f'test {t}')
              for t in range(n_test)]

    plotted = plot_panels(
        kl_distances, facets, mean, se,
        xlabel=r'KL$(p_0 \| p_1)$',
        ylabel='ELDR Error (abs)',
        out_dir=figures_dir, prefix='model_selection_eldr_err',
        xscale='log', yscale='log',
    )
    plot_legend(plotted, figures_dir, prefix='model_selection_eldr_err')

    print(f'\nDone. Figures in: {figures_dir}')
    print(f'Methods plotted: {len(plotted)}')


if __name__ == '__main__':
    main()
