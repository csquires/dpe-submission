"""plot per-method normalized EIG regret vs design beta, split into 4 panels.

line = median-of-medians regret per beta (see step3); shaded band = IQR of
1000 bootstrap resamples (priors and designs drawn with replacement).

styles + colors come from ex/utils/plot_style.py.
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ex.utils.plot_style import (
    apply as apply_style,
    style_for,
    METHOD_GROUPS,
    GROUP_LABEL,
    ERROR_BAND_ALPHA,
)


config = yaml.load(open('ex/synth/eig/config1.yaml', 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/regret_by_beta_d={DATA_DIM},nsamples={NSAMPLES}.h5'

with h5py.File(processed_results_filename, 'r') as f:
    design_eig_percentages = f['design_eig_percentages'][:]
    regret = {k[len('regret_by_beta_'):]: f[k][:]
              for k in f.keys() if k.startswith('regret_by_beta_')}
    lo = {k[len('regret_lo_by_beta_'):]: f[k][:]
          for k in f.keys() if k.startswith('regret_lo_by_beta_')}
    hi = {k[len('regret_hi_by_beta_'):]: f[k][:]
          for k in f.keys() if k.startswith('regret_hi_by_beta_')}

apply_style()

# shared y-limits across panels for side-by-side comparison.
all_hi = []
for group_methods in METHOD_GROUPS.values():
    for m in group_methods:
        if m not in regret:
            continue
        all_hi.append(np.nanmax(hi.get(m, regret[m])))
y_max = (max(all_hi) * 1.05) if all_hi else 1.0


os.makedirs(FIGURES_DIR, exist_ok=True)
for group, methods in METHOD_GROUPS.items():
    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        if method not in regret:
            continue
        r = regret[method]
        if not np.any(np.isfinite(r)):
            continue
        kw = style_for(method)
        ax.plot(design_eig_percentages, r, label=method, **kw)
        ax.fill_between(design_eig_percentages, lo[method], hi[method],
                        color=kw['color'], alpha=ERROR_BAND_ALPHA, linewidth=0)
    ax.set_title(GROUP_LABEL[group])
    ax.set_xlabel(r'Design optimality $\beta = \mathrm{EIG}(\xi) / \mathrm{EIG}_{\max}$')
    ax.set_ylabel('Rel. EIG regret (MoM, IQR band)')
    ax.legend(loc='best')
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    pdf_path = os.path.join(FIGURES_DIR, f'eig_estimation_{group}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'eig_estimation_{group}.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f'  {group:9s} -> {png_path}')
