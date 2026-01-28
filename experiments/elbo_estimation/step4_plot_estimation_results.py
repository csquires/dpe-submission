import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


sns.set_style('whitegrid')
plt.style.use('full-width.mplstyle')

config = yaml.load(open('experiments/elbo_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
ALPHAS = config['alphas']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/errors_d={DATA_DIM},nsamples={NSAMPLES}.h5'

# Load processed results
with h5py.File(processed_results_filename, 'r') as f:
    # Find all algorithms
    alg_names = [key.replace('mae_', '') for key in f.keys() if key.startswith('mae_')]
    mae_by_alg = {alg_name: f[f'mae_{alg_name}'][:] for alg_name in alg_names}

# Plot
plt.clf()
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=len(ALPHAS))
for i, alpha in enumerate(ALPHAS):
    for alg_name, mae in mae_by_alg.items():
        # mae shape: (len(DESIGN_EIG_PERCENTAGES), len(ALPHAS))
        axes[i].plot(DESIGN_EIG_PERCENTAGES, mae[:, i], label=alg_name)
    axes[i].set_title(fr"$\alpha = {alpha}$")

fig.supxlabel(r"$\beta$ (Design Optimality Percentage)")
fig.supylabel(r"ELBO Estimation Error")
handles, labels = axes[0].get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/elbo_estimation.pdf')