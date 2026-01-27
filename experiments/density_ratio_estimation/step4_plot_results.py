import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
PROCESSED_RESULTS_DIR = config['processed_results_dir']
FIGURES_DIR = config['figures_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']

#filename = f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5'
filename = f'{PROCESSED_RESULTS_DIR}/added_cauchy_01.h5'

with h5py.File(filename, 'r') as f:
    maes_by_kl = {key.replace('maes_by_kl_', ''): f[key][:] for key in f.keys()}

avg_mae_by_kl = {alg: np.mean(arr, axis=1) for alg, arr in maes_by_kl.items()}
all_mins = [arr.min() for arr in avg_mae_by_kl.values()]
all_maxs = [arr.max() for arr in avg_mae_by_kl.values()]
y_min = min(all_mins) if all_mins else 0.0
y_max = max(all_maxs) if all_maxs else 1.0

# setup
plt.clf()
sns.set_style('whitegrid')
plt.style.use('full-width.mplstyle')
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)
# colors
colors = {
    "BDRE": "#1f77b4",
    "MDRE": "#2ca02c",
    "TSM": "#d62728",
    "TriangularTSM": "#17becf",
    "TDRE_5": "#ff7f0e",
    "TDRE_10": "#8c564b", # default TDRE
    "TDRE_15": "#9467bd",
    "TDRE_20": "#e377c2",
    "TDRE_30": "#7f7f7f",
    "MDRE_5": "#17becf",
    "MDRE_10": "#7f7f7f", # default MDRE
    "MDRE_15": "#2ca02c",
    "MDRE_20": "#8c564b",
    "MDRE_30": "#e377c2",
}

# tdre_order = ["TDRE_5", "TDRE_10", "TDRE_15", "TDRE_20", "TDRE_30"]
# mdre_order = ["MDRE_5", "MDRE_10", "MDRE_15", "MDRE_20", "MDRE_30"]

tdre_order = ["TDRE_5"]
mdre_order = ["MDRE_15"]

# plotting
for i in range(NTEST_SETS):
    axes[i].plot(KL_DISTANCES, avg_mae_by_kl["BDRE"][:, i], label='BDRE', color=colors["BDRE"])
    # axes[i].plot(KL_DISTANCES, avg_mae_by_kl["MDRE"][:, i], label='MDRE', color=colors["MDRE"])
    axes[i].plot(KL_DISTANCES, avg_mae_by_kl["TSM"][:, i], label='TSM', color=colors["TSM"])
    axes[i].plot(KL_DISTANCES, avg_mae_by_kl["TriangularTSM"][:, i], label="TriangularTSM", color=colors["TriangularTSM"])
    # for tdre_name in tdre_order:
    #     if tdre_name in avg_mae_by_kl:
    #         label = f'TDRE ({tdre_name.split("_")[1]})'
    #         axes[i].plot(KL_DISTANCES, avg_mae_by_kl[tdre_name][:, i], label=label, color=colors[tdre_name])
    # for tdre_name in tdre_order:
    #     if tdre_name in avg_mae_by_kl:
    #         axes[i].plot(KL_DISTANCES, avg_mae_by_kl[tdre_name][:, i], label="TDRE", color=colors[tdre_name])
    # for mdre_name in mdre_order:
    #     if mdre_name in avg_mae_by_kl:
    #         label = f'MDRE ({mdre_name.split("_")[1]})'
    #         axes[i].plot(KL_DISTANCES, avg_mae_by_kl[mdre_name][:, i], label=label, color=colors[mdre_name])
    # for mdre_name in mdre_order:
    #     if mdre_name in avg_mae_by_kl:
    #         axes[i].plot(KL_DISTANCES, avg_mae_by_kl[mdre_name][:, i], label="MDRE", color=colors[mdre_name])
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[i].set_xlabel(r'KL$(p_0 \| p_1)$')
axes[0].set_ylabel('Mean Absolute Error \n (Test Set)')
axes[0].set_title(r'$p_* = p_0$')
axes[1].set_title(r'$p_* = p_1$')
axes[2].set_title(r'$p_* = q_0$')
axes[3].set_title(r'$p_* = q_1$') # Cauchy
handles, labels = axes[0].get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs(FIGURES_DIR, exist_ok=True)
# plt.savefig(f'{FIGURES_DIR}/varying_kl_03.pdf')
plt.savefig(f'{FIGURES_DIR}/Cauchy_test.pdf')
