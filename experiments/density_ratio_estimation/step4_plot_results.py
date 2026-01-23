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

filename = f'{PROCESSED_RESULTS_DIR}/maes_by_kl_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5'
with h5py.File(filename, 'r') as f:
    maes_by_kl_bdre = f['maes_by_kl_bdre'][:]  # (n_kl, n_instances, n_test_sets)
avg_mae_by_kl_bdre = np.mean(maes_by_kl_bdre, axis=1)  # (n_kl, n_test_sets)
y_min = avg_mae_by_kl_bdre.min()
y_max = avg_mae_by_kl_bdre.max()

# setup
plt.clf()
sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)
# plotting
for i in range(3):
    bdre_results = avg_mae_by_kl_bdre[:, i]
    axes[i].plot(KL_DISTANCES, bdre_results, label='BDRE')
    # axes[i].plot(KL_DISTANCES, results[test_set]['tdre'], label='TDRE')
    # axes[i].plot(KL_DISTANCES, results[test_set]['mdre'], label='MDRE')
    # tsm_results = mean_results_df.loc[KL_DISTANCES, i, 'tsm']
    # axes[i].plot(KL_DISTANCES, tsm_results, label='TSM')
    axes[i].set_ylim(y_min, y_max)
    axes[i].set_xscale('log')
    axes[i].set_yscale('log')
    axes[i].set_xlabel(r'KL$(p_0 \| p_1)$')
axes[0].set_ylabel('Density Ratio Error on Test Data')
axes[0].set_title(r'$p_* = p_0$')
axes[1].set_title(r'$p_* = p_1$')
axes[2].set_title(r'$p_* = q_0$')
plt.legend(['BDRE', 'TDRE', 'MDRE', 'TSM'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(f'{FIGURES_DIR}/varying_kl_01.pdf')