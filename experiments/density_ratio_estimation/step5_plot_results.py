import os
import pickle

import yaml
import matplotlib.pyplot as plt
import seaborn as sns


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
PROCESSED_RESULTS_DIR = config['processed_results_dir']
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NUM_INSTANCES = config['num_instances']

results_df = pickle.load(open(f'{PROCESSED_RESULTS_DIR}/results_d={DATA_DIM}.pkl', 'rb'))
mean_results_df = results_df.groupby(["kl_distance", "test_set_idx", "algorithm"]).mean()
y_min = mean_results_df["mae"].min()
y_max = mean_results_df["mae"].max()

# setup
plt.clf()
sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=3)
# plotting
for i in range(3):
    bdre_results = mean_results_df.loc[KL_DISTANCES, i, 'bdre']
    axes[i].plot(KL_DISTANCES, bdre_results, label='BDRE')
    # axes[i].plot(KL_DISTANCES, results[test_set]['tdre'], label='TDRE')
    # axes[i].plot(KL_DISTANCES, results[test_set]['mdre'], label='MDRE')
    # axes[i].plot(KL_DISTANCES, results[test_set]['tsm'], label='TSM')
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
os.makedirs('experiments/density_ratio_estimation/figures', exist_ok=True)
plt.savefig('experiments/density_ratio_estimation/figures/varying_kl_01.pdf')