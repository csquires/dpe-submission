import os

import matplotlib.pyplot as plt
import seaborn as sns

kl_values = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]


# PLACEHOLDER: should be loaded from experiments/density_ratio_estimation/results.pkl
results = {
    'p0_test': {
        'bdre': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'tdre': [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5],
        'mdre': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'tsm': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    },
    'p1_test': {
        'bdre': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'tdre': [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5],
        'mdre': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'tsm': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    },
    'laplace0_test': {
        'bdre': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'tdre': [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5],
        'mdre': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'tsm': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    },
    'laplace1_test': {
        'bdre': [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
        'tdre': [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5],
        'mdre': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,  0.8, 0.9, 1.0],
        'tsm': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    }
}
test_sets = ['p0_test', 'p1_test', 'laplace0_test', 'laplace1_test']

# setup
plt.clf()
sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=4)
# plotting
for i, test_set in enumerate(test_sets):
    axes[i].plot(kl_values, results[test_set]['bdre'])
    axes[i].plot(kl_values, results[test_set]['tdre'])
    axes[i].plot(kl_values, results[test_set]['mdre'])
    axes[i].plot(kl_values, results[test_set]['tsm'])
    axes[i].set_xscale('log')
    axes[i].set_xlabel(r'KL$(p_0 \| p_1)$')
axes[0].set_ylabel('Density Ratio Error on Test Data')
axes[0].set_title(r'$p_* = p_0$')
axes[1].set_title(r'$p_* = p_1$')
axes[2].set_title(r'$p_* = q_0$')
axes[3].set_title(r'$p_* = q_1$')
plt.legend(['BDRE', 'TDRE', 'MDRE', 'TSM'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs('experiments/density_ratio_estimation/figures', exist_ok=True)
plt.savefig('experiments/density_ratio_estimation/figures/varying_kl_01.pdf')