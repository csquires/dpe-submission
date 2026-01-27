import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from experiments.eig_estimation.step1_create_data import DESIGN_EIG_PERCENTAGES

sns.set_style('whitegrid')
plt.style.use('full-width.mplstyle')

config = yaml.load(open('experiments/elbo_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
ALPHAS = config['alphas']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']


plt.clf()
fig, axes = plt.subplots(figsize=(10, 3), nrows=1, ncols=len(ALPHAS))
estimation_errors_bdre = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for i, alpha in enumerate(ALPHAS):
    axes[i].plot(DESIGN_EIG_PERCENTAGES, estimation_errors_bdre, label='bdre')
    axes[i].set_title(fr"$\alpha = {alpha}$")

fig.supxlabel(r"$\beta$ (Design Optimality Percentage)")
fig.supylabel(r"ELBO Estimation Error")
handles, labels = axes[0].get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
# saving
os.makedirs('experiments/elbo_estimation/figures', exist_ok=True)
plt.savefig('experiments/elbo_estimation/figures/elbo_estimation.pdf')