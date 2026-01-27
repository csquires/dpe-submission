import os

import matplotlib.pyplot as plt
import seaborn as sns
import yaml




config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
EIG_MIN = config['eig_min']
EIG_MAX = config['eig_max']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']

estimation_errors_bdre = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

plt.clf()
sns.set_style('whitegrid')
plt.style.use('half-width.mplstyle')
plt.plot(DESIGN_EIG_PERCENTAGES, estimation_errors_bdre, label='BDRE')
plt.xlabel(r"$\beta$ (Design Optimality Percentage)")
plt.ylabel(r"EIG Estimation Error")
plt.legend()
plt.tight_layout()
os.makedirs('experiments/eig_estimation/figures', exist_ok=True)
plt.savefig('experiments/eig_estimation/figures/eig_estimation.pdf')