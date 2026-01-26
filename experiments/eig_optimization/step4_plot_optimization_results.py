import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')


true_eigs = np.linspace(0, 1, 100)


plt.clf()
plt.plot(true_eigs, true_eigs - true_eigs, label='Perfect')
plt.xlabel(r"EIG$(\xi^*)$")
plt.ylabel(r"EIG$(\xi^*)$ - EIG($\hat{\xi}^*$)")
plt.legend()
plt.tight_layout()
os.makedirs('experiments/eig_optimization/figures', exist_ok=True)
plt.savefig('experiments/eig_optimization/figures/eig_optimization.pdf')