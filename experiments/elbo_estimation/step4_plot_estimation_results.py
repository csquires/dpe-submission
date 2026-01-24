import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')


true_elbos = np.linspace(0, 1, 100)


plt.clf()
plt.plot(true_elbos, true_elbos, label='Perfect')
plt.xlabel(r"EIG$(\xi^*)$")
plt.ylabel(r"$\widehat{\text{EIG}}(\xi^*)$")
plt.legend()
plt.tight_layout()
os.makedirs('experiments/elbo_estimation/figures', exist_ok=True)
plt.savefig('experiments/elbo_estimation/figures/elbo_estimation.pdf')