import os

import matplotlib.pyplot as plt
import seaborn as sns

nrounds = [8, 16, 32, 64, 128, 256, 512, 1024]

results = {
    'random': [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.025], 
    'ours': [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.0125],
}


# setup
plt.clf()
sns.set_style('whitegrid')
plt.style.use('our_style.mplstyle')
# plotting
for estimator, errors in results.items():
    plt.plot(nrounds, errors, label=estimator)
plt.xscale('log')
plt.xlabel('Number of Rounds')
plt.ylabel('Estimator Error')
plt.legend()
plt.tight_layout()
# saving
os.makedirs('experiments/experimental_design/figures', exist_ok=True)
plt.savefig('experiments/experimental_design/figures/nrounds_vs_estimator_error.pdf')