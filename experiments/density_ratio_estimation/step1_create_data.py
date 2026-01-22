import os
import pickle

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from experiments.utils.two_gaussians_kl import create_two_gaussians_kl_range


KL_DISTANCES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
NSAMPLES = 1000
SEED = 1729
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = 'experiments/density_ratio_estimation/data'
os.makedirs(DATA_DIR, exist_ok=True)
for kl_distance in KL_DISTANCES:
    gaussian_pairs = create_two_gaussians_kl_range(dim=3, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=100)
    datasets = []
    for gaussian_pair in gaussian_pairs:
        p0 = MultivariateNormal(gaussian_pair['mu0'], covariance_matrix=gaussian_pair['Sigma0'])
        p1 = MultivariateNormal(gaussian_pair['mu1'], covariance_matrix=gaussian_pair['Sigma1'])
        samples_p0 = p0.sample((NSAMPLES,))
        samples_p1 = p1.sample((NSAMPLES,))
        datasets.append(dict(samples_p0=samples_p0, samples_p1=samples_p1))

    pickle.dump(datasets, open(f'{DATA_DIR}/k={kl_distance},n={NSAMPLES}.pkl', 'wb'))
pickle.dump(KL_DISTANCES, open(f'{DATA_DIR}/kl_distances.pkl', 'wb'))