import os
import pickle

from scipy.stats import multivariate_normal

from experiments.utils.two_gaussians_kl import create_two_gaussians_kl_range


KL_DISTANCES = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
NSAMPLES = 1000

DATA_DIR = 'experiments/density_ratio_estimation/data'
os.makedirs(DATA_DIR, exist_ok=True)
for kl_distance in KL_DISTANCES:
    gaussian_pairs = create_two_gaussians_kl_range(dim=3, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=100)
    data = []
    for gaussian_pair in gaussian_pairs:
        p0 = multivariate_normal(gaussian_pair['mu0'], gaussian_pair['Sigma0'])
        p1 = multivariate_normal(gaussian_pair['mu1'], gaussian_pair['Sigma1'])
        samples_p0 = p0.rvs(NSAMPLES)
        samples_p1 = p1.rvs(NSAMPLES)
        data.append(dict(samples_p0=samples_p0, samples_p1=samples_p1))

    pickle.dump(data, open(f'{DATA_DIR}/k={kl_distance}.pkl', 'wb'))