import os
import pickle

import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from experiments.utils.two_gaussians_kl import create_two_gaussians_kl_range


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
SEED = config['seed']
NUM_INSTANCES = config['num_instances']
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = 'experiments/density_ratio_estimation/data'
os.makedirs(DATA_DIR, exist_ok=True)
for kl_distance in tqdm(KL_DISTANCES):
    gaussian_pairs = create_two_gaussians_kl_range(dim=DATA_DIM, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=NUM_INSTANCES)
    datasets = []
    for gaussian_pair in gaussian_pairs:
        mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
        mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
        # define the two distributions whose density ratios we want to estimate
        p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
        p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
        # define test distributions
        pstar1 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
        pstar2 = MultivariateNormal(mu1, covariance_matrix=Sigma1)
        mu_star3 = (mu1 + mu0) * 0.5
        Sigma_star3 = torch.sqrt(Sigma1)  # in our setup, this is "halfway" between Sigma0 and Sigma1
        pstar3 = MultivariateNormal(mu_star3, covariance_matrix=Sigma_star3)
        # draw samples from training distributions
        samples_p0 = p0.sample((NSAMPLES_TRAIN,))
        samples_p1 = p1.sample((NSAMPLES_TRAIN,))
        # draw samples from test distributions
        samples_pstar1 = pstar1.sample((NSAMPLES_TEST,))
        samples_pstar2 = pstar2.sample((NSAMPLES_TEST,))
        samples_pstar3 = pstar3.sample((NSAMPLES_TEST,))
        # store all samples in a dictionary
        datasets.append(dict(
            mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1,
            samples_p0=samples_p0, samples_p1=samples_p1,
            samples_pstar1=samples_pstar1, samples_pstar2=samples_pstar2, samples_pstar3=samples_pstar3
        ))

    pickle.dump(datasets, open(f'{DATA_DIR}/d={DATA_DIM},k={kl_distance},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.pkl', 'wb'))