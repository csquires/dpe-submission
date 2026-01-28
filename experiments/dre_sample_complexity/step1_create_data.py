import os

import h5py
import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from experiments.utils.prescribed_kls import create_two_gaussians_kl_range


config = yaml.load(open('experiments/dre_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
# dataset parameters
DATA_DIM = config['data_dim']
GAMMA = config['gamma']
KL_DISTANCES = config['kl_distances']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
NSAMPLES_TRAIN_VALUES = config['nsamples_train_values']
NSAMPLES_TEST = config['nsamples_test']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use max training samples to allow subsampling later
MAX_NSAMPLES_TRAIN = max(NSAMPLES_TRAIN_VALUES)

# dataset storage
nrows = len(KL_DISTANCES) * NUM_INSTANCES_PER_KL
kl_distance_arr = np.zeros(nrows, dtype=np.float32)
# true parameters (metadata)
mu0_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
mu1_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
Sigma0_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
Sigma1_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
# data itself - store max samples for subsampling
samples_p0_arr = np.zeros((nrows, MAX_NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
samples_p1_arr = np.zeros((nrows, MAX_NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
# only use q_1 (pstar4) as test distribution
samples_pstar_arr = np.zeros((nrows, NSAMPLES_TEST, DATA_DIM), dtype=np.float32)
# true density ratios for q_1 only
true_ldrs_arr = np.zeros((nrows, NSAMPLES_TEST), dtype=np.float32)

idx = 0
for kl_distance in tqdm(KL_DISTANCES):
    gaussian_pairs = create_two_gaussians_kl_range(dim=DATA_DIM, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=NUM_INSTANCES_PER_KL)
    for gaussian_pair in gaussian_pairs:
        mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
        mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']
        # define the two distributions whose density ratios we want to estimate
        p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
        p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

        # define test distribution: only q_1 (pstar4 - distant Gaussian)
        mu_star = -mu1
        Sigma_star = Sigma0 @ Sigma0
        pstar = MultivariateNormal(mu_star, covariance_matrix=Sigma_star)

        # store parameters
        kl_distance_arr[idx] = kl_distance
        mu0_arr[idx] = mu0.numpy()
        mu1_arr[idx] = mu1.numpy()
        Sigma0_arr[idx] = Sigma0.numpy()
        Sigma1_arr[idx] = Sigma1.numpy()

        # draw and store samples from training distributions (max samples)
        samples_p0_arr[idx] = p0.sample((MAX_NSAMPLES_TRAIN,)).numpy()
        samples_p1_arr[idx] = p1.sample((MAX_NSAMPLES_TRAIN,)).numpy()

        # draw and store samples from test distribution (q_1 only)
        samples_pstar = pstar.sample((NSAMPLES_TEST,))
        samples_pstar_arr[idx] = samples_pstar.numpy()

        # compute true ldrs for q_1
        true_ldrs = p0.log_prob(samples_pstar) - p1.log_prob(samples_pstar)
        true_ldrs_arr[idx] = true_ldrs.numpy()

        idx += 1


os.makedirs(DATA_DIR, exist_ok=True)
with h5py.File(f'{DATA_DIR}/dataset.h5', 'w') as f:
    f.create_dataset('kl_distance_arr', data=kl_distance_arr)
    f.create_dataset('mu0_arr', data=mu0_arr)
    f.create_dataset('mu1_arr', data=mu1_arr)
    f.create_dataset('Sigma0_arr', data=Sigma0_arr)
    f.create_dataset('Sigma1_arr', data=Sigma1_arr)
    f.create_dataset('samples_p0_arr', data=samples_p0_arr)
    f.create_dataset('samples_p1_arr', data=samples_p1_arr)
    f.create_dataset('samples_pstar_arr', data=samples_pstar_arr)
    f.create_dataset('true_ldrs_arr', data=true_ldrs_arr)

print(f"Dataset saved to {DATA_DIR}/dataset.h5")
print(f"  - nrows: {nrows}")
print(f"  - max training samples: {MAX_NSAMPLES_TRAIN}")
print(f"  - test samples: {NSAMPLES_TEST}")
print(f"  - KL distances: {KL_DISTANCES}")
