import os

import h5py  # HDF5
import yaml
from tqdm import tqdm
import numpy as np
import torch
from torch.distributions import MultivariateNormal, Cauchy, Independent

from experiments.utils.prescribed_kls import create_two_gaussians_kl_range


config = yaml.load(open('experiments/density_ratio_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
# dataset parameters
DATA_DIM = config['data_dim']
GAMMA = config['gamma']
KL_DISTANCES = config['kl_distances']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

# dataset storage
nrows = len(KL_DISTANCES) * NUM_INSTANCES_PER_KL
kl_distance_arr = np.zeros(nrows, dtype=np.float32)
# true parameters (metadata)
mu0_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
mu1_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
Sigma0_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
Sigma1_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
cauchy_loc = torch.zeros(DATA_DIM)
cauchy_scale = torch.ones(DATA_DIM) * GAMMA
# data itself
samples_p0_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
samples_p1_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
samples_pstar_arr = np.zeros((nrows, NTEST_SETS, NSAMPLES_TEST, DATA_DIM), dtype=np.float32)
samples_pstar_train_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
# true density ratios
true_ldrs_arr = np.zeros((nrows, NTEST_SETS, NSAMPLES_TEST), dtype=np.float32)

idx = 0
for kl_distance in tqdm(KL_DISTANCES):
    gaussian_pairs = create_two_gaussians_kl_range(dim=DATA_DIM, k=kl_distance, beta_min=0.3, beta_max=0.7, npairs=NUM_INSTANCES_PER_KL)
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
        Sigma_star3 = torch.sqrt(Sigma0)  # in our setup, this is "halfway" between Sigma0 and Sigma1
        pstar3 = MultivariateNormal(mu_star3, covariance_matrix=Sigma_star3)
        # add a Cauchy distribution for more distinct data
        pstar4 = Independent(Cauchy(loc=cauchy_loc, scale=cauchy_scale), 1)

        # store parameters
        kl_distance_arr[idx] = kl_distance
        mu0_arr[idx] = mu0.numpy()
        mu1_arr[idx] = mu1.numpy()
        Sigma0_arr[idx] = Sigma0.numpy()
        Sigma1_arr[idx] = Sigma1.numpy()

        # draw and store samples from training distributions
        samples_p0_arr[idx] = p0.sample((NSAMPLES_TRAIN,)).numpy()
        samples_p1_arr[idx] = p1.sample((NSAMPLES_TRAIN,)).numpy()
        # draw and store samples from test distributions
        samples_pstar1 = pstar1.sample((NSAMPLES_TEST,))
        samples_pstar2 = pstar2.sample((NSAMPLES_TEST,))
        samples_pstar3 = pstar3.sample((NSAMPLES_TEST,))
        samples_pstar4 = pstar4.sample((NSAMPLES_TEST,))
        samples_pstar_arr[idx] = torch.stack([samples_pstar1, samples_pstar2, samples_pstar3, samples_pstar4], dim=0).numpy()

        samples_pstar_train = pstar4.sample((NSAMPLES_TRAIN,))
        samples_pstar_train_arr[idx] = samples_pstar_train.numpy()

        # compute true ldrs
        true_ldrs1 = p0.log_prob(samples_pstar1) - p1.log_prob(samples_pstar1)
        true_ldrs2 = p0.log_prob(samples_pstar2) - p1.log_prob(samples_pstar2)
        true_ldrs3 = p0.log_prob(samples_pstar3) - p1.log_prob(samples_pstar3)
        true_ldrs4 = p0.log_prob(samples_pstar4) - p1.log_prob(samples_pstar4)
        true_ldrs_arr[idx] = torch.stack([true_ldrs1, true_ldrs2, true_ldrs3, true_ldrs4], dim=0).numpy()

        idx += 1


os.makedirs(DATA_DIR, exist_ok=True)
# with h5py.File(f'{DATA_DIR}/dataset.h5', 'w') as f:
with h5py.File(f'{DATA_DIR}/dataset_d={DATA_DIM},ntrain={NSAMPLES_TRAIN},ntest={NSAMPLES_TEST}.h5', 'w') as f:
    f.create_dataset('kl_distance_arr', data=kl_distance_arr)
    f.create_dataset('mu0_arr', data=mu0_arr)
    f.create_dataset('mu1_arr', data=mu1_arr)
    f.create_dataset('Sigma0_arr', data=Sigma0_arr)
    f.create_dataset('Sigma1_arr', data=Sigma1_arr)
    f.create_dataset('samples_p0_arr', data=samples_p0_arr)
    f.create_dataset('samples_p1_arr', data=samples_p1_arr)
    f.create_dataset('samples_pstar_arr', data=samples_pstar_arr)
    f.create_dataset('samples_pstar_train_arr', data=samples_pstar_train_arr)
    f.create_dataset('true_ldrs_arr', data=true_ldrs_arr)