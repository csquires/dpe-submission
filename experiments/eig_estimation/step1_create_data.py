import os

import h5py  # HDF5
import yaml
from tqdm import trange
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from experiments.utils.prescribed_eigs import create_prior_eig_range, create_design_eig


config = yaml.load(open('experiments/eig_estimation/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
# dataset parameters
DATA_DIM = config['data_dim']
EIG_MIN = config['eig_min']
EIG_MAX = config['eig_max']
NUM_PRIORS = config['num_priors']
NUM_DESIGNS_PER_SETTING = config['num_designs_per_setting']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']
NSAMPLES = config['nsamples']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

nrows = NUM_PRIORS * len(DESIGN_EIG_PERCENTAGES) * NUM_DESIGNS_PER_SETTING
# priors
prior_mean_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
prior_covariance_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
# designs
design_eig_percentage_arr = np.zeros((nrows, 1), dtype=np.float32)
design_arr = np.zeros((nrows, DATA_DIM, 1), dtype=np.float32)
# data
theta_samples_arr = np.zeros((nrows, NSAMPLES, DATA_DIM), dtype=np.float32)
y_samples_arr = np.zeros((nrows, NSAMPLES, 1), dtype=np.float32)


idx = 0
for _ in trange(NUM_PRIORS):
    mu_pi, Sigma_pi = create_prior_eig_range(dim=DATA_DIM, eig_min=EIG_MIN, eig_max=EIG_MAX)
    
    for design_eig_percentage in DESIGN_EIG_PERCENTAGES:
        desired_eig = EIG_MAX * design_eig_percentage

        for _ in range(NUM_DESIGNS_PER_SETTING):
            xi = create_design_eig(mu_pi, Sigma_pi, desired_eig, sigma=1.0)

            theta_samples = MultivariateNormal(mu_pi, covariance_matrix=Sigma_pi).sample((NSAMPLES,))
            y_samples = theta_samples @ xi + torch.randn(NSAMPLES, 1)
            
            # store prior
            prior_mean_arr[idx] = mu_pi.numpy()
            prior_covariance_arr[idx] = Sigma_pi.numpy()
            # store design
            design_eig_percentage_arr[idx] = design_eig_percentage
            design_arr[idx] = xi.numpy()
            # store data
            theta_samples_arr[idx] = theta_samples.numpy()
            y_samples_arr[idx] = y_samples.numpy()
            idx += 1


os.makedirs(DATA_DIR, exist_ok=True)
dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM}.h5'
with h5py.File(dataset_filename, 'w') as f:
    f.create_dataset('prior_mean_arr', data=prior_mean_arr)
    f.create_dataset('prior_covariance_arr', data=prior_covariance_arr)
    f.create_dataset('design_eig_percentage_arr', data=design_eig_percentage_arr)
    f.create_dataset('design_arr', data=design_arr)
    f.create_dataset('theta_samples_arr', data=theta_samples_arr)
    f.create_dataset('y_samples_arr', data=y_samples_arr)