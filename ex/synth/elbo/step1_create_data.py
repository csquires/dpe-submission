import os

import h5py  # HDF5
import yaml
from tqdm import trange
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from ex.utils.prescribed_eigs import create_prior_eig_range, create_design_eig
from ex.utils.fractional_posterior import get_fractional_posterior


config = yaml.load(open('ex/synth/elbo/config1.yaml', 'r'), Loader=yaml.FullLoader)
# directories
DATA_DIR = config['data_dir']
# dataset parameters
DATA_DIM = config['data_dim']
EIG_MIN = config['eig_min']
EIG_MAX = config['eig_max']
NUM_PRIORS = config['num_priors']
NUM_DESIGNS_PER_SETTING = config['num_designs_per_setting']
DESIGN_EIG_PERCENTAGES = config['design_eig_percentages']
ALPHAS = config['alphas']
NSAMPLES = config['nsamples']
N_ALPHAS = len(ALPHAS)
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

nrows = NUM_PRIORS * len(DESIGN_EIG_PERCENTAGES) * NUM_DESIGNS_PER_SETTING * len(ALPHAS)
design_eig_percentage_arr = np.zeros((nrows, 1), dtype=np.float32)
alpha_arr = np.zeros((nrows, 1), dtype=np.float32)
# priors
prior_mean_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
prior_covariance_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
# variational posteriors
mu_q_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
Sigma_q_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
# observed pair of design and y
design_arr = np.zeros((nrows, DATA_DIM, 1), dtype=np.float32)
obs_y_arr = np.zeros((nrows, 1), dtype=np.float32)
# data
theta0_samples_arr = np.zeros((nrows, NSAMPLES, DATA_DIM), dtype=np.float32)
y0_samples_arr = np.zeros((nrows, NSAMPLES, 1), dtype=np.float32)
theta1_samples_arr = np.zeros((nrows, NSAMPLES, DATA_DIM), dtype=np.float32)
y1_samples_arr = np.zeros((nrows, NSAMPLES, 1), dtype=np.float32)
theta_star_samples_arr = np.zeros((nrows, NSAMPLES, DATA_DIM), dtype=np.float32)
y_star_samples_arr = np.zeros((nrows, NSAMPLES, 1), dtype=np.float32)


def sample_uniform_over_sphere(dim: int) -> torch.Tensor:
    theta_star_unnormalized = MultivariateNormal(torch.zeros(dim), covariance_matrix=torch.eye(dim)).sample((1,))
    theta_star = theta_star_unnormalized / torch.norm(theta_star_unnormalized)
    return theta_star.squeeze()


idx = 0
for _ in trange(NUM_PRIORS):
    mu_pi, Sigma_pi = create_prior_eig_range(dim=DATA_DIM, eig_min=EIG_MIN, eig_max=EIG_MAX)
    prior_mvn = MultivariateNormal(mu_pi, covariance_matrix=Sigma_pi)

    for design_eig_percentage in DESIGN_EIG_PERCENTAGES:
        desired_eig = EIG_MAX * design_eig_percentage

        for _ in range(NUM_DESIGNS_PER_SETTING):
            obs_xi = create_design_eig(mu_pi, Sigma_pi, desired_eig, sigma=1.0)
            theta_star = sample_uniform_over_sphere(DATA_DIM)
            obs_y = MultivariateNormal(obs_xi.T @ theta_star, covariance_matrix=torch.eye(1)).sample((1,))

            # alpha-independent draws: prior-induced joint and prior-predictive marginal
            theta0_samples = prior_mvn.sample((NSAMPLES,))
            y0_samples = theta0_samples @ obs_xi + torch.randn(NSAMPLES, 1)
            hidden_theta1_samples = prior_mvn.sample((NSAMPLES,))
            y1_samples = hidden_theta1_samples @ obs_xi + torch.randn(NSAMPLES, 1)

            # stack variational posteriors across alpha to enable batched MVN sampling
            mu_q_stack = torch.zeros(N_ALPHAS, DATA_DIM)
            Sigma_q_stack = torch.zeros(N_ALPHAS, DATA_DIM, DATA_DIM)
            for a_i, alpha in enumerate(ALPHAS):
                mu_q_a, Sigma_q_a = get_fractional_posterior(mu_pi, Sigma_pi, obs_xi, obs_y, alpha)
                mu_q_stack[a_i] = mu_q_a
                Sigma_q_stack[a_i] = Sigma_q_a

            # batched draws from variational posteriors: (N_ALPHAS, NSAMPLES, DATA_DIM)
            q_mvn = MultivariateNormal(mu_q_stack, covariance_matrix=Sigma_q_stack)
            theta1_stack = q_mvn.sample((NSAMPLES,)).transpose(0, 1)
            theta_star_stack = q_mvn.sample((NSAMPLES,)).transpose(0, 1)

            theta0_np = theta0_samples.numpy()
            y0_np = y0_samples.numpy()
            y1_np = y1_samples.numpy()
            mu_pi_np = mu_pi.numpy()
            Sigma_pi_np = Sigma_pi.numpy()
            obs_xi_np = obs_xi.numpy()

            for a_i, alpha in enumerate(ALPHAS):
                # store alpha and beta
                design_eig_percentage_arr[idx] = design_eig_percentage
                alpha_arr[idx] = alpha
                # store prior and variational posteriors
                prior_mean_arr[idx] = mu_pi_np
                prior_covariance_arr[idx] = Sigma_pi_np
                mu_q_arr[idx] = mu_q_stack[a_i].numpy()
                Sigma_q_arr[idx] = Sigma_q_stack[a_i].numpy()
                # store observed pair of design and y
                design_arr[idx] = obs_xi_np
                obs_y_arr[idx] = obs_y
                # store data
                theta0_samples_arr[idx] = theta0_np
                y0_samples_arr[idx] = y0_np
                theta1_samples_arr[idx] = theta1_stack[a_i].numpy()
                y1_samples_arr[idx] = y1_np
                theta_star_samples_arr[idx] = theta_star_stack[a_i].numpy()
                y_star_samples_arr[idx] = obs_y
                idx += 1


os.makedirs(DATA_DIR, exist_ok=True)
dataset_filename = f'{DATA_DIR}/dataset_d={DATA_DIM},nsamples={NSAMPLES}.h5'
with h5py.File(dataset_filename, 'w') as f:
    # metadata
    f.create_dataset('design_eig_percentage_arr', data=design_eig_percentage_arr)
    f.create_dataset('alpha_arr', data=alpha_arr)
    # prior parameters
    f.create_dataset('prior_mean_arr', data=prior_mean_arr)
    f.create_dataset('prior_covariance_arr', data=prior_covariance_arr)
    # variational posteriors
    f.create_dataset('mu_q_arr', data=mu_q_arr)
    f.create_dataset('Sigma_q_arr', data=Sigma_q_arr)
    # observed pair
    f.create_dataset('design_arr', data=design_arr)
    f.create_dataset('obs_y_arr', data=obs_y_arr)
    # data
    f.create_dataset('theta0_samples_arr', data=theta0_samples_arr)
    f.create_dataset('y0_samples_arr', data=y0_samples_arr)
    f.create_dataset('theta1_samples_arr', data=theta1_samples_arr)
    f.create_dataset('y1_samples_arr', data=y1_samples_arr)
    f.create_dataset('theta_star_samples_arr', data=theta_star_samples_arr)
    f.create_dataset('y_star_samples_arr', data=y_star_samples_arr)