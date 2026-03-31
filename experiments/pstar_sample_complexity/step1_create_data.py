import os
import h5py
import yaml
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from experiments.utils.prescribed_kls import create_two_gaussians_kl


# load config
config = yaml.load(open('experiments/pstar_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract constants
DATA_DIR = config['data_dir']
DATA_DIM = config['data_dim']
NSAMPLES_P0_P1 = config['nsamples_p0_p1']
NSAMPLES_PSTAR_VALUES = config['nsamples_pstar_values']
MAX_NSAMPLES_PSTAR = max(NSAMPLES_PSTAR_VALUES)
NSAMPLES_TEST = config['nsamples_test']
NUM_INSTANCES = config['num_instances']
SEED = config['seed']
N_STD_PADDING = config['n_std_padding']

# set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_bounding_box(mu0, Sigma0, mu1, Sigma1, n_std=3.0):
    """
    compute bounding box containing both Gaussians with margin.

    args:
        mu0: torch.Tensor [dim] - mean of p0
        Sigma0: torch.Tensor [dim, dim] - covariance of p0
        mu1: torch.Tensor [dim] - mean of p1
        Sigma1: torch.Tensor [dim, dim] - covariance of p1
        n_std: float - number of standard deviations for margin

    returns:
        (lower, upper): tuple of torch.Tensor [dim]

    procedure:
        1. extract std devs from diagonal of covariance matrices
        2. compute bounds for p0: mu0 +/- n_std * std0
        3. compute bounds for p1: mu1 +/- n_std * std1
        4. take element-wise min/max to get union of bounds
        5. return (lower, upper)
    """
    std0 = torch.sqrt(torch.diag(Sigma0))  # [dim]
    std1 = torch.sqrt(torch.diag(Sigma1))  # [dim]

    lower0 = mu0 - n_std * std0  # [dim]
    upper0 = mu0 + n_std * std0  # [dim]

    lower1 = mu1 - n_std * std1  # [dim]
    upper1 = mu1 + n_std * std1  # [dim]

    lower = torch.minimum(lower0, lower1)  # [dim]
    upper = torch.maximum(upper0, upper1)  # [dim]

    return (lower, upper)


def sample_uniform_box(lower, upper, n_samples):
    """
    sample uniformly from hyperrectangle [lower, upper]^d.

    args:
        lower: torch.Tensor [dim]
        upper: torch.Tensor [dim]
        n_samples: int

    returns:
        torch.Tensor [n_samples, dim]

    procedure:
        1. extract dimension
        2. sample uniform on [0, 1]
        3. scale to [lower, upper] via affine transformation
        4. return samples
    """
    dim = lower.shape[0]
    u = torch.rand(n_samples, dim)  # [n_samples, dim]
    samples = lower + u * (upper - lower)  # [n_samples, dim]
    return samples


# pre-allocate storage arrays
nrows = NUM_INSTANCES

mu0_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
mu1_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
Sigma0_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
Sigma1_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
samples_p0_arr = np.zeros((nrows, NSAMPLES_P0_P1, DATA_DIM), dtype=np.float32)
samples_p1_arr = np.zeros((nrows, NSAMPLES_P0_P1, DATA_DIM), dtype=np.float32)
samples_pstar_arr = np.zeros((nrows, MAX_NSAMPLES_PSTAR, DATA_DIM), dtype=np.float32)
samples_test_arr = np.zeros((nrows, NSAMPLES_TEST, DATA_DIM), dtype=np.float32)
true_ldrs_arr = np.zeros((nrows, NSAMPLES_TEST), dtype=np.float32)
box_lower_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
box_upper_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)

# generate instances
for idx in tqdm(range(NUM_INSTANCES)):
    # create Gaussian pair with fixed k=1.5, beta=0.5
    gaussian_pair = create_two_gaussians_kl(dim=DATA_DIM, k=1.5, beta=0.5)
    mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
    mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']

    # define the distributions
    p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
    p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

    # store Gaussian parameters as metadata
    mu0_arr[idx] = mu0.numpy()  # [DATA_DIM]
    mu1_arr[idx] = mu1.numpy()  # [DATA_DIM]
    Sigma0_arr[idx] = Sigma0.numpy()  # [DATA_DIM, DATA_DIM]
    Sigma1_arr[idx] = Sigma1.numpy()  # [DATA_DIM, DATA_DIM]

    # compute bounding box
    lower, upper = compute_bounding_box(mu0, Sigma0, mu1, Sigma1, n_std=N_STD_PADDING)
    box_lower_arr[idx] = lower.numpy()  # [DATA_DIM]
    box_upper_arr[idx] = upper.numpy()  # [DATA_DIM]

    # sample from training distributions
    samples_p0_arr[idx] = p0.sample((NSAMPLES_P0_P1,)).numpy()  # [NSAMPLES_P0_P1, DATA_DIM]
    samples_p1_arr[idx] = p1.sample((NSAMPLES_P0_P1,)).numpy()  # [NSAMPLES_P0_P1, DATA_DIM]

    # sample from pstar (uniform box) - use max to allow subsampling in step2
    samples_pstar = sample_uniform_box(lower, upper, MAX_NSAMPLES_PSTAR)
    samples_pstar_arr[idx] = samples_pstar.numpy()  # [MAX_NSAMPLES_PSTAR, DATA_DIM]

    # generate test samples
    samples_test = sample_uniform_box(lower, upper, NSAMPLES_TEST)
    samples_test_arr[idx] = samples_test.numpy()  # [NSAMPLES_TEST, DATA_DIM]

    # compute true log density ratios
    true_ldrs = p0.log_prob(samples_test) - p1.log_prob(samples_test)
    true_ldrs_arr[idx] = true_ldrs.numpy()  # [NSAMPLES_TEST]


# save to HDF5
os.makedirs(DATA_DIR, exist_ok=True)
with h5py.File(f'{DATA_DIR}/dataset.h5', 'w') as f:
    f.create_dataset('mu0_arr', data=mu0_arr)
    f.create_dataset('mu1_arr', data=mu1_arr)
    f.create_dataset('Sigma0_arr', data=Sigma0_arr)
    f.create_dataset('Sigma1_arr', data=Sigma1_arr)
    f.create_dataset('samples_p0_arr', data=samples_p0_arr)
    f.create_dataset('samples_p1_arr', data=samples_p1_arr)
    f.create_dataset('samples_pstar_arr', data=samples_pstar_arr)
    f.create_dataset('samples_test_arr', data=samples_test_arr)
    f.create_dataset('true_ldrs_arr', data=true_ldrs_arr)
    f.create_dataset('box_lower_arr', data=box_lower_arr)
    f.create_dataset('box_upper_arr', data=box_upper_arr)

# print summary statistics
print(f"Dataset saved to {DATA_DIR}/dataset.h5")
print(f"  - num_instances: {NUM_INSTANCES}")
print(f"  - samples_train per instance: {NSAMPLES_P0_P1}")
print(f"  - samples_test per instance: {NSAMPLES_TEST}")
print(f"  - data_dim: {DATA_DIM}")
print(f"  - total training samples: {NUM_INSTANCES * NSAMPLES_P0_P1 * DATA_DIM}")
print(f"  - total test samples: {NUM_INSTANCES * NSAMPLES_TEST * DATA_DIM}")
