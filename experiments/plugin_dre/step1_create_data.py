"""
Step 1: Create Data for Plugin DRE Experiment

Generates 2D Gaussian pairs for each KL distance, computes evaluation grid,
and stores true log density ratios at each grid point.
"""
import os

import h5py
import yaml
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm

from experiments.utils.prescribed_kls import create_two_gaussians_kl


config = yaml.load(open('experiments/plugin_dre/config.yaml', 'r'), Loader=yaml.FullLoader)

# directories
DATA_DIR = config['data_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DIVERGENCES = config['kl_divergences']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
NSAMPLES_TRAIN = config['nsamples_train']
GRID_SIZE = config['grid_size']
N_STD_PADDING = config['n_std_padding']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_bounding_square(mu0, Sigma0, mu1, Sigma1, n_std=3.0):
    """Compute a square bounding box that covers both Gaussians within n_std."""
    std0 = torch.sqrt(torch.diag(Sigma0))
    std1 = torch.sqrt(torch.diag(Sigma1))

    x_min = min(mu0[0] - n_std * std0[0], mu1[0] - n_std * std1[0]).item()
    x_max = max(mu0[0] + n_std * std0[0], mu1[0] + n_std * std1[0]).item()
    y_min = min(mu0[1] - n_std * std0[1], mu1[1] - n_std * std1[1]).item()
    y_max = max(mu0[1] + n_std * std0[1], mu1[1] + n_std * std1[1]).item()

    # Make it a square
    max_range = max(x_max - x_min, y_max - y_min)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max_range / 2
    return (cx - half, cx + half, cy - half, cy + half)


def create_uniform_grid(bounds, grid_size):
    """Create a uniform grid of points within the bounding box."""
    x_min, x_max, y_min, y_max = bounds
    xs = torch.linspace(x_min, x_max, grid_size)
    ys = torch.linspace(y_min, y_max, grid_size)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid_points


# Dataset storage
nrows = len(KL_DIVERGENCES) * NUM_INSTANCES_PER_KL
num_grid_points = GRID_SIZE * GRID_SIZE

kl_divergence_arr = np.zeros(nrows, dtype=np.float32)
# true parameters (metadata)
mu0_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
mu1_arr = np.zeros((nrows, DATA_DIM), dtype=np.float32)
Sigma0_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
Sigma1_arr = np.zeros((nrows, DATA_DIM, DATA_DIM), dtype=np.float32)
# training samples
samples_p0_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
samples_p1_arr = np.zeros((nrows, NSAMPLES_TRAIN, DATA_DIM), dtype=np.float32)
# grid data
grid_points_arr = np.zeros((nrows, num_grid_points, DATA_DIM), dtype=np.float32)
grid_bounds_arr = np.zeros((nrows, 4), dtype=np.float32)  # x_min, x_max, y_min, y_max
true_ldrs_grid_arr = np.zeros((nrows, num_grid_points), dtype=np.float32)

idx = 0
for kl_divergence in tqdm(KL_DIVERGENCES, desc="Creating data"):
    for instance_idx in range(NUM_INSTANCES_PER_KL):
        # Create Gaussian pair with specified KL divergence
        gaussian_pair = create_two_gaussians_kl(dim=DATA_DIM, k=kl_divergence, beta=0.5)
        mu0, Sigma0 = gaussian_pair['mu0'], gaussian_pair['Sigma0']
        mu1, Sigma1 = gaussian_pair['mu1'], gaussian_pair['Sigma1']

        # Define the two distributions
        p0 = MultivariateNormal(mu0, covariance_matrix=Sigma0)
        p1 = MultivariateNormal(mu1, covariance_matrix=Sigma1)

        # Store parameters
        kl_divergence_arr[idx] = kl_divergence
        mu0_arr[idx] = mu0.numpy()
        mu1_arr[idx] = mu1.numpy()
        Sigma0_arr[idx] = Sigma0.numpy()
        Sigma1_arr[idx] = Sigma1.numpy()

        # Draw training samples
        samples_p0_arr[idx] = p0.sample((NSAMPLES_TRAIN,)).numpy()
        samples_p1_arr[idx] = p1.sample((NSAMPLES_TRAIN,)).numpy()

        # Compute bounding box and create grid
        bounds = compute_bounding_square(mu0, Sigma0, mu1, Sigma1, n_std=N_STD_PADDING)
        grid_points = create_uniform_grid(bounds, GRID_SIZE)

        grid_bounds_arr[idx] = np.array(bounds, dtype=np.float32)
        grid_points_arr[idx] = grid_points.numpy()

        # Compute true log density ratios at grid points
        true_ldrs = p0.log_prob(grid_points) - p1.log_prob(grid_points)
        true_ldrs_grid_arr[idx] = true_ldrs.numpy()

        idx += 1

# Save dataset
os.makedirs(DATA_DIR, exist_ok=True)
dataset_filename = f'{DATA_DIR}/dataset.h5'
with h5py.File(dataset_filename, 'w') as f:
    f.create_dataset('kl_divergence_arr', data=kl_divergence_arr)
    f.create_dataset('mu0_arr', data=mu0_arr)
    f.create_dataset('mu1_arr', data=mu1_arr)
    f.create_dataset('Sigma0_arr', data=Sigma0_arr)
    f.create_dataset('Sigma1_arr', data=Sigma1_arr)
    f.create_dataset('samples_p0_arr', data=samples_p0_arr)
    f.create_dataset('samples_p1_arr', data=samples_p1_arr)
    f.create_dataset('grid_points_arr', data=grid_points_arr)
    f.create_dataset('grid_bounds_arr', data=grid_bounds_arr)
    f.create_dataset('true_ldrs_grid_arr', data=true_ldrs_grid_arr)
    # Also store config values for reference
    f.attrs['grid_size'] = GRID_SIZE
    f.attrs['n_std_padding'] = N_STD_PADDING
    f.attrs['nsamples_train'] = NSAMPLES_TRAIN

print(f"Dataset saved to: {dataset_filename}")
print(f"  - {nrows} instances ({len(KL_DIVERGENCES)} KL distances x {NUM_INSTANCES_PER_KL} instances)")
print(f"  - {GRID_SIZE}x{GRID_SIZE} = {num_grid_points} grid points per instance")
print(f"  - {NSAMPLES_TRAIN} training samples per distribution")
