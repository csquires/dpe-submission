"""
Step 2: Run DRE Algorithms for Plugin DRE Experiment

Trains density ratio estimation algorithms on the training samples
and evaluates them on the uniform grid.
"""
import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Force re-run of all algorithms, overwriting existing results')
args = parser.parse_args()

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser


config = yaml.load(open('experiments/plugin_dre/config.yaml', 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RESULTS_DIR = config['results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES_TRAIN = config['nsamples_train']
# algorithm parameters
TDRE_WAYPOINTS = config['tdre_waypoints']
MDRE_WAYPOINTS = config['mdre_waypoints']
# random seed
SEED = config['seed']
np.random.seed(SEED)
torch.manual_seed(SEED)

dataset_filename = f'{DATA_DIR}/dataset.h5'
results_filename = f'{RESULTS_DIR}/raw_results.h5'

# Check existing results
existing_results = set()
if os.path.exists(results_filename):
    with h5py.File(results_filename, 'r') as results_file:
        existing_results = set(results_file.keys())
        print("Existing results for:", list(results_file.keys()))

# Instantiate BDRE
bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM)
bdre = BDRE(bdre_classifier, device=DEVICE)

# Instantiate TDRE variants
tdre_variants = []
for num_waypoints_tdre in TDRE_WAYPOINTS:
    tdre_classifiers = make_pairwise_binary_classifiers(
        name="default",
        num_classes=num_waypoints_tdre,
        input_dim=DATA_DIM,
    )
    tdre_variants.append((f"TDRE_{num_waypoints_tdre}", TDRE(tdre_classifiers, num_waypoints=num_waypoints_tdre, device=DEVICE)))

# Instantiate MDRE variants
mdre_variants = []
for num_waypoints_mdre in MDRE_WAYPOINTS:
    mdre_classifier = make_multiclass_classifier(
        name="default",
        input_dim=DATA_DIM,
        num_classes=num_waypoints_mdre,
    )
    mdre_variants.append((f"MDRE_{num_waypoints_mdre}", MDRE(mdre_classifier, device=DEVICE)))

# Instantiate Spatial
spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM, device=DEVICE)

# List of all algorithms to run
algorithms = [
    ("BDRE", bdre),
    *tdre_variants,
    *mdre_variants,
    ("Spatial", spatial),
]

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['kl_distance_arr'].shape[0]
    num_grid_points = dataset_file['grid_points_arr'].shape[1]

    for alg_name, alg in algorithms:
        dataset_name = f'est_ldrs_grid_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        print(f"\nRunning {alg_name}...")
        est_ldrs_arr = np.zeros((nrows, num_grid_points))

        for idx in trange(nrows, desc=alg_name):
            # Load training samples
            samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][idx]).to(DEVICE)
            samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][idx]).to(DEVICE)

            # Train algorithm
            alg.fit(samples_p0, samples_p1)

            # Evaluate on grid
            grid_points = torch.from_numpy(dataset_file['grid_points_arr'][idx]).to(DEVICE)
            est_ldrs = alg.predict_ldr(grid_points)
            est_ldrs_arr[idx] = est_ldrs.cpu().numpy()

        # Save results
        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_ldrs_arr)

        print(f"  Results saved for {alg_name}")

print(f"\nAll results saved to: {results_filename}")
