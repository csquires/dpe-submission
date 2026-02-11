"""
Step 2a: Run DRE Algorithms for Plugin DRE Experiment (Parallel Version)

Trains density ratio estimation algorithms on the training samples
and evaluates them on the uniform grid. Processes a subset of datasets
for parallel execution.
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import trange
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--task-id', type=int, default=None,
                    help='Task ID. If not provided, uses SLURM_ARRAY_TASK_ID')
parser.add_argument('--num-tasks', type=int, required=True,
                    help='Total number of tasks')
parser.add_argument('--config', type=str, required=True,
                    help='Path to config YAML')
parser.add_argument('--force', action='store_true',
                    help='Force re-run of all algorithms, overwriting existing results')
args = parser.parse_args()

# Get task ID from env or arg
task_id = args.task_id if args.task_id is not None else int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))

from src.models.binary_classification import make_binary_classifier, make_pairwise_binary_classifiers
from src.models.multiclass_classification import make_multiclass_classifier
from src.density_ratio_estimation.bdre import BDRE
from src.density_ratio_estimation.tdre import TDRE
from src.density_ratio_estimation.mdre import MDRE
from src.density_ratio_estimation.tsm import TSM
from src.density_ratio_estimation.triangular_mdre import TriangularMDRE
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
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
np.random.seed(SEED + task_id)  # Different seed per task for reproducibility
torch.manual_seed(SEED + task_id)

dataset_filename = f'{DATA_DIR}/dataset.h5'
results_filename = f'{RESULTS_DIR}/raw_results_task_{task_id}.h5'

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

# Instantiate Spatial (VFM)
spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM, device=DEVICE)

# Instantiate TSM
tsm = TSM(DATA_DIM, device=DEVICE)

# Instantiate TriangularMDRE
triangular_mdre_waypoints = 15
triangular_mdre_classifier = make_multiclass_classifier(
    name="default",
    input_dim=DATA_DIM,
    num_classes=triangular_mdre_waypoints,
)
triangular_mdre = TriangularMDRE(
    triangular_mdre_classifier,
    device=DEVICE,
    midpoint_oversample=7,
    gamma_power=3.0,
)

# List of all algorithms to run
algorithms = [
    ("BDRE", bdre),
    *tdre_variants,
    *mdre_variants,
    ("TSM", tsm),
    ("TriangularMDRE", triangular_mdre),
    ("VFM", spatial),
]

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Open dataset file in read-only mode (safe for concurrent access)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['kl_divergence_arr'].shape[0]
    num_grid_points = dataset_file['grid_points_arr'].shape[1]

    # Compute dataset range for this task
    datasets_per_task = math.ceil(nrows / args.num_tasks)
    start_idx = task_id * datasets_per_task
    end_idx = min((task_id + 1) * datasets_per_task, nrows)
    job_nrows = end_idx - start_idx

    if job_nrows <= 0:
        print(f"Task {task_id}: No datasets to process (nrows={nrows}, num_tasks={args.num_tasks})")
        sys.exit(0)

    print(f"Task {task_id}: Processing datasets {start_idx} to {end_idx-1} ({job_nrows} datasets)")

    # Check for existing results in task file
    existing_results = set()
    if os.path.exists(results_filename):
        with h5py.File(results_filename, 'r') as results_file:
            existing_results = set(results_file.keys())
            print(f"Task {task_id}: Existing results for: {list(results_file.keys())}")

    for alg_name, alg in algorithms:
        dataset_name = f'est_ldrs_grid_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Task {task_id}: Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        print(f"\nTask {task_id}: Running {alg_name}...")
        est_ldrs_arr = np.zeros((job_nrows, num_grid_points))

        for local_idx, global_idx in enumerate(trange(start_idx, end_idx, desc=f"Task {task_id} - {alg_name}")):
            # Load training samples
            samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][global_idx]).to(DEVICE)
            samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][global_idx]).to(DEVICE)

            # Train algorithm (special handling for TriangularMDRE)
            if alg_name in {"TriangularMDRE"}:
                alg.fit(samples_p0, samples_p1, samples_p0)  # use p0 as pstar
            else:
                alg.fit(samples_p0, samples_p1)

            # Evaluate on grid
            grid_points = torch.from_numpy(dataset_file['grid_points_arr'][global_idx]).to(DEVICE)
            est_ldrs = alg.predict_ldr(grid_points)
            est_ldrs_arr[local_idx] = est_ldrs.cpu().numpy()

        # Save results
        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_ldrs_arr)

        print(f"  Task {task_id}: Results saved for {alg_name}")

    # Store metadata for step2b
    alg_names = [name for name, _ in algorithms]
    with h5py.File(results_filename, 'a') as f:
        f.attrs['task_id'] = task_id
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['num_tasks'] = args.num_tasks
        f.attrs['nrows_total'] = nrows
        f.attrs['num_grid_points'] = num_grid_points
        f.attrs['algorithm_names'] = json.dumps(alg_names)

print(f"\nTask {task_id}: Completed. Results saved to {results_filename}")
