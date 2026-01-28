"""
Step 2a: Run DRE Algorithms for Sample Complexity Experiment (Parallel Version)

Processes a subset of dataset rows for parallel execution.
Parallelization is across instances (nrows), with sample_size loop running serial.
"""
import argparse
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm
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
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DISTANCES = config['kl_distances']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
NSAMPLES_TRAIN_VALUES = config['nsamples_train_values']
NSAMPLES_TEST = config['nsamples_test']
# random seed
SEED = config['seed']
np.random.seed(SEED + task_id)  # Different seed per task for reproducibility
torch.manual_seed(SEED + task_id)

dataset_filename = f'{DATA_DIR}/dataset.h5'
results_filename = f'{RAW_RESULTS_DIR}/results_task_{task_id}.h5'


def make_algorithms():
    """Create fresh algorithm instances."""
    # instantiate bdre
    bdre_classifier = make_binary_classifier(name="default", input_dim=DATA_DIM)
    bdre = BDRE(bdre_classifier, device=DEVICE)

    # instantiate tdre variants
    tdre_waypoints = [5]
    tdre_variants = []
    for num_waypoints_tdre in tdre_waypoints:
        tdre_classifiers = make_pairwise_binary_classifiers(
            name="default",
            num_classes=num_waypoints_tdre,
            input_dim=DATA_DIM,
        )
        tdre_variants.append((f"TDRE_{num_waypoints_tdre}", TDRE(tdre_classifiers, num_waypoints=num_waypoints_tdre, device=DEVICE)))

    # instantiate mdre variants
    mdre_waypoints = [15]
    mdre_variants = []
    for num_waypoints_mdre in mdre_waypoints:
        mdre_classifier = make_multiclass_classifier(
            name="default",
            input_dim=DATA_DIM,
            num_classes=num_waypoints_mdre,
        )
        mdre_variants.append((f"MDRE_{num_waypoints_mdre}", MDRE(mdre_classifier, device=DEVICE)))

    # instantiate tsm
    tsm = TSM(DATA_DIM, device=DEVICE)

    # instantiate triangular tsm
    triangular_tsm = TriangularTSM(DATA_DIM, device=DEVICE)

    # instantiate spatial velo denoiser
    spatial = make_spatial_velo_denoiser(input_dim=DATA_DIM, device=DEVICE)

    algorithms = [
        ("BDRE", bdre),
        ("TSM", tsm),
        ("TriangularTSM", triangular_tsm),
        *tdre_variants,
        *mdre_variants,
        ("VFM", spatial),
    ]
    return algorithms


os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

# Open dataset file in read-only mode (safe for concurrent access)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['kl_distance_arr'].shape[0]
    n_nsamples_train = len(NSAMPLES_TRAIN_VALUES)

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

    # Get algorithm names
    algorithms = make_algorithms()
    alg_names = [name for name, _ in algorithms]

    for alg_name in alg_names:
        dataset_name = f'est_ldrs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Task {task_id}: Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        print(f"\nTask {task_id}: Running {alg_name}...")
        # Shape: (job_nrows, n_nsamples_train, nsamples_test)
        est_ldrs_arr = np.zeros((job_nrows, n_nsamples_train, NSAMPLES_TEST))

        for ntrain_idx, nsamples_train in enumerate(NSAMPLES_TRAIN_VALUES):
            print(f"  Task {task_id}: nsamples_train = {nsamples_train}")

            # Create fresh algorithm instance for each sample size
            algorithms = make_algorithms()
            alg = dict(algorithms)[alg_name]

            for local_idx, global_idx in enumerate(tqdm(range(start_idx, end_idx), desc=f"    Task {task_id} instances")):
                # Subsample training data
                samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][global_idx][:nsamples_train]).to(DEVICE)
                samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][global_idx][:nsamples_train]).to(DEVICE)

                # Train algorithm (special handling for TriangularTSM)
                if alg_name == "TriangularTSM":
                    samples_pstar = torch.from_numpy(dataset_file['samples_pstar_arr'][global_idx]).to(DEVICE)
                    alg.fit(samples_p0, samples_p1, samples_pstar)
                else:
                    alg.fit(samples_p0, samples_p1)

                # Predict on test set (only q_1)
                samples_pstar = torch.from_numpy(dataset_file['samples_pstar_arr'][global_idx]).to(DEVICE)
                est_ldrs = alg.predict_ldr(samples_pstar)
                est_ldrs_arr[local_idx, ntrain_idx] = est_ldrs.cpu().numpy()

        with h5py.File(results_filename, 'a') as results_file:
            if dataset_name in results_file:
                del results_file[dataset_name]
            results_file.create_dataset(dataset_name, data=est_ldrs_arr)

    # Store metadata for step2b
    with h5py.File(results_filename, 'a') as f:
        f.attrs['task_id'] = task_id
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['num_tasks'] = args.num_tasks
        f.attrs['nrows_total'] = nrows
        f.attrs['n_nsamples_train'] = n_nsamples_train
        f.attrs['nsamples_test'] = NSAMPLES_TEST
        f.attrs['algorithm_names'] = json.dumps(alg_names)

print(f"\nTask {task_id}: Results saved to {results_filename}")
