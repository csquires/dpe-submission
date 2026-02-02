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
from src.density_ratio_estimation.triangular_tsm import TriangularTSM
from src.density_ratio_estimation.spatial_adapters import make_spatial_velo_denoiser

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
DEVICE = config['device']
# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
NSAMPLES_TRAIN = config['nsamples_train']
NSAMPLES_TEST = config['nsamples_test']
NTEST_SETS = config['ntest_sets']
# random seed
SEED = config['seed']
np.random.seed(SEED + task_id)  # Different seed per task for reproducibility
torch.manual_seed(SEED + task_id)

dataset_filename = f'{DATA_DIR}/dataset_newpstar.h5'
results_filename = f'{RAW_RESULTS_DIR}/new_pstar_task_{task_id}.h5'

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
    # ("TriangularTSM", triangular_tsm),
    *tdre_variants,  # TDRE_5
    *mdre_variants,  # MDRE_15
    ("VFM", spatial),
]

os.makedirs(RAW_RESULTS_DIR, exist_ok=True)

# Open dataset file in read-only mode (safe for concurrent access)
with h5py.File(dataset_filename, 'r') as dataset_file:
    nrows = dataset_file['kl_distance_arr'].shape[0]

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
        dataset_name = f'est_ldrs_arr_{alg_name}'
        if dataset_name in existing_results and not args.force:
            print(f"Task {task_id}: Skipping {alg_name} (results exist, use --force to overwrite)")
            continue

        est_ldrs_arr = np.zeros((job_nrows, NTEST_SETS, NSAMPLES_TEST))

        for local_idx, global_idx in enumerate(trange(start_idx, end_idx, desc=f"Task {task_id} - {alg_name}")):
            samples_p0 = torch.from_numpy(dataset_file['samples_p0_arr'][global_idx]).to(DEVICE)
            samples_p1 = torch.from_numpy(dataset_file['samples_p1_arr'][global_idx]).to(DEVICE)
            if alg_name in {"TriangularTSM"}:
                pstar_train = torch.from_numpy(dataset_file['samples_pstar_train_arr'][global_idx]).to(DEVICE)
                alg.fit(samples_p0, samples_p1, pstar_train)
            else:
                alg.fit(samples_p0, samples_p1)

            samples_pstar = torch.from_numpy(dataset_file['samples_pstar_arr'][global_idx]).to(DEVICE)
            for test_set_idx in range(NTEST_SETS):
                est_ldrs = alg.predict_ldr(samples_pstar[test_set_idx])
                est_ldrs_arr[local_idx, test_set_idx] = est_ldrs.cpu().numpy()

        # Write to task-specific file
        with h5py.File(results_filename, 'a') as f:
            if dataset_name in f:
                del f[dataset_name]
            f.create_dataset(dataset_name, data=est_ldrs_arr)

    # Store metadata for step2b
    alg_names = [name for name, _ in algorithms]
    with h5py.File(results_filename, 'a') as f:
        f.attrs['task_id'] = task_id
        f.attrs['start_idx'] = start_idx
        f.attrs['end_idx'] = end_idx
        f.attrs['num_tasks'] = args.num_tasks
        f.attrs['nrows_total'] = nrows
        f.attrs['algorithm_names'] = json.dumps(alg_names)

print(f"Task {task_id}: Completed. Results saved to {results_filename}")
