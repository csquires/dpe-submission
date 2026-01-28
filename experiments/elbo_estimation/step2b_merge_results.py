import argparse
import glob
import json
import os

import h5py
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='Path to config YAML')
parser.add_argument('--num-tasks', type=int, default=None,
                    help='Number of tasks. If not provided, auto-discovers from files')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
RAW_RESULTS_DIR = config['raw_results_dir']
DATA_DIM = config['data_dim']
NSAMPLES = config['nsamples']

# Auto-discover num_tasks if not provided
if args.num_tasks is None:
    task_files = glob.glob(f'{RAW_RESULTS_DIR}/results_task_*.h5')
    num_tasks = len(task_files)
    if num_tasks == 0:
        raise FileNotFoundError(f"No task files found in {RAW_RESULTS_DIR}")
else:
    num_tasks = args.num_tasks

# Read metadata from task 0
with h5py.File(f'{RAW_RESULTS_DIR}/results_task_0.h5', 'r') as f:
    nrows_total = f.attrs['nrows_total']
    alg_names = json.loads(f.attrs['algorithm_names'])

print(f"Merging {num_tasks} task files")
print(f"Total datasets: {nrows_total}")
print(f"Algorithms: {alg_names}")

# Verify all task files exist
for task_id in range(num_tasks):
    task_file = f'{RAW_RESULTS_DIR}/results_task_{task_id}.h5'
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Missing task file: {task_file}")

# Output file (main results file)
output_file = f'{RAW_RESULTS_DIR}/results_d={DATA_DIM},nsamples={NSAMPLES}.h5'

with h5py.File(output_file, 'w') as out_f:
    # Merge each algorithm
    for alg_name in alg_names:
        dataset_name = f'est_eldrs_arr_{alg_name}'
        full_arr = np.zeros(nrows_total)

        for task_id in range(num_tasks):
            with h5py.File(f'{RAW_RESULTS_DIR}/results_task_{task_id}.h5', 'r') as f:
                start_idx = f.attrs['start_idx']
                end_idx = f.attrs['end_idx']
                full_arr[start_idx:end_idx] = f[dataset_name][:]

        out_f.create_dataset(dataset_name, data=full_arr)
        print(f"Merged {dataset_name}: shape {full_arr.shape}")

print(f"Saved merged results to {output_file}")
