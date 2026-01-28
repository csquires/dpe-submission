"""
Step 2b: Merge parallel DRE results for Plugin DRE Experiment

Merges results from multiple parallel tasks into a single results file.
"""
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
RESULTS_DIR = config['results_dir']

# Auto-discover num_tasks if not provided
if args.num_tasks is None:
    task_files = glob.glob(f'{RESULTS_DIR}/raw_results_task_*.h5')
    num_tasks = len(task_files)
    if num_tasks == 0:
        raise FileNotFoundError(f"No task files found in {RESULTS_DIR}")
else:
    num_tasks = args.num_tasks

# Read metadata from task 0
with h5py.File(f'{RESULTS_DIR}/raw_results_task_0.h5', 'r') as f:
    nrows_total = f.attrs['nrows_total']
    num_grid_points = f.attrs['num_grid_points']
    alg_names = json.loads(f.attrs['algorithm_names'])

print(f"Merging {num_tasks} task files")
print(f"Total datasets: {nrows_total}")
print(f"Grid points per dataset: {num_grid_points}")
print(f"Algorithms: {alg_names}")

# Verify all task files exist
for task_id in range(num_tasks):
    task_file = f'{RESULTS_DIR}/raw_results_task_{task_id}.h5'
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Missing task file: {task_file}")

# Output file (main results file)
output_file = f'{RESULTS_DIR}/raw_results.h5'

with h5py.File(output_file, 'w') as out_f:
    # Merge each algorithm
    for alg_name in alg_names:
        dataset_name = f'est_ldrs_grid_{alg_name}'
        full_arr = np.zeros((nrows_total, num_grid_points))

        for task_id in range(num_tasks):
            with h5py.File(f'{RESULTS_DIR}/raw_results_task_{task_id}.h5', 'r') as f:
                start_idx = f.attrs['start_idx']
                end_idx = f.attrs['end_idx']
                full_arr[start_idx:end_idx] = f[dataset_name][:]

        out_f.create_dataset(dataset_name, data=full_arr)
        print(f"Merged {dataset_name}: shape {full_arr.shape}")

print(f"Saved merged results to {output_file}")
