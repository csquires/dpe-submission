import os

import h5py
import numpy as np
import yaml
from pathlib import Path
import re
from collections import defaultdict


config = yaml.load(open('experiments/hidden_dim_scaling/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract directory paths
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']

# extract experimental sweep parameters
HIDDEN_DIMS = config['hidden_dims']
ALGORITHMS = config.get('algorithms', ['TSM', 'TriangularMDRE'])

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

# load true eigs from dedicated file
true_eigs_file = f'{RAW_RESULTS_DIR}/true_eigs.h5'
if not os.path.exists(true_eigs_file):
    print(f"Error: {true_eigs_file} not found. Run step2 first.")
    exit(1)

with h5py.File(true_eigs_file, 'r') as f:
    true_eigs_arr = f['true_eigs'][:]

# discover and parse granular result files
pattern = r'^(.+?)_hidden_dim_(\d+)\.h5$'
result_files = {}

for filepath in Path(RAW_RESULTS_DIR).glob('*.h5'):
    if filepath.name == 'true_eigs.h5':
        continue
    match = re.match(pattern, filepath.name)
    if match:
        method = match.group(1)
        dim = int(match.group(2))
        result_files[(method, dim)] = filepath
    else:
        print(f"Warning: skipping malformed file {filepath.name}")

print(f"Discovered {len(result_files)} result files")

# initialize nested dicts using defaultdict for cleaner code
est_eigs_by_alg = defaultdict(dict)
timing_by_alg = defaultdict(dict)
peak_memory_by_alg = defaultdict(dict)
param_count_by_alg = defaultdict(dict)

# load metrics for all (algorithm, hidden_dim) pairs
for alg in ALGORITHMS:
    for dim in HIDDEN_DIMS:
        key = (alg, dim)

        if key in result_files:
            filepath = result_files[key]
            try:
                with h5py.File(filepath, 'r') as f:
                    est_eigs_by_alg[alg][dim] = f['est_eigs_arr'][:]
                    timing_by_alg[alg][dim] = f['timing_arr'][:]
                    peak_memory_by_alg[alg][dim] = f['peak_memory'][()]
                    param_count_by_alg[alg][dim] = f['param_count'][()]
            except (KeyError, OSError) as e:
                print(f"Warning: error reading {filepath}: {e}, using NaN")
                est_eigs_by_alg[alg][dim] = np.full_like(true_eigs_arr, np.nan)
                timing_by_alg[alg][dim] = np.full_like(true_eigs_arr, np.nan)
                peak_memory_by_alg[alg][dim] = np.nan
                param_count_by_alg[alg][dim] = np.nan
        else:
            print(f"Warning: no results for {alg} hidden_dim={dim}, using NaN")
            est_eigs_by_alg[alg][dim] = np.full_like(true_eigs_arr, np.nan)
            timing_by_alg[alg][dim] = np.full_like(true_eigs_arr, np.nan)
            peak_memory_by_alg[alg][dim] = np.nan
            param_count_by_alg[alg][dim] = np.nan


def compute_metrics(est_eigs, true_eigs, timing_arr):
    """aggregate metrics for a single (alg, hidden_dim) pair.

    compute mae and std of absolute errors across instances.
    each instance has one EIG value, so abs_errors is already per-instance.

    args:
        est_eigs: (num_instances,) estimated eig values
        true_eigs: (num_instances,) true eig values
        timing_arr: (num_instances,) timing measurements in seconds

    returns:
        dict: {
            'mae': float - mean absolute error across instances,
            'std': float - sample std of absolute errors (ddof=1),
            'timing_mean': float,
            'timing_std': float
        }
    """
    # compute absolute errors (already per-instance for EIG)
    abs_errors = np.abs(est_eigs - true_eigs)

    # mae: mean of absolute errors
    mae = np.mean(abs_errors)

    # std: sample standard deviation across instances
    std = np.std(abs_errors, ddof=1)

    # timing statistics
    timing_mean = np.mean(timing_arr)
    timing_std = np.std(timing_arr, ddof=1)

    return {
        'mae': mae,
        'std': std,
        'timing_mean': timing_mean,
        'timing_std': timing_std
    }


# compute metrics for all (algorithm, hidden_dim) pairs
metrics = {alg: {} for alg in ALGORITHMS}

for alg in ALGORITHMS:
    for hidden_dim in HIDDEN_DIMS:
        est_eigs = est_eigs_by_alg[alg][hidden_dim]
        timing_arr = timing_by_alg[alg][hidden_dim]

        metrics[alg][hidden_dim] = compute_metrics(est_eigs, true_eigs_arr, timing_arr)

        # log what was processed
        print(f"{alg:20s} hidden_dim={hidden_dim:3d}: "
              f"mae={metrics[alg][hidden_dim]['mae']:.6f} "
              f"timing={metrics[alg][hidden_dim]['timing_mean']:.3f}s")

# aggregate metrics into arrays
aggregated_metrics = {}
for alg in ALGORITHMS:
    aggregated_metrics[alg] = {
        'mae': np.array([metrics[alg][hd]['mae'] for hd in HIDDEN_DIMS], dtype=np.float32),
        'std': np.array([metrics[alg][hd]['std'] for hd in HIDDEN_DIMS], dtype=np.float32),
        'timing_mean': np.array([metrics[alg][hd]['timing_mean'] for hd in HIDDEN_DIMS], dtype=np.float32),
        'timing_std': np.array([metrics[alg][hd]['timing_std'] for hd in HIDDEN_DIMS], dtype=np.float32),
        'peak_memory': np.array([peak_memory_by_alg[alg][hd] for hd in HIDDEN_DIMS], dtype=np.float32),
        'param_count': np.array([param_count_by_alg[alg][hd] for hd in HIDDEN_DIMS], dtype=np.float32),
    }

# validation before save
for alg in ALGORITHMS:
    mae_arr = aggregated_metrics[alg]['mae']
    assert len(mae_arr) == len(HIDDEN_DIMS), \
        f"Algorithm {alg}: mae array length {len(mae_arr)} != {len(HIDDEN_DIMS)}"

    # check for all-NaN arrays
    if np.all(np.isnan(mae_arr)):
        print(f"Warning: {alg} has all-NaN mae values; possible data loading error")

# create output directory
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)

# write to hdf5
with h5py.File(processed_results_filename, 'w') as out_file:
    # store hidden_dims axis
    out_file.create_dataset('hidden_dims', data=np.array(HIDDEN_DIMS, dtype=np.float32))

    # store metrics for each algorithm
    for alg in ALGORITHMS:
        for metric_name in ['mae', 'std', 'timing_mean', 'timing_std', 'peak_memory', 'param_count']:
            arr = aggregated_metrics[alg][metric_name]
            dataset_name = f'{metric_name}_{alg}'
            out_file.create_dataset(dataset_name, data=arr)

# summary report
print("\n" + "="*80)
print(f"Processed results saved to {processed_results_filename}")
print(f"  - hidden_dims: {HIDDEN_DIMS}")
print(f"  - algorithms: {ALGORITHMS}")
print(f"  - metrics per algorithm: {list(aggregated_metrics[ALGORITHMS[0]].keys())}")
print("="*80)
