import os

import h5py
import numpy as np
import yaml
from pathlib import Path
import re
from collections import defaultdict


config = yaml.load(open('experiments/dre_hidden_dim_scaling/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract directory paths
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']

# extract experimental sweep parameters
HIDDEN_DIMS = config['hidden_dims']
ALGORITHMS = config.get('algorithms', ['TSM', 'TriangularMDRE'])

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

# load true ldrs from data file
data_file = f'{DATA_DIR}/dataset.h5'
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found. Run step1 first.")
    exit(1)

with h5py.File(data_file, 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]


def select_best_trials(grouped_trials, true_ldrs_arr):
    """select the trial with lowest mean MAE for each (method, dim) pair.

    for each (method, dim) in grouped_trials, load est_ldrs_arr from all trials,
    compute mean MAE against true_ldrs_arr, and return the best trial info.
    ties are broken by lowest trial_num (sorted earlier).

    args:
        grouped_trials: dict (method, dim) -> [(trial_num, filepath), ...]
        true_ldrs_arr: (num_instances, num_test) ground truth log density ratios

    returns:
        dict: (method, dim) -> {
            'filepath': Path,
            'trial_num': int,
            'learning_rate': float,
            'mean_mae': float
        }
    """
    best_trials = {}

    for (method, dim), trials in grouped_trials.items():
        if not trials:
            continue

        best_trial_info = None
        best_mae = float('inf')

        for trial_num, filepath in trials:
            try:
                with h5py.File(filepath, 'r') as f:
                    est_ldrs = f['est_ldrs_arr'][:]
                    lr = f['learning_rate'][()]

                    # compute mean MAE for this trial
                    abs_errors = np.abs(est_ldrs - true_ldrs_arr)
                    mae_per_instance = np.mean(abs_errors, axis=1)
                    mean_mae = np.mean(mae_per_instance)

                    if mean_mae < best_mae:
                        best_mae = mean_mae
                        best_trial_info = {
                            'filepath': filepath,
                            'trial_num': trial_num,
                            'learning_rate': lr,
                            'mean_mae': mean_mae
                        }
            except (KeyError, OSError) as e:
                print(f"Warning: error loading trial {trial_num} for ({method}, {dim}): {e}")
                continue

        if best_trial_info is not None:
            best_trials[(method, dim)] = best_trial_info

    return best_trials

# discover and parse granular result files
pattern = r'^(.+?)_hidden_dim_(\d+)_trial_(\d+)\.h5$'
grouped_trials = defaultdict(list)  # (method, dim) -> [(trial_num, filepath), ...]

for filepath in Path(RAW_RESULTS_DIR).glob('*.h5'):
    match = re.match(pattern, filepath.name)
    if match:
        method = match.group(1)
        dim = int(match.group(2))
        trial_num = int(match.group(3))
        grouped_trials[(method, dim)].append((trial_num, filepath))
    else:
        print(f"Warning: skipping malformed file {filepath.name}")

# sort trials by trial_num for deterministic tie-breaking
for key in grouped_trials:
    grouped_trials[key].sort(key=lambda x: x[0])

num_trials_found = sum(len(v) for v in grouped_trials.values())
print(f"Discovered {num_trials_found} trial files across {len(grouped_trials)} (algorithm, dim) pairs")

# select best trial for each (method, dim) pair
best_trials = select_best_trials(grouped_trials, true_ldrs_arr)
print(f"Selected best trials for {len(best_trials)} (algorithm, dim) pairs")

# initialize nested dicts using defaultdict for cleaner code
est_ldrs_by_alg = defaultdict(dict)
timing_by_alg = defaultdict(dict)
peak_memory_by_alg = defaultdict(dict)
param_count_by_alg = defaultdict(dict)
best_lr_by_alg = defaultdict(dict)  # alg -> dim -> float
best_trial_num_by_alg = defaultdict(dict)  # alg -> dim -> int (for diagnostics)

# load metrics for all (algorithm, hidden_dim) pairs
for alg in ALGORITHMS:
    for dim in HIDDEN_DIMS:
        key = (alg, dim)

        if key in best_trials:
            trial_info = best_trials[key]
            filepath = trial_info['filepath']
            trial_num = trial_info['trial_num']
            learning_rate = trial_info['learning_rate']

            try:
                with h5py.File(filepath, 'r') as f:
                    est_ldrs_by_alg[alg][dim] = f['est_ldrs_arr'][:]
                    timing_by_alg[alg][dim] = f['timing_arr'][:]
                    peak_memory_by_alg[alg][dim] = f['peak_memory'][()]
                    param_count_by_alg[alg][dim] = f['param_count'][()]

                best_lr_by_alg[alg][dim] = learning_rate
                best_trial_num_by_alg[alg][dim] = trial_num

            except (KeyError, OSError) as e:
                print(f"Warning: error reading {filepath}: {e}, using NaN")
                est_ldrs_by_alg[alg][dim] = np.full_like(true_ldrs_arr, np.nan)
                timing_by_alg[alg][dim] = np.full(20, np.nan, dtype=np.float32)
                peak_memory_by_alg[alg][dim] = np.nan
                param_count_by_alg[alg][dim] = np.nan
                best_lr_by_alg[alg][dim] = np.nan
                best_trial_num_by_alg[alg][dim] = -1
        else:
            print(f"Warning: no trials found for {alg} hidden_dim={dim}, using NaN")
            est_ldrs_by_alg[alg][dim] = np.full_like(true_ldrs_arr, np.nan)
            timing_by_alg[alg][dim] = np.full(20, np.nan, dtype=np.float32)
            peak_memory_by_alg[alg][dim] = np.nan
            param_count_by_alg[alg][dim] = np.nan
            best_lr_by_alg[alg][dim] = np.nan
            best_trial_num_by_alg[alg][dim] = -1


def compute_metrics(est_ldrs, true_ldrs, timing_arr):
    """aggregate metrics for a single (alg, hidden_dim) pair.

    compute mae per instance, then report mean and std across instances.
    this captures instance-to-instance variability for proper error bars.

    args:
        est_ldrs: (num_instances, num_test) estimated log density ratios
        true_ldrs: (num_instances, num_test) true log density ratios
        timing_arr: (num_instances,) timing measurements in seconds

    returns:
        dict: {
            'mae': float - mean of per-instance MAEs,
            'std': float - std of per-instance MAEs (ddof=1),
            'timing_mean': float,
            'timing_std': float
        }
    """
    # compute mae per instance: [num_instances]
    abs_errors = np.abs(est_ldrs - true_ldrs)  # [num_instances, num_test]
    mae_per_instance = np.mean(abs_errors, axis=1)  # [num_instances]

    # aggregate across instances
    mae = np.mean(mae_per_instance)
    std = np.std(mae_per_instance, ddof=1)  # sample std for proper error bars

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
        est_ldrs = est_ldrs_by_alg[alg][hidden_dim]
        timing_arr = timing_by_alg[alg][hidden_dim]

        metrics[alg][hidden_dim] = compute_metrics(est_ldrs, true_ldrs_arr, timing_arr)

        # log what was processed
        print(f"{alg:20s} hidden_dim={hidden_dim:3d}: "
              f"mae={metrics[alg][hidden_dim]['mae']:.6f} "
              f"timing={metrics[alg][hidden_dim]['timing_mean']:.3f}s")

# print best learning rates summary table
print("\n" + "="*80)
print("Best learning rate per (method, hidden_dim):")
print("="*80)

# construct table: rows = algorithms, columns = hidden_dims
header = "Method" + "".join(f"{hd:>8d}" for hd in HIDDEN_DIMS)
print(header)
print("-" * len(header))

for alg in ALGORITHMS:
    row = f"{alg:<20s}"
    for hd in HIDDEN_DIMS:
        lr = best_lr_by_alg[alg][hd]
        if np.isnan(lr):
            row += "     NaN "
        else:
            row += f"{lr:8.4f}"
    print(row)

print("="*80 + "\n")

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
        'best_lr': np.array([best_lr_by_alg[alg][hd] for hd in HIDDEN_DIMS], dtype=np.float32),
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
        for metric_name in ['mae', 'std', 'timing_mean', 'timing_std', 'peak_memory', 'param_count', 'best_lr']:
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
