import os
import re
import h5py
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict


config = yaml.load(open('experiments/pstar_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract directory paths
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']

# extract experimental sweep parameters
NSAMPLES_PSTAR_VALUES = config['nsamples_pstar_values']
ALGORITHMS = config.get('algorithms', [])

processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

# load ground truth
dataset_filename = f'{DATA_DIR}/dataset.h5'
if not os.path.exists(dataset_filename):
    print(f"Error: {dataset_filename} not found. Run step1 first.")
    exit(1)

with h5py.File(dataset_filename, 'r') as f:
    true_ldrs_arr = f['true_ldrs_arr'][:]  # [20, 1024]

# discover and parse result files
pattern = r'^(.+?)_nsamples_pstar_(\d+)\.h5$'
result_files = {}

for filepath in Path(RAW_RESULTS_DIR).glob('*.h5'):
    match = re.match(pattern, filepath.name)
    if match:
        method = match.group(1)
        nsamples_pstar = int(match.group(2))
        result_files[(method, nsamples_pstar)] = filepath
    else:
        print(f"Warning: skipping malformed file {filepath.name}")

print(f"Discovered {len(result_files)} result files")

# initialize nested dicts
est_ldrs_by_alg = defaultdict(dict)
timing_by_alg = defaultdict(dict)
peak_memory_by_alg = defaultdict(dict)

# load metrics for all (algorithm, nsamples_pstar) pairs
for alg in ALGORITHMS:
    for nsamples_pstar in NSAMPLES_PSTAR_VALUES:
        key = (alg, nsamples_pstar)

        if key in result_files:
            filepath = result_files[key]
            try:
                with h5py.File(filepath, 'r') as f:
                    est_ldrs_by_alg[alg][nsamples_pstar] = f['est_ldrs_arr'][:]
                    timing_by_alg[alg][nsamples_pstar] = f['timing_arr'][:]
                    peak_memory_by_alg[alg][nsamples_pstar] = f['peak_memory'][()]
            except (KeyError, OSError) as e:
                print(f"Warning: error reading {filepath}: {e}, using NaN")
                est_ldrs_by_alg[alg][nsamples_pstar] = np.full_like(true_ldrs_arr, np.nan)
                timing_by_alg[alg][nsamples_pstar] = np.full(true_ldrs_arr.shape[0], np.nan)
                peak_memory_by_alg[alg][nsamples_pstar] = np.nan
        else:
            print(f"Warning: no results for {alg} nsamples_pstar={nsamples_pstar}, using NaN")
            est_ldrs_by_alg[alg][nsamples_pstar] = np.full_like(true_ldrs_arr, np.nan)
            timing_by_alg[alg][nsamples_pstar] = np.full(true_ldrs_arr.shape[0], np.nan)
            peak_memory_by_alg[alg][nsamples_pstar] = np.nan


def compute_metrics(est_ldrs, true_ldrs, timing_arr):
    """Aggregate metrics for a single (method, nsamples_pstar) pair.

    Compute MAE per instance, then report mean and std across instances.
    This captures instance-to-instance variability for proper error bars.

    args:
        est_ldrs: [num_instances, num_test] estimated log density ratios
        true_ldrs: [num_instances, num_test] ground truth log density ratios
        timing_arr: [num_instances] timing measurements in seconds

    returns:
        dict with keys: mae, mae_std, timing_mean, timing_std (all scalars)
    """
    # compute mae per instance: [num_instances]
    abs_errors = np.abs(est_ldrs - true_ldrs)  # [num_instances, num_test]
    mae_per_instance = np.mean(abs_errors, axis=1)  # [num_instances]

    # aggregate across instances
    mae = np.mean(mae_per_instance)
    mae_std = np.std(mae_per_instance, ddof=1)  # sample std for proper error bars

    # timing statistics
    timing_mean = np.mean(timing_arr)
    timing_std = np.std(timing_arr, ddof=1)

    return {
        'mae': mae,
        'mae_std': mae_std,
        'timing_mean': timing_mean,
        'timing_std': timing_std
    }


# compute metrics for all (algorithm, nsamples_pstar) pairs
metrics = {alg: {} for alg in ALGORITHMS}

for alg in ALGORITHMS:
    for nsamples_pstar in NSAMPLES_PSTAR_VALUES:
        est_ldrs = est_ldrs_by_alg[alg][nsamples_pstar]
        timing_arr = timing_by_alg[alg][nsamples_pstar]

        metrics[alg][nsamples_pstar] = compute_metrics(est_ldrs, true_ldrs_arr, timing_arr)

        # log what was processed
        print(f"{alg:30s} nsamples_pstar={nsamples_pstar:5d}: "
              f"mae={metrics[alg][nsamples_pstar]['mae']:.6f} "
              f"timing={metrics[alg][nsamples_pstar]['timing_mean']:.3f}s")

# aggregate metrics into arrays
aggregated_metrics = {}
for alg in ALGORITHMS:
    aggregated_metrics[alg] = {
        'mae': np.array([metrics[alg][ns]['mae'] for ns in NSAMPLES_PSTAR_VALUES], dtype=np.float32),
        'mae_std': np.array([metrics[alg][ns]['mae_std'] for ns in NSAMPLES_PSTAR_VALUES], dtype=np.float32),
        'timing_mean': np.array([metrics[alg][ns]['timing_mean'] for ns in NSAMPLES_PSTAR_VALUES], dtype=np.float32),
        'timing_std': np.array([metrics[alg][ns]['timing_std'] for ns in NSAMPLES_PSTAR_VALUES], dtype=np.float32),
        'peak_memory': np.array([peak_memory_by_alg[alg][ns] for ns in NSAMPLES_PSTAR_VALUES], dtype=np.float32),
    }

# validation before save
for alg in ALGORITHMS:
    mae_arr = aggregated_metrics[alg]['mae']
    assert len(mae_arr) == len(NSAMPLES_PSTAR_VALUES), \
        f"Algorithm {alg}: mae array length {len(mae_arr)} != {len(NSAMPLES_PSTAR_VALUES)}"

    # check for all-NaN arrays
    if np.all(np.isnan(mae_arr)):
        print(f"Warning: {alg} has all-NaN mae values; possible data loading error")

# create output directory
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)

# write to hdf5
with h5py.File(processed_results_filename, 'w') as out_file:
    # store the canonical nsamples_pstar_values axis
    out_file.create_dataset('nsamples_pstar_values',
                            data=np.array(NSAMPLES_PSTAR_VALUES, dtype=np.float32))

    # store metrics for each algorithm
    for alg in ALGORITHMS:
        for metric_name in ['mae', 'mae_std', 'timing_mean', 'timing_std', 'peak_memory']:
            arr = aggregated_metrics[alg][metric_name]
            dataset_name = f'{metric_name}_{alg}'
            out_file.create_dataset(dataset_name, data=arr)

# summary report
print("\n" + "="*80)
print(f"Processed results saved to {processed_results_filename}")
print(f"  - nsamples_pstar_values: {NSAMPLES_PSTAR_VALUES}")
print(f"  - algorithms: {ALGORITHMS}")
print(f"  - metrics per algorithm: {list(aggregated_metrics[ALGORITHMS[0]].keys())}")
print("="*80)
