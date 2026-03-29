import os

import h5py
import numpy as np
import yaml


config = yaml.load(open('experiments/hidden_dim_scaling/config.yaml', 'r'), Loader=yaml.FullLoader)

# extract directory paths
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']

# extract experimental sweep parameters
HIDDEN_DIMS = config['hidden_dims']
ALGORITHMS = config.get('algorithms', ['TSM', 'TriangularMDRE'])

raw_results_filename = f'{RAW_RESULTS_DIR}/results.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

# load true eig values
with h5py.File(raw_results_filename, 'r') as f:
    true_eigs_arr = f['true_eigs_arr'][:]

# load raw estimates for all (algorithm, hidden_dim) pairs
est_eigs_by_alg_dim = {}
for alg in ALGORITHMS:
    est_eigs_by_alg_dim[alg] = {}
    for hidden_dim in HIDDEN_DIMS:
        key = f'est_eigs_arr_{alg}_hidden_dim_{hidden_dim}'
        try:
            with h5py.File(raw_results_filename, 'r') as f:
                if key in f:
                    est_eigs_by_alg_dim[alg][hidden_dim] = f[key][:]
                else:
                    print(f"Warning: {key} not found in {raw_results_filename}, skipping")
                    est_eigs_by_alg_dim[alg][hidden_dim] = np.full_like(true_eigs_arr, np.nan)
        except Exception as e:
            print(f"Warning: Error loading {key}: {e}")
            est_eigs_by_alg_dim[alg][hidden_dim] = np.full_like(true_eigs_arr, np.nan)

# load timing arrays
timing_by_alg_dim = {}
for alg in ALGORITHMS:
    timing_by_alg_dim[alg] = {}
    for hidden_dim in HIDDEN_DIMS:
        key = f'timing_arr_{alg}_hidden_dim_{hidden_dim}'
        try:
            with h5py.File(raw_results_filename, 'r') as f:
                if key in f:
                    timing_by_alg_dim[alg][hidden_dim] = f[key][:]
                else:
                    print(f"Warning: {key} not found in {raw_results_filename}, skipping")
                    timing_by_alg_dim[alg][hidden_dim] = np.array([np.nan])
        except Exception as e:
            print(f"Warning: Error loading {key}: {e}")
            timing_by_alg_dim[alg][hidden_dim] = np.array([np.nan])

# load peak memory and parameter count
peak_memory_by_alg_dim = {}
param_count_by_alg_dim = {}
for alg in ALGORITHMS:
    peak_memory_by_alg_dim[alg] = {}
    param_count_by_alg_dim[alg] = {}
    for hidden_dim in HIDDEN_DIMS:
        peak_mem_key = f'peak_memory_{alg}_hidden_dim_{hidden_dim}'
        param_count_key = f'param_count_{alg}_hidden_dim_{hidden_dim}'

        try:
            with h5py.File(raw_results_filename, 'r') as f:
                if peak_mem_key in f:
                    peak_memory_by_alg_dim[alg][hidden_dim] = f[peak_mem_key][()]
                else:
                    print(f"Warning: {peak_mem_key} not found")
                    peak_memory_by_alg_dim[alg][hidden_dim] = np.nan

                if param_count_key in f:
                    param_count_by_alg_dim[alg][hidden_dim] = f[param_count_key][()]
                else:
                    print(f"Warning: {param_count_key} not found")
                    param_count_by_alg_dim[alg][hidden_dim] = np.nan
        except Exception as e:
            print(f"Warning: Error loading scalars for {alg} hidden_dim {hidden_dim}: {e}")
            peak_memory_by_alg_dim[alg][hidden_dim] = np.nan
            param_count_by_alg_dim[alg][hidden_dim] = np.nan


def compute_metrics(est_eigs, true_eigs, timing_arr):
    """aggregate metrics for a single (alg, hidden_dim) pair.

    compute mae and std of absolute errors from estimates vs. true values.
    compute timing mean and std from raw timing array.

    args:
        est_eigs: (100,) estimated eig values
        true_eigs: (100,) true eig values
        timing_arr: (100,) timing measurements in seconds

    returns:
        dict: {
            'mae': float or nan,
            'std': float or nan,
            'timing_mean': float or nan,
            'timing_std': float or nan
        }
    """
    # compute absolute errors
    abs_errors = np.abs(est_eigs - true_eigs)

    # mae: mean of absolute errors
    mae = np.mean(abs_errors)

    # std: standard deviation of absolute errors
    std = np.std(abs_errors)

    # timing statistics
    timing_mean = np.mean(timing_arr)
    timing_std = np.std(timing_arr)

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
        est_eigs = est_eigs_by_alg_dim[alg][hidden_dim]
        timing_arr = timing_by_alg_dim[alg][hidden_dim]

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
        'peak_memory': np.array([peak_memory_by_alg_dim[alg][hd] for hd in HIDDEN_DIMS], dtype=np.float32),
        'param_count': np.array([param_count_by_alg_dim[alg][hd] for hd in HIDDEN_DIMS], dtype=np.float32),
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
