import os

import h5py
import numpy as np
from scipy import stats
import yaml


config = yaml.load(open('experiments/dre_sample_complexity/config.yaml', 'r'), Loader=yaml.FullLoader)

# directories
DATA_DIR = config['data_dir']
RAW_RESULTS_DIR = config['raw_results_dir']
PROCESSED_RESULTS_DIR = config['processed_results_dir']
# dataset parameters
DATA_DIM = config['data_dim']
KL_DIVERGENCES = config['kl_divergences']
NUM_INSTANCES_PER_KL = config['num_instances_per_kl']
NSAMPLES_TRAIN_VALUES = config['nsamples_train_values']
NSAMPLES_TEST = config['nsamples_test']

raw_results_filename = f'{RAW_RESULTS_DIR}/results.h5'
dataset_filename = f'{DATA_DIR}/dataset.h5'
processed_results_filename = f'{PROCESSED_RESULTS_DIR}/metrics.h5'

with h5py.File(raw_results_filename, 'r') as f:
    result_keys = [key for key in f.keys() if key.startswith('est_ldrs_arr_')]
    # Shape: (nrows, n_nsamples_train, nsamples_test)
    est_ldrs_by_alg = {key.replace('est_ldrs_arr_', ''): f[key][:] for key in result_keys}

with h5py.File(dataset_filename, 'r') as f:
    # Shape: (nrows, nsamples_test)
    true_ldrs_arr = f['true_ldrs_arr'][:]

nrows = true_ldrs_arr.shape[0]
n_kl = len(KL_DIVERGENCES)
n_instances = NUM_INSTANCES_PER_KL
n_nsamples_train = len(NSAMPLES_TRAIN_VALUES)


def compute_spearman_correlation(est_ldrs, true_ldrs):
    """Compute Spearman correlation for each (row, nsamples_train) pair.

    Args:
        est_ldrs: (nrows, n_nsamples_train, nsamples_test)
        true_ldrs: (nrows, nsamples_test)
    Returns:
        (nrows, n_nsamples_train)
    """
    result = np.zeros((nrows, n_nsamples_train))
    for i in range(nrows):
        for j in range(n_nsamples_train):
            corr, _ = stats.spearmanr(est_ldrs[i, j, :], true_ldrs[i, :])
            result[i, j] = corr
    return result


def compute_median_ae(est_ldrs, true_ldrs):
    """Compute Median Absolute Error for each (row, nsamples_train) pair.

    Args:
        est_ldrs: (nrows, n_nsamples_train, nsamples_test)
        true_ldrs: (nrows, nsamples_test)
    Returns:
        (nrows, n_nsamples_train)
    """
    # Broadcast true_ldrs to match est_ldrs shape
    true_ldrs_expanded = true_ldrs[:, np.newaxis, :]  # (nrows, 1, nsamples_test)
    absolute_errors = np.abs(est_ldrs - true_ldrs_expanded)
    return np.median(absolute_errors, axis=2)  # (nrows, n_nsamples_train)


def compute_trimmed_mae_iqr(est_ldrs, true_ldrs):
    """Compute MAE only for points within the IQR of true LDRs.

    Args:
        est_ldrs: (nrows, n_nsamples_train, nsamples_test)
        true_ldrs: (nrows, nsamples_test)
    Returns:
        (nrows, n_nsamples_train)
    """
    result = np.zeros((nrows, n_nsamples_train))
    for i in range(nrows):
        true_vals = true_ldrs[i, :]
        q1, q3 = np.percentile(true_vals, [25, 75])
        mask = (true_vals >= q1) & (true_vals <= q3)
        for j in range(n_nsamples_train):
            est_vals = est_ldrs[i, j, :]
            if mask.sum() > 0:
                result[i, j] = np.mean(np.abs(est_vals[mask] - true_vals[mask]))
            else:
                result[i, j] = np.nan
    return result


def compute_stratified_mae_quartiles(est_ldrs, true_ldrs):
    """Compute MAE stratified by quartiles of true LDR.

    Args:
        est_ldrs: (nrows, n_nsamples_train, nsamples_test)
        true_ldrs: (nrows, nsamples_test)
    Returns:
        dict with keys 'q1', 'q2', 'q3', 'q4', each (nrows, n_nsamples_train)
    """
    quartile_maes = {f'q{q}': np.zeros((nrows, n_nsamples_train)) for q in range(1, 5)}
    for i in range(nrows):
        true_vals = true_ldrs[i, :]
        percentiles = np.percentile(true_vals, [25, 50, 75])
        # Q1: 0-25th percentile
        mask_q1 = true_vals <= percentiles[0]
        # Q2: 25-50th percentile
        mask_q2 = (true_vals > percentiles[0]) & (true_vals <= percentiles[1])
        # Q3: 50-75th percentile
        mask_q3 = (true_vals > percentiles[1]) & (true_vals <= percentiles[2])
        # Q4: 75-100th percentile
        mask_q4 = true_vals > percentiles[2]

        for j in range(n_nsamples_train):
            est_vals = est_ldrs[i, j, :]
            for q, mask in enumerate([mask_q1, mask_q2, mask_q3, mask_q4], 1):
                if mask.sum() > 0:
                    quartile_maes[f'q{q}'][i, j] = np.mean(np.abs(est_vals[mask] - true_vals[mask]))
                else:
                    quartile_maes[f'q{q}'][i, j] = np.nan
    return quartile_maes


# Compute all metrics for each algorithm
# Final shapes: (n_kl, n_instances, n_nsamples_train)
maes_by_alg = {}
median_aes_by_alg = {}
spearman_by_alg = {}
trimmed_mae_iqr_by_alg = {}
stratified_mae_by_alg = {}  # nested: {alg: {q1: ..., q2: ..., q3: ..., q4: ...}}

for alg_name, est_ldrs_arr in est_ldrs_by_alg.items():
    # est_ldrs_arr shape: (nrows, n_nsamples_train, nsamples_test)
    # true_ldrs_arr shape: (nrows, nsamples_test)

    # MAE (original)
    true_ldrs_expanded = true_ldrs_arr[:, np.newaxis, :]  # (nrows, 1, nsamples_test)
    absolute_errors = np.abs(est_ldrs_arr - true_ldrs_expanded)
    maes = np.mean(absolute_errors, axis=2)  # (nrows, n_nsamples_train)
    maes_by_alg[alg_name] = maes.reshape(n_kl, n_instances, n_nsamples_train)

    # Median AE
    median_aes = compute_median_ae(est_ldrs_arr, true_ldrs_arr)
    median_aes_by_alg[alg_name] = median_aes.reshape(n_kl, n_instances, n_nsamples_train)

    # Spearman correlation
    spearman = compute_spearman_correlation(est_ldrs_arr, true_ldrs_arr)
    spearman_by_alg[alg_name] = spearman.reshape(n_kl, n_instances, n_nsamples_train)

    # Trimmed MAE within IQR
    trimmed_mae = compute_trimmed_mae_iqr(est_ldrs_arr, true_ldrs_arr)
    trimmed_mae_iqr_by_alg[alg_name] = trimmed_mae.reshape(n_kl, n_instances, n_nsamples_train)

    # Stratified MAE by quartiles
    strat_maes = compute_stratified_mae_quartiles(est_ldrs_arr, true_ldrs_arr)
    stratified_mae_by_alg[alg_name] = {
        q: arr.reshape(n_kl, n_instances, n_nsamples_train)
        for q, arr in strat_maes.items()
    }

# save results
os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
with h5py.File(processed_results_filename, 'w') as f:
    for alg_name, maes_arr in maes_by_alg.items():
        f.create_dataset(f'maes_{alg_name}', data=maes_arr)
    for alg_name, median_aes_arr in median_aes_by_alg.items():
        f.create_dataset(f'median_aes_{alg_name}', data=median_aes_arr)
    for alg_name, spearman_arr in spearman_by_alg.items():
        f.create_dataset(f'spearman_{alg_name}', data=spearman_arr)
    for alg_name, trimmed_mae_arr in trimmed_mae_iqr_by_alg.items():
        f.create_dataset(f'trimmed_mae_iqr_{alg_name}', data=trimmed_mae_arr)
    for alg_name, strat_dict in stratified_mae_by_alg.items():
        for quartile, arr in strat_dict.items():
            f.create_dataset(f'stratified_mae_{quartile}_{alg_name}', data=arr)

print(f"Processed results saved to {processed_results_filename}")
print(f"  - n_kl: {n_kl}")
print(f"  - n_instances: {n_instances}")
print(f"  - n_nsamples_train: {n_nsamples_train}")
print(f"  - Algorithms: {list(maes_by_alg.keys())}")
